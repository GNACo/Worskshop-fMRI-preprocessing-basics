"""Preprocesamiento anatómico sencillo para el taller de rs-fMRI.

El notebook solo necesita llamar ``extraer_craneo_lote`` y
``mostrar_entrada_salida``. Los detalles técnicos quedan encapsulados aquí.
"""

import json
from pathlib import Path
from time import perf_counter

import ants
import antspynet
import ipywidgets as widgets
import nibabel as nib
import numpy as np
import pandas as pd
from IPython.display import HTML, clear_output, display
from matplotlib.colors import ListedColormap
from nilearn.datasets import load_mni152_template
from nilearn import plotting
from scipy.ndimage import binary_fill_holes, gaussian_gradient_magnitude, label


ALGORITMO = (
    "ANTsPyNet T1: reorientación RPI → recorte robusto → "
    "U-Net 3D → aplicación de la máscara cerebral"
)


def _entidad(ruta, prefijo):
    """Obtiene una entidad ``sub-`` o ``ses-`` desde una ruta."""

    return next(
        (parte for parte in Path(ruta).parts if parte.startswith(prefijo)),
        None,
    )


def _prefijo_t1(ruta):
    """Retira el sufijo T1w de un nombre NIfTI."""

    nombre = Path(ruta).name
    for sufijo in ("_T1w.nii.gz", "_T1w.nii"):
        if nombre.endswith(sufijo):
            return nombre[: -len(sufijo)]
    raise ValueError(f"El archivo no tiene un nombre T1w reconocido: {nombre}")


def _rutas_salida(t1_path, output_root):
    """Construye las rutas de los productos anatómicos."""

    sujeto = _entidad(t1_path, "sub-")
    sesion = _entidad(t1_path, "ses-")

    if sujeto is None:
        raise ValueError(f"No se encontró una entidad sub- en: {t1_path}")

    carpeta = Path(output_root) / sujeto
    if sesion is not None:
        carpeta = carpeta / sesion
    carpeta = carpeta / "anat"
    carpeta.mkdir(parents=True, exist_ok=True)

    prefijo = _prefijo_t1(t1_path)

    return sujeto, sesion, {
        "reorient": carpeta / f"{prefijo}_desc-reorient_T1w.nii.gz",
        "crop": carpeta / f"{prefijo}_desc-crop_T1w.nii.gz",
        "probability": carpeta / f"{prefijo}_desc-bet_probseg.nii.gz",
        "mask": carpeta / f"{prefijo}_desc-bet_mask.nii.gz",
        "brain": carpeta / f"{prefijo}_desc-bet_T1w.nii.gz",
    }


def _metricas_mascara(mask):
    """Calcula métricas pequeñas de control de calidad."""

    mascara = mask.numpy() > 0
    fraccion_fov = float(mascara.mean())
    volumen_ml = float(mascara.sum() * np.prod(mask.spacing) / 1000)
    return fraccion_fov, volumen_ml


def extraer_craneo_t1(
    t1_path,
    output_root,
    overwrite=False,
    verbose=False,
):
    """Extrae el cerebro de un T1w y devuelve sus rutas principales.

    Orden del algoritmo:
    1. Reorientar a RPI.
    2. Recortar cuello y fondo mediante una máscara robusta.
    3. Estimar una probabilidad cerebral con la U-Net T1 de ANTsPyNet.
    4. Aplicar la salida cerebral como máscara al T1 recortado.

    El último paso reproduce el comportamiento del script de referencia que
    funcionó con estos datos. No se aplica ``fill_holes`` porque, en un campo
    de visión muy recortado, puede convertir la máscara en todo el volumen.
    """

    t1_path = Path(t1_path)
    sujeto, sesion, salidas = _rutas_salida(t1_path, output_root)
    productos = tuple(salidas.values())

    if all(ruta.exists() for ruta in productos) and not overwrite:
        mask = ants.image_read(str(salidas["mask"]))
        fraccion_fov, volumen_ml = _metricas_mascara(mask)
        return {
            "subject": sujeto,
            "session": sesion,
            "algorithm": ALGORITMO,
            "input_path": t1_path,
            "brain_path": salidas["brain"],
            "mask_path": salidas["mask"],
            "status": "reutilizado",
            "seconds": 0.0,
            "brain_volume_ml": round(volumen_ml, 1),
            "mask_fov_pct": round(fraccion_fov * 100, 1),
        }

    existentes = [ruta for ruta in productos if ruta.exists()]
    if existentes and not overwrite:
        raise RuntimeError(
            f"Hay resultados incompletos para {sujeto}. "
            "Ejecuta una vez con overwrite=True."
        )

    inicio = perf_counter()
    t1 = ants.image_read(str(t1_path))

    if t1.dimension != 3:
        raise ValueError(f"El T1w de {sujeto} no es tridimensional.")

    # Estos dos pasos provienen del flujo que funcionó previamente.
    t1_rpi = ants.reorient_image2(t1, orientation="RPI")
    crop_mask = ants.get_mask(
        t1_rpi,
        low_thresh=float(t1_rpi.mean()),
        cleanup=2,
    )

    if np.count_nonzero(crop_mask.numpy()) == 0:
        raise RuntimeError(f"No se pudo calcular el recorte para {sujeto}.")

    t1_crop = ants.crop_image(t1_rpi, crop_mask)

    probability = antspynet.brain_extraction(
        t1_crop,
        modality="t1",
        verbose=verbose,
    )

    # Reproduce el BET del script original: mask_image conserva el nivel 1
    # de la salida de ANTsPyNet. Después generamos una máscara binaria para QC.
    brain = ants.mask_image(t1_crop, probability)
    brain_max = float(brain.max())

    if brain_max <= 0:
        raise RuntimeError(
            f"ANTsPyNet produjo una extracción vacía para {sujeto}."
        )

    mask = ants.threshold_image(brain, 1e-8, brain_max, 1, 0)

    fraccion_fov, volumen_ml = _metricas_mascara(mask)

    # Evita aceptar silenciosamente una máscara vacía o de todo el volumen.
    if fraccion_fov <= 0.005 or fraccion_fov >= 0.95:
        raise RuntimeError(
            f"Máscara no válida para {sujeto}: ocupa "
            f"{fraccion_fov * 100:.1f}% del campo de visión."
        )

    ants.image_write(t1_rpi, str(salidas["reorient"]))
    ants.image_write(t1_crop, str(salidas["crop"]))
    ants.image_write(probability, str(salidas["probability"]))
    ants.image_write(mask, str(salidas["mask"]))
    ants.image_write(brain, str(salidas["brain"]))

    return {
        "subject": sujeto,
        "session": sesion,
        "algorithm": ALGORITMO,
        "input_path": t1_path,
        "brain_path": salidas["brain"],
        "mask_path": salidas["mask"],
        "status": "procesado",
        "seconds": round(perf_counter() - inicio, 1),
        "brain_volume_ml": round(volumen_ml, 1),
        "mask_fov_pct": round(fraccion_fov * 100, 1),
    }


def extraer_craneo_lote(inventario, output_root, overwrite=False):
    """Ejecuta la extracción cerebral para todos los T1w del inventario."""

    resultados = []

    for numero, fila in enumerate(inventario.itertuples(), start=1):
        print(f"[{numero}/{len(inventario)}] {fila.sujeto}")
        resultado = extraer_craneo_t1(
            t1_path=fila.ruta,
            output_root=output_root,
            overwrite=overwrite,
        )
        resultados.append(resultado)
        print(f"    {resultado['status']} → {resultado['brain_path'].name}")

    return pd.DataFrame(resultados)


def mostrar_entrada_salida(resultados):
    """Crea visores Nilearn navegables para la entrada y la salida BET."""

    if resultados.empty:
        raise ValueError("No hay resultados para visualizar.")

    opciones = []
    for posicion, resultado in enumerate(resultados.itertuples()):
        etiqueta = f"{resultado.subject} | {resultado.session or 'sin sesión'}"
        opciones.append((etiqueta, posicion))

    selector = widgets.Dropdown(
        options=opciones,
        description="Sujeto:",
    )
    panel = widgets.Output()

    def actualizar_visor(change=None):
        resultado = resultados.iloc[selector.value]

        with panel:
            clear_output(wait=True)

            print(f"Entrada: {Path(resultado['input_path']).name}")
            display(
                plotting.view_img(
                    str(resultado["input_path"]),
                    bg_img=False,
                    cmap="gray",
                    symmetric_cmap=False,
                    threshold=None,
                    colorbar=False,
                    black_bg=True,
                    draw_cross=True,
                    width_view=700,
                    title="T1w original",
                )
            )

            print(f"Salida: {Path(resultado['brain_path']).name}")
            display(
                plotting.view_img(
                    str(resultado["brain_path"]),
                    bg_img=False,
                    cmap="gray",
                    symmetric_cmap=False,
                    threshold=None,
                    colorbar=False,
                    black_bg=True,
                    draw_cross=True,
                    width_view=700,
                    title="T1w con extracción cerebral",
                )
            )

            print(f"Máscara: {Path(resultado['mask_path']).name}")
            display(
                plotting.view_img(
                    str(resultado["mask_path"]),
                    bg_img=str(resultado["input_path"]),
                    cmap="autumn",
                    symmetric_cmap=False,
                    threshold=0.5,
                    colorbar=False,
                    black_bg=True,
                    draw_cross=True,
                    opacity=0.35,
                    resampling_interpolation="nearest",
                    width_view=700,
                    title="Máscara cerebral sobre el T1w original",
                )
            )

    selector.observe(actualizar_visor, names="value")
    display(selector, panel)
    actualizar_visor()


# -----------------------------------------------------------------------------
# Segmentación de tejidos
# -----------------------------------------------------------------------------

ALGORITMO_SEGMENTACION = (
    "ANTs Atropos: KMeans[3] → MRF espacial [0.1,1x1x1]"
)

TEJIDOS = {
    1: "CSF",
    2: "GM",
    3: "WM",
}

TEJIDOS_CMAP = ListedColormap(
    ["#0072B2", "#E69F00", "#009E73"],
    name="tejidos",
)

LEYENDA_TEJIDOS = """
<div style="display:flex; gap:18px; flex-wrap:wrap; margin:6px 0 10px 0;">
  <span><span style="display:inline-block;width:12px;height:12px;background:#0072B2;
  margin-right:5px;"></span><b>1 · CSF/LCR</b></span>
  <span><span style="display:inline-block;width:12px;height:12px;background:#E69F00;
  margin-right:5px;"></span><b>2 · Sustancia gris (GM)</b></span>
  <span><span style="display:inline-block;width:12px;height:12px;background:#009E73;
  margin-right:5px;"></span><b>3 · Sustancia blanca (WM)</b></span>
</div>
"""


def _rutas_segmentacion(brain_path):
    """Construye las rutas de la segmentación y sus probabilidades."""

    brain_path = Path(brain_path)
    sufijo = "_desc-bet_T1w.nii.gz"

    if not brain_path.name.endswith(sufijo):
        raise ValueError(
            f"No se reconoce el nombre del T1w con BET: {brain_path.name}"
        )

    prefijo = brain_path.name[: -len(sufijo)]
    carpeta = brain_path.parent

    return {
        "segmentation": carpeta / f"{prefijo}_desc-atropos_dseg.nii.gz",
        "csf": carpeta / f"{prefijo}_label-CSF_probseg.nii.gz",
        "gm": carpeta / f"{prefijo}_label-GM_probseg.nii.gz",
        "wm": carpeta / f"{prefijo}_label-WM_probseg.nii.gz",
    }


def _misma_geometria(imagen_a, imagen_b):
    """Comprueba que dos imágenes compartan matriz y espacio físico."""

    return (
        imagen_a.shape == imagen_b.shape
        and np.allclose(imagen_a.spacing, imagen_b.spacing)
        and np.allclose(imagen_a.origin, imagen_b.origin)
        and np.allclose(imagen_a.direction, imagen_b.direction)
    )


def _metricas_segmentacion(segmentation, mask):
    """Calcula cobertura y volumen de cada tejido."""

    etiquetas = np.rint(segmentation.numpy()).astype(np.uint8)
    mascara = mask.numpy() > 0
    presentes = set(np.unique(etiquetas[mascara]))
    faltantes = set(TEJIDOS) - presentes

    if faltantes:
        nombres = [TEJIDOS[etiqueta] for etiqueta in sorted(faltantes)]
        raise RuntimeError(
            "Atropos no produjo las tres clases esperadas: "
            + ", ".join(nombres)
        )

    cobertura = float(np.count_nonzero(etiquetas[mascara]) / mascara.sum())
    if cobertura < 0.95:
        raise RuntimeError(
            f"La segmentación solo cubre {cobertura * 100:.1f}% "
            "de la máscara cerebral."
        )

    voxel_ml = float(np.prod(segmentation.spacing) / 1000)
    volumenes = {
        nombre: float(np.count_nonzero(etiquetas == etiqueta) * voxel_ml)
        for etiqueta, nombre in TEJIDOS.items()
    }

    return cobertura, volumenes


def segmentar_tejidos_t1(
    brain_path,
    mask_path,
    overwrite=False,
    verbose=False,
):
    """Segmenta un T1w sin cráneo en CSF, GM y WM con Atropos."""

    brain_path = Path(brain_path)
    mask_path = Path(mask_path)
    salidas = _rutas_segmentacion(brain_path)
    productos = tuple(salidas.values())

    sujeto = next(
        (parte for parte in brain_path.parts if parte.startswith("sub-")),
        "desconocido",
    )
    sesion = next(
        (parte for parte in brain_path.parts if parte.startswith("ses-")),
        None,
    )

    if all(ruta.exists() for ruta in productos) and not overwrite:
        segmentation = ants.image_read(str(salidas["segmentation"]))
        mask = ants.image_read(str(mask_path))
        cobertura, volumenes = _metricas_segmentacion(segmentation, mask)

        return {
            "subject": sujeto,
            "session": sesion,
            "algorithm": ALGORITMO_SEGMENTACION,
            "input_path": brain_path,
            "segmentation_path": salidas["segmentation"],
            "csf_path": salidas["csf"],
            "gm_path": salidas["gm"],
            "wm_path": salidas["wm"],
            "status": "reutilizado",
            "seconds": 0.0,
            "coverage_pct": round(cobertura * 100, 1),
            "csf_ml": round(volumenes["CSF"], 1),
            "gm_ml": round(volumenes["GM"], 1),
            "wm_ml": round(volumenes["WM"], 1),
        }

    existentes = [ruta for ruta in productos if ruta.exists()]
    if existentes and not overwrite:
        raise RuntimeError(
            f"Hay productos de segmentación incompletos para {sujeto}. "
            "Ejecuta una vez con overwrite=True."
        )

    inicio = perf_counter()
    brain = ants.image_read(str(brain_path))
    mask = ants.image_read(str(mask_path))

    if not _misma_geometria(brain, mask):
        raise ValueError(
            f"El T1w y la máscara de {sujeto} no comparten geometría."
        )

    opciones_atropos = {"verbose": 1} if verbose else {}

    resultado = ants.atropos(
        a=brain,
        x=mask,
        i="KMeans[3]",
        m="[0.1,1x1x1]",
        c="[5,0]",
        **opciones_atropos,
    )

    segmentation = resultado["segmentation"]
    probabilidades = resultado["probabilityimages"]

    if len(probabilidades) != 3:
        raise RuntimeError(
            f"Atropos devolvió {len(probabilidades)} clases para {sujeto}."
        )

    cobertura, volumenes = _metricas_segmentacion(segmentation, mask)

    ants.image_write(segmentation, str(salidas["segmentation"]))
    ants.image_write(probabilidades[0], str(salidas["csf"]))
    ants.image_write(probabilidades[1], str(salidas["gm"]))
    ants.image_write(probabilidades[2], str(salidas["wm"]))

    return {
        "subject": sujeto,
        "session": sesion,
        "algorithm": ALGORITMO_SEGMENTACION,
        "input_path": brain_path,
        "segmentation_path": salidas["segmentation"],
        "csf_path": salidas["csf"],
        "gm_path": salidas["gm"],
        "wm_path": salidas["wm"],
        "status": "procesado",
        "seconds": round(perf_counter() - inicio, 1),
        "coverage_pct": round(cobertura * 100, 1),
        "csf_ml": round(volumenes["CSF"], 1),
        "gm_ml": round(volumenes["GM"], 1),
        "wm_ml": round(volumenes["WM"], 1),
    }


def segmentar_tejidos_lote(resultados_extraccion, overwrite=False):
    """Segmenta todos los T1w incluidos en los resultados del BET."""

    resultados = []

    for numero, fila in enumerate(
        resultados_extraccion.itertuples(),
        start=1,
    ):
        print(f"[{numero}/{len(resultados_extraccion)}] {fila.subject}")

        resultado = segmentar_tejidos_t1(
            brain_path=fila.brain_path,
            mask_path=fila.mask_path,
            overwrite=overwrite,
        )
        resultados.append(resultado)
        print(
            f"    {resultado['status']} → "
            f"{resultado['segmentation_path'].name}"
        )

    return pd.DataFrame(resultados)


def mostrar_segmentacion(resultados):
    """Abre un visor interactivo de la segmentación sobre el T1w."""

    if resultados.empty:
        raise ValueError("No hay resultados de segmentación para visualizar.")

    opciones = []
    for posicion, resultado in enumerate(resultados.itertuples()):
        etiqueta = f"{resultado.subject} | {resultado.session or 'sin sesión'}"
        opciones.append((etiqueta, posicion))

    selector = widgets.Dropdown(
        options=opciones,
        description="Sujeto:",
    )
    panel = widgets.Output()

    def actualizar_visor(change=None):
        resultado = resultados.iloc[selector.value]

        with panel:
            clear_output(wait=True)
            print(f"Entrada: {Path(resultado['input_path']).name}")
            display(HTML(LEYENDA_TEJIDOS))

            display(
                plotting.view_img(
                    str(resultado["segmentation_path"]),
                    bg_img=str(resultado["input_path"]),
                    cmap=TEJIDOS_CMAP,
                    symmetric_cmap=False,
                    threshold=0.5,
                    vmin=0.5,
                    vmax=3.5,
                    colorbar=False,
                    black_bg=True,
                    draw_cross=True,
                    opacity=0.45,
                    resampling_interpolation="nearest",
                    width_view=700,
                    title="Atropos: CSF / GM / WM",
                )
            )

    selector.observe(actualizar_visor, names="value")
    display(selector, panel)
    actualizar_visor()


# -----------------------------------------------------------------------------
# Normalización anatómica a MNI152
# -----------------------------------------------------------------------------

ALGORITMO_NORMALIZACION = (
    "ANTs antsRegistrationSyNQuick[s]: T1w BET → MNI152 T1 2 mm"
)

BORDES_CMAP = ListedColormap(
    ["#00E5FF", "#00E5FF"],
    name="bordes_normalizados",
)


def preparar_plantilla_mni(template_root):
    """Crea localmente la plantilla MNI152 skull-stripped de 2 mm."""

    template_root = Path(template_root)
    template_root.mkdir(parents=True, exist_ok=True)
    template_path = template_root / "MNI152_T1_2mm.nii.gz"

    if not template_path.exists():
        template = load_mni152_template(resolution=2)
        nib.save(template, template_path)

    return template_path


def _rutas_normalizacion(brain_path):
    """Construye las rutas del T1 normalizado y sus transformaciones."""

    brain_path = Path(brain_path)
    sufijo = "_desc-bet_T1w.nii.gz"

    if not brain_path.name.endswith(sufijo):
        raise ValueError(
            f"No se reconoce el nombre del T1w con BET: {brain_path.name}"
        )

    prefijo = brain_path.name[: -len(sufijo)]
    transform_dir = brain_path.parent.parent / "transforms"
    transform_dir.mkdir(parents=True, exist_ok=True)

    return {
        "normalized": (
            brain_path.parent
            / f"{prefijo}_space-MNI152_desc-preproc_T1w.nii.gz"
        ),
        "manifest": (
            transform_dir
            / f"{prefijo}_from-native_to-MNI152_xfm.json"
        ),
        "transform_prefix": (
            transform_dir
            / f"{prefijo}_from-native_to-MNI152_"
        ),
    }


def _lista_transformaciones(transformaciones):
    """Convierte la salida de ANTs en una lista uniforme de rutas."""

    if isinstance(transformaciones, (str, Path)):
        transformaciones = [transformaciones]
    return [Path(ruta) for ruta in transformaciones]


def _leer_transformaciones(manifest_path, template_path):
    """Recupera y verifica transformaciones guardadas previamente."""

    manifest_path = Path(manifest_path)
    datos = json.loads(manifest_path.read_text(encoding="utf-8"))
    carpeta = manifest_path.parent

    if Path(datos["template"]).resolve() != Path(template_path).resolve():
        raise RuntimeError(
            "La normalización existente utilizó otra plantilla MNI. "
            "Ejecuta una vez con overwrite=True."
        )

    forward = [carpeta / nombre for nombre in datos["forward_transforms"]]
    inverse = [carpeta / nombre for nombre in datos["inverse_transforms"]]
    faltantes = [ruta for ruta in forward + inverse if not ruta.exists()]

    if faltantes:
        raise RuntimeError(
            "Faltan archivos de transformación: "
            + ", ".join(ruta.name for ruta in faltantes)
        )

    return forward, inverse


def normalizar_t1_mni(
    brain_path,
    mni_template_path,
    overwrite=False,
    verbose=False,
):
    """Normaliza un T1w sin cráneo a MNI152 con SyNQuick."""

    brain_path = Path(brain_path)
    mni_template_path = Path(mni_template_path)
    salidas = _rutas_normalizacion(brain_path)

    sujeto = next(
        (parte for parte in brain_path.parts if parte.startswith("sub-")),
        "desconocido",
    )
    sesion = next(
        (parte for parte in brain_path.parts if parte.startswith("ses-")),
        None,
    )

    if not mni_template_path.exists():
        raise FileNotFoundError(
            f"No se encontró la plantilla MNI: {mni_template_path}"
        )

    completos = (
        salidas["normalized"].exists()
        and salidas["manifest"].exists()
    )

    if completos and not overwrite:
        forward, inverse = _leer_transformaciones(
            salidas["manifest"],
            mni_template_path,
        )
        return {
            "subject": sujeto,
            "session": sesion,
            "algorithm": ALGORITMO_NORMALIZACION,
            "input_path": brain_path,
            "template_path": mni_template_path,
            "normalized_path": salidas["normalized"],
            "forward_transforms": forward,
            "inverse_transforms": inverse,
            "status": "reutilizado",
            "seconds": 0.0,
        }

    parciales = (
        salidas["normalized"].exists()
        or salidas["manifest"].exists()
    )
    if parciales and not overwrite:
        raise RuntimeError(
            f"Hay productos de normalización incompletos para {sujeto}. "
            "Ejecuta una vez con overwrite=True."
        )

    inicio = perf_counter()
    template = ants.image_read(str(mni_template_path))
    brain = ants.image_read(str(brain_path))
    opciones_registro = {"verbose": True} if verbose else {}

    registro = ants.registration(
        fixed=template,
        moving=brain,
        type_of_transform="antsRegistrationSyNQuick[s]",
        outprefix=str(salidas["transform_prefix"]),
        **opciones_registro,
    )

    normalized = registro["warpedmovout"]
    forward = _lista_transformaciones(registro["fwdtransforms"])
    inverse = _lista_transformaciones(registro["invtransforms"])

    if not _misma_geometria(normalized, template):
        raise RuntimeError(
            f"La salida normalizada de {sujeto} no coincide con la geometría MNI."
        )

    if np.count_nonzero(normalized.numpy()) == 0:
        raise RuntimeError(f"La normalización de {sujeto} produjo una imagen vacía.")

    faltantes = [ruta for ruta in forward + inverse if not ruta.exists()]
    if faltantes:
        raise RuntimeError(
            "ANTs no generó todas las transformaciones: "
            + ", ".join(ruta.name for ruta in faltantes)
        )

    ants.image_write(normalized, str(salidas["normalized"]))

    manifest = {
        "algorithm": ALGORITMO_NORMALIZACION,
        "template": str(mni_template_path.resolve()),
        "forward_transforms": [ruta.name for ruta in forward],
        "inverse_transforms": [ruta.name for ruta in inverse],
    }
    salidas["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "subject": sujeto,
        "session": sesion,
        "algorithm": ALGORITMO_NORMALIZACION,
        "input_path": brain_path,
        "template_path": mni_template_path,
        "normalized_path": salidas["normalized"],
        "forward_transforms": forward,
        "inverse_transforms": inverse,
        "status": "procesado",
        "seconds": round(perf_counter() - inicio, 1),
    }


def normalizar_t1_lote(
    resultados_segmentacion,
    mni_template_path,
    overwrite=False,
):
    """Normaliza a MNI todos los T1w segmentados."""

    resultados = []

    for numero, fila in enumerate(
        resultados_segmentacion.itertuples(),
        start=1,
    ):
        print(f"[{numero}/{len(resultados_segmentacion)}] {fila.subject}")

        resultado = normalizar_t1_mni(
            brain_path=fila.input_path,
            mni_template_path=mni_template_path,
            overwrite=overwrite,
        )
        resultados.append(resultado)
        print(
            f"    {resultado['status']} → "
            f"{resultado['normalized_path'].name}"
        )

    return pd.DataFrame(resultados)


def _mapa_bordes(image_path):
    """Genera bordes anatómicos finos para el control visual del registro."""

    image = nib.load(image_path)
    data = np.asarray(image.dataobj, dtype=np.float32)
    positivos = data[np.isfinite(data) & (data > 0)]

    if positivos.size == 0:
        raise RuntimeError(f"La imagen está vacía: {image_path}")

    p01, p99 = np.percentile(positivos, (1, 99))
    escala = max(float(p99 - p01), np.finfo(np.float32).eps)
    normalizada = np.clip((data - p01) / escala, 0, 1)
    gradiente = gaussian_gradient_magnitude(normalizada, sigma=1.0)
    gradientes_validos = gradiente[data > 0]
    umbral = float(np.percentile(gradientes_validos, 80))

    bordes = np.where(
        (data > 0) & (gradiente >= umbral),
        gradiente,
        0,
    ).astype(np.float32)

    header = image.header.copy()
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(bordes, image.affine, header)


def _anatomia_sin_fondo(image_path):
    """Conserva toda la anatomía cerebral y elimina el fondo del T1w."""

    image = nib.load(image_path)
    data = np.asarray(image.dataobj, dtype=np.float32)
    positivos = data[np.isfinite(data) & (data > 0)]

    if positivos.size == 0:
        raise RuntimeError(f"La imagen está vacía: {image_path}")

    # La transformación puede dejar intensidades diminutas fuera del cerebro.
    # Se conserva el componente anatómico principal y se rellenan sus huecos
    # para no perder ventrículos, corteza ni núcleos profundos.
    umbral_fondo = float(np.percentile(positivos, 1))
    componentes, cantidad = label(data > umbral_fondo)

    if cantidad == 0:
        raise RuntimeError(f"No se detectó anatomía en: {image_path}")

    tamanos = np.bincount(componentes.ravel())
    tamanos[0] = 0
    componente_principal = int(np.argmax(tamanos))
    mascara = binary_fill_holes(componentes == componente_principal)
    anatomia = np.where(mascara, data, 0).astype(np.float32)

    header = image.header.copy()
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(anatomia, image.affine, header)


def mostrar_qc_normalizacion(resultados):
    """Muestra el T1 normalizado y sus bordes sobre la plantilla MNI."""

    if resultados.empty:
        raise ValueError("No hay normalizaciones para visualizar.")

    opciones = []
    for posicion, resultado in enumerate(resultados.itertuples()):
        etiqueta = f"{resultado.subject} | {resultado.session or 'sin sesión'}"
        opciones.append((etiqueta, posicion))

    selector = widgets.Dropdown(
        options=opciones,
        description="Sujeto:",
    )
    panel = widgets.Output()

    def actualizar_visor(change=None):
        resultado = resultados.iloc[selector.value]

        with panel:
            clear_output(wait=True)

            print(f"Referencia: {Path(resultado['template_path']).name}")
            display(
                plotting.view_img(
                    str(resultado["template_path"]),
                    bg_img=False,
                    cmap="gray",
                    symmetric_cmap=False,
                    threshold=None,
                    colorbar=False,
                    black_bg=True,
                    draw_cross=True,
                    width_view=700,
                    title="Plantilla MNI152 T1 2 mm",
                )
            )

            print(
                "Gris: plantilla MNI152 | Color: anatomía completa del "
                "T1w normalizado"
            )
            anatomia_normalizada = _anatomia_sin_fondo(
                resultado["normalized_path"]
            )
            display(
                plotting.view_img(
                    anatomia_normalizada,
                    bg_img=str(resultado["template_path"]),
                    cmap="autumn",
                    symmetric_cmap=False,
                    threshold=1e-6,
                    colorbar=False,
                    black_bg=True,
                    draw_cross=True,
                    dim=0,
                    opacity=0.55,
                    resampling_interpolation="nearest",
                    width_view=700,
                    title="Anatomía del T1w sobre la plantilla MNI152",
                )
            )

            print("Gris: plantilla MNI152 | Cian: bordes del T1w normalizado")
            bordes_normalizados = _mapa_bordes(
                resultado["normalized_path"]
            )
            display(
                plotting.view_img(
                    bordes_normalizados,
                    bg_img=str(resultado["template_path"]),
                    cmap=BORDES_CMAP,
                    symmetric_cmap=False,
                    threshold=1e-6,
                    colorbar=False,
                    black_bg=True,
                    draw_cross=True,
                    dim=0,
                    opacity=0.85,
                    resampling_interpolation="nearest",
                    width_view=700,
                    title="Bordes del T1w sobre la plantilla MNI152",
                )
            )

    selector.observe(actualizar_visor, names="value")
    display(selector, panel)
    actualizar_visor()


def mostrar_normalizacion(resultados):
    """Alias compatible para el control visual de la normalización."""

    return mostrar_qc_normalizacion(resultados)
