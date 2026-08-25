"""Preprocesamiento funcional sencillo para el taller de rs-fMRI."""

import json
import re
import tempfile
from pathlib import Path
from time import perf_counter

import ants
import ipywidgets as widgets
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from IPython.display import HTML, clear_output, display
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from nilearn import datasets, image, masking, plotting
from scipy.ndimage import binary_erosion, gaussian_gradient_magnitude
from scipy.ndimage import shift as ndimage_shift
from scipy.signal import butter, detrend, sosfiltfilt, welch
from scipy.spatial.transform import Rotation


ALGORITMO_INVENTARIO = (
    "Búsqueda recursiva de BOLD 4D → lectura del JSON → "
    "verificación de TR, volúmenes y SliceTiming"
)

ALGORITMO_DESCARTE = (
    "Eliminación de volúmenes iniciales no estacionarios del BOLD 4D"
)

ALGORITMO_MOVIMIENTO = (
    "ANTs BOLDRigid: registro rígido de cada volumen a la imagen BOLD media"
)

ALGORITMO_QC_MOVIMIENTO = (
    "Reestimación BOLDRigid sobre el BOLD corregido → movimiento residual"
)

ALGORITMO_SLICE_TIMING = (
    "Interpolación spline cúbica por corte → referencia temporal TR/2"
)

ALGORITMO_CORREGISTRO = (
    "ANTs Rigid con información mutua: BOLD medio → T1w BET"
)

ALGORITMO_NORMALIZACION_BOLD = (
    "ANTs: composición BOLD→T1w + T1w→MNI152 → una interpolación lineal"
)

ALGORITMO_ACOMPCOR = (
    "aCompCor: WM/CSF anatómicos en MNI152 → umbral + erosión → PCA temporal"
)

ALGORITMO_DENOISING = (
    "24HMP + 5 WM aCompCor + 5 CSF aCompCor + censurado "
    "→ regresión y filtro 0.008–0.09 Hz simultáneos"
)

ALGORITMO_PARCELACION = (
    "Schaefer 2018 en MNI152 2 mm → promedio de los vóxeles "
    "cubiertos por cada región"
)

ALGORITMO_CONECTIVIDAD = (
    "Correlación de Pearson entre las series Schaefer de las ROIs "
    "válidas en todos los sujetos"
)

BORDES_FUNC_CMAP = ListedColormap(
    ["#00E5FF", "#00E5FF"],
    name="bordes_bold",
)

TEJIDOS_FUNC_CMAP = ListedColormap(
    ["#0072B2", "#E69F00", "#009E73"],
    name="tejidos_funcionales",
)

COMPCOR_CMAP = ListedColormap(
    ["#00B8D9", "#F4A261"],
    name="tejidos_acompcor",
)

CAJA_MM = np.array([140.0, 180.0, 115.0])
PUNTOS_CONTROL = np.vstack(
    [np.diag(CAJA_MM / 2.0), -np.diag(CAJA_MM / 2.0)]
)


def _entidad(ruta, nombre):
    """Obtiene una entidad BIDS desde el nombre o la ruta del archivo."""

    ruta = Path(ruta)
    patron = re.compile(rf"(?:^|_){re.escape(nombre)}-([^_]+)")
    coincidencia = patron.search(ruta.name)

    if coincidencia:
        return coincidencia.group(1)

    prefijo = f"{nombre}-"
    return next(
        (
            parte.removeprefix(prefijo)
            for parte in ruta.parts
            if parte.startswith(prefijo)
        ),
        None,
    )


def _json_asociado(nifti_path):
    """Construye la ruta del JSON asociado a un archivo NIfTI."""

    nifti_path = Path(nifti_path)
    nombre = nifti_path.name

    if nombre.endswith(".nii.gz"):
        return nifti_path.with_name(nombre[:-7] + ".json")
    if nombre.endswith(".nii"):
        return nifti_path.with_name(nombre[:-4] + ".json")

    raise ValueError(f"No se reconoce como NIfTI: {nifti_path}")


def inventariar_bold(bids_root, minimo_volumenes=20):
    """Localiza y verifica todos los BOLD 4D y sus metadatos."""

    bids_root = Path(bids_root)

    if not bids_root.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de datos: {bids_root}")

    bold_paths = sorted(
        list(bids_root.rglob("*_bold.nii.gz"))
        + list(bids_root.rglob("*_bold.nii"))
    )

    if not bold_paths:
        raise FileNotFoundError("No se encontró ningún archivo BOLD.")

    filas = []
    errores = []

    for bold_path in bold_paths:
        json_path = _json_asociado(bold_path)

        if not json_path.exists():
            errores.append(f"Falta el JSON de {bold_path.name}")
            continue

        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errores.append(f"JSON inválido {json_path.name}: {error}")
            continue

        bold = nib.load(bold_path)

        if len(bold.shape) != 4:
            errores.append(f"{bold_path.name} no es un BOLD 4D: {bold.shape}")
            continue

        volumenes = int(bold.shape[3])
        tr = float(metadata.get("RepetitionTime", 0) or 0)
        slice_timing = metadata.get("SliceTiming")
        numero_cortes = int(bold.shape[2])

        if volumenes < minimo_volumenes:
            errores.append(
                f"{bold_path.name} tiene solo {volumenes} volúmenes "
                f"(mínimo: {minimo_volumenes})"
            )

        if tr <= 0:
            errores.append(f"Falta RepetitionTime válido en {json_path.name}")

        if slice_timing is None:
            estado_slice_timing = "no informado"
        elif len(slice_timing) != numero_cortes:
            estado_slice_timing = (
                f"revisar: {len(slice_timing)} tiempos / {numero_cortes} cortes"
            )
        else:
            estado_slice_timing = "correcto"

        filas.append(
            {
                "sujeto": _entidad(bold_path, "sub"),
                "sesion": _entidad(bold_path, "ses"),
                "tarea": _entidad(bold_path, "task"),
                "corrida": _entidad(bold_path, "run"),
                "dimensiones": tuple(int(x) for x in bold.shape),
                "volumenes": volumenes,
                "voxel_mm": tuple(
                    round(float(x), 3) for x in bold.header.get_zooms()[:3]
                ),
                "orientacion": "".join(nib.aff2axcodes(bold.affine)),
                "TR_s": tr,
                "duracion_min": round(volumenes * tr / 60, 2),
                "SliceTiming": estado_slice_timing,
                "descartados_scanner": int(
                    metadata.get("NumberOfVolumesDiscardedByScanner", 0) or 0
                ),
                "descartados_usuario": int(
                    metadata.get("NumberOfVolumesDiscardedByUser", 0) or 0
                ),
                "ruta": bold_path,
                "json_path": json_path,
            }
        )

    if errores:
        detalle = "\n- ".join(errores)
        raise ValueError(f"El inventario funcional encontró errores:\n- {detalle}")

    inventario = pd.DataFrame(filas)

    if inventario.empty:
        raise ValueError("No quedó ningún BOLD válido después de verificarlo.")

    return inventario


def mostrar_qc_bold_crudo(inventario):
    """Muestra el BOLD medio crudo y sus datos básicos de adquisición."""

    if inventario.empty:
        raise ValueError("No hay BOLD para visualizar.")

    opciones = []
    for posicion, fila in enumerate(inventario.itertuples()):
        etiqueta = f"sub-{fila.sujeto}"
        if fila.sesion:
            etiqueta += f" | ses-{fila.sesion}"
        if fila.tarea:
            etiqueta += f" | task-{fila.tarea}"
        if fila.corrida:
            etiqueta += f" | run-{fila.corrida}"
        opciones.append((etiqueta, posicion))

    selector = widgets.Dropdown(options=opciones, description="BOLD:")
    panel = widgets.Output()

    def actualizar(change=None):
        fila = inventario.iloc[selector.value]
        bold = nib.load(fila["ruta"])
        bold_medio = image.mean_img(bold)

        with panel:
            clear_output(wait=True)
            print(f"Entrada: {Path(fila['ruta']).name}")
            print(
                f"Dimensiones: {fila['dimensiones']} | "
                f"Volúmenes: {fila['volumenes']} | "
                f"TR: {fila['TR_s']} s | "
                f"Duración: {fila['duracion_min']} min"
            )
            print(f"SliceTiming: {fila['SliceTiming']}")

            display(
                plotting.view_img(
                    bold_medio,
                    bg_img=False,
                    cmap="gray",
                    symmetric_cmap=False,
                    threshold=None,
                    colorbar=False,
                    black_bg=True,
                    draw_cross=True,
                    width_view=700,
                    title="QC inicial: BOLD medio crudo",
                )
            )

    selector.observe(actualizar, names="value")
    display(selector, panel)
    actualizar()


# -----------------------------------------------------------------------------
# Corrección temporal por cortes
# -----------------------------------------------------------------------------


def _entidad_limpia(valor, prefijo):
    """Normaliza entidades que pueden venir como ``s003`` o ``sub-s003``."""

    if valor is None or bool(pd.isna(valor)):
        return None
    return str(valor).removeprefix(prefijo)


def _fila_inventario_correspondiente(fila_funcional, inventario):
    """Encuentra los metadatos BIDS de una salida funcional."""

    sujeto = _entidad_limpia(fila_funcional.subject, "sub-")
    sesion = _entidad_limpia(fila_funcional.session, "ses-")
    tarea = _entidad_limpia(fila_funcional.task, "task-")
    corrida = _entidad_limpia(fila_funcional.run, "run-")

    candidatos = inventario.copy()
    candidatos = candidatos[
        candidatos["sujeto"].map(
            lambda valor: _entidad_limpia(valor, "sub-") == sujeto
        )
    ]

    for columna, valor, prefijo in (
        ("sesion", sesion, "ses-"),
        ("tarea", tarea, "task-"),
        ("corrida", corrida, "run-"),
    ):
        if valor is not None:
            candidatos = candidatos[
                candidatos[columna].map(
                    lambda actual: _entidad_limpia(actual, prefijo) == valor
                )
            ]

    if len(candidatos) != 1:
        raise ValueError(
            f"Se esperaban metadatos únicos para sub-{sujeto}; "
            f"se encontraron {len(candidatos)} filas."
        )

    return candidatos.iloc[0]


def _informacion_slice_timing(json_path, shape, tr):
    """Lee y valida SliceTiming y su eje de adquisición BIDS."""

    metadata = json.loads(Path(json_path).read_text(encoding="utf-8"))
    slice_timing = metadata.get("SliceTiming")

    if slice_timing is None:
        raise ValueError(
            f"El JSON no contiene SliceTiming: {Path(json_path).name}"
        )

    tiempos = np.asarray(slice_timing, dtype=float)
    if tiempos.ndim != 1 or tiempos.size < 2:
        raise ValueError("SliceTiming debe ser una lista numérica por corte.")
    if not np.isfinite(tiempos).all():
        raise ValueError("SliceTiming contiene valores no finitos.")
    if tiempos.min() < 0 or tiempos.max() >= float(tr):
        raise ValueError(
            f"SliceTiming debe estar dentro del TR ({tr} s)."
        )

    direccion = metadata.get("SliceEncodingDirection")
    if direccion:
        eje = {"i": 0, "j": 1, "k": 2}.get(direccion[0])
        if eje is None:
            raise ValueError(
                f"SliceEncodingDirection no reconocido: {direccion}"
            )
        if direccion.endswith("-"):
            tiempos = tiempos[::-1]
    else:
        candidatos = [
            eje for eje, dimension in enumerate(shape[:3])
            if int(dimension) == int(tiempos.size)
        ]
        if len(candidatos) == 1:
            eje = candidatos[0]
        elif shape[2] == tiempos.size:
            eje = 2
        else:
            raise ValueError(
                "No se pudo inferir el eje de SliceTiming; añade "
                "SliceEncodingDirection al JSON."
            )

    if shape[eje] != tiempos.size:
        raise ValueError(
            f"SliceTiming tiene {tiempos.size} valores, pero el eje "
            f"{eje} contiene {shape[eje]} cortes."
        )

    return tiempos, eje, direccion


def corregir_slice_timing_lote(
    resultados_movimiento,
    inventario_bold,
    referencia="mitad_TR",
    orden_spline=3,
    overwrite=False,
):
    """Alinea temporalmente los cortes de cada BOLD corregido por movimiento."""

    if resultados_movimiento.empty:
        raise ValueError("No hay BOLD corregidos por movimiento.")
    if inventario_bold.empty:
        raise ValueError("El inventario BOLD está vacío.")

    orden_spline = int(orden_spline)
    if orden_spline < 0 or orden_spline > 5:
        raise ValueError("orden_spline debe estar entre 0 y 5.")

    resultados = []

    for numero, fila in enumerate(
        resultados_movimiento.itertuples(index=False),
        start=1,
    ):
        entrada = Path(fila.corrected_path)
        sufijo = "_desc-moco_bold.nii.gz"
        if not entrada.name.endswith(sufijo):
            raise ValueError(f"No se reconoce la entrada: {entrada.name}")

        metadata_fila = _fila_inventario_correspondiente(
            fila,
            inventario_bold,
        )
        bold = nib.load(entrada)
        tr = float(fila.TR_s)
        tiempos, eje, direccion = _informacion_slice_timing(
            metadata_fila.json_path,
            bold.shape,
            tr,
        )

        if referencia == "mitad_TR":
            referencia_s = tr / 2.0
        else:
            referencia_s = float(referencia)
            if referencia_s < 0 or referencia_s >= tr:
                raise ValueError("La referencia debe quedar dentro del TR.")

        desplazamientos = (tiempos - referencia_s) / tr
        prefijo = entrada.name[: -len(sufijo)]
        salida = entrada.parent / f"{prefijo}_desc-mocoStc_bold.nii.gz"

        print(
            f"[{numero}/{len(resultados_movimiento)}] "
            f"sub-{fila.subject}: slice timing"
        )

        if salida.exists() and not overwrite:
            salida_img = nib.load(salida)
            if salida_img.shape != bold.shape:
                raise RuntimeError(
                    f"La salida existente no coincide con la entrada: {salida}"
                )
            estado = "reutilizado"
            segundos = 0.0
        else:
            inicio = perf_counter()
            datos = np.asarray(bold.dataobj, dtype=np.float32)
            corregidos = np.empty_like(datos, dtype=np.float32)

            for indice, desplazamiento in enumerate(desplazamientos):
                selector = [slice(None)] * 4
                selector[eje] = indice
                selector = tuple(selector)
                corte = datos[selector]
                vector = [0.0] * corte.ndim
                vector[-1] = float(desplazamiento)
                corregidos[selector] = ndimage_shift(
                    corte,
                    shift=vector,
                    order=orden_spline,
                    mode="nearest",
                    prefilter=orden_spline > 1,
                    output=np.float32,
                )

            header = bold.header.copy()
            header.set_data_dtype(np.float32)
            salida_img = nib.Nifti1Image(
                corregidos,
                bold.affine,
                header,
            )
            nib.save(salida_img, salida)
            estado = "procesado"
            segundos = round(perf_counter() - inicio, 1)

        resultado = fila._asdict()
        resultado.update(
            {
                "stc_input_path": entrada,
                "stc_path": salida,
                "slice_timing_json": Path(metadata_fila.json_path),
                "slice_axis": int(eje),
                "slice_encoding_direction": direccion,
                "reference_time_s": round(float(referencia_s), 6),
                "max_slice_shift_s": round(
                    float(np.max(np.abs(tiempos - referencia_s))),
                    6,
                ),
                "max_slice_shift_volumes": round(
                    float(np.max(np.abs(desplazamientos))),
                    4,
                ),
                "stc_status": estado,
                "stc_seconds": segundos,
            }
        )
        resultados.append(resultado)
        print(
            f"    {estado} → {salida.name} | "
            f"desplazamiento máximo: "
            f"{resultado['max_slice_shift_volumes']} vol"
        )

    return pd.DataFrame(resultados)


def mostrar_qc_slice_timing(resultados_stc):
    """Compara temporalmente un corte representativo antes y después de STC."""

    if resultados_stc.empty:
        raise ValueError("No hay resultados de slice timing para revisar.")

    opciones = []
    for posicion, fila in enumerate(resultados_stc.itertuples()):
        etiqueta = f"sub-{fila.subject}"
        if fila.session:
            etiqueta += f" | ses-{fila.session}"
        opciones.append((etiqueta, posicion))

    selector_sujeto = widgets.Dropdown(
        options=opciones,
        description="BOLD:",
    )
    panel = widgets.Output()

    def actualizar(change=None):
        fila = resultados_stc.iloc[selector_sujeto.value]
        antes_img = nib.load(fila["stc_input_path"])
        despues_img = nib.load(fila["stc_path"])
        eje = int(fila["slice_axis"])
        indice = int(antes_img.shape[eje] // 2)

        selector_corte = [slice(None)] * 4
        selector_corte[eje] = indice
        selector_corte = tuple(selector_corte)
        antes = np.asarray(
            antes_img.dataobj[selector_corte],
            dtype=np.float32,
        )
        despues = np.asarray(
            despues_img.dataobj[selector_corte],
            dtype=np.float32,
        )

        media_espacial = np.mean(antes, axis=-1)
        positivos = media_espacial[
            np.isfinite(media_espacial) & (media_espacial > 0)
        ]
        if positivos.size == 0:
            raise RuntimeError("El corte representativo está vacío.")

        mascara = media_espacial > np.percentile(positivos, 25)
        serie_antes = antes[mascara].mean(axis=0)
        serie_despues = despues[mascara].mean(axis=0)
        diferencias = (despues - antes)[mascara].ravel()
        if diferencias.size > 200000:
            paso = int(np.ceil(diferencias.size / 200000))
            diferencias = diferencias[::paso]

        correlacion = float(
            np.corrcoef(serie_antes, serie_despues)[0, 1]
        )
        cambio_relativo = 100 * float(
            np.mean(np.abs(serie_despues - serie_antes))
            / max(np.mean(np.abs(serie_antes)), np.finfo(float).eps)
        )
        volumenes = np.arange(len(serie_antes))

        with panel:
            clear_output(wait=True)
            print(
                f"Eje: {eje} | corte QC: {indice} | "
                f"referencia: {fila['reference_time_s']:.3f} s"
            )
            print(
                f"Correlación antes/después: {correlacion:.4f} | "
                f"cambio temporal medio: {cambio_relativo:.3f}%"
            )

            fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
            axes[0].plot(
                volumenes,
                serie_antes,
                linewidth=0.9,
                label="Antes",
            )
            axes[0].plot(
                volumenes,
                serie_despues,
                linewidth=0.9,
                alpha=0.8,
                label="Después",
            )
            axes[0].set_title("Serie temporal del corte central")
            axes[0].set_xlabel("Volumen")
            axes[0].set_ylabel("Señal media")
            axes[0].legend(frameon=False)

            axes[1].hist(diferencias, bins=50, alpha=0.8)
            axes[1].axvline(0, color="black", linestyle=":")
            axes[1].set_title("Cambios producidos por STC")
            axes[1].set_xlabel("Después − antes")
            axes[1].set_ylabel("Vóxeles × volúmenes")
            plt.tight_layout()
            plt.show()

    selector_sujeto.observe(actualizar, names="value")
    display(selector_sujeto, panel)
    actualizar()


# -----------------------------------------------------------------------------
# Corregistro funcional a T1w
# -----------------------------------------------------------------------------


def _fila_anatomica_correspondiente(fila_funcional, resultados_segmentacion):
    """Selecciona la segmentación anatómica del mismo sujeto y sesión."""

    sujeto = _entidad_limpia(fila_funcional.subject, "sub-")
    sesion = _entidad_limpia(fila_funcional.session, "ses-")
    candidatos = resultados_segmentacion.copy()
    candidatos = candidatos[
        candidatos["subject"].map(
            lambda valor: _entidad_limpia(valor, "sub-") == sujeto
        )
    ]

    if sesion is not None:
        candidatos = candidatos[
            candidatos["session"].map(
                lambda valor: _entidad_limpia(valor, "ses-") == sesion
            )
        ]

    if len(candidatos) != 1:
        raise ValueError(
            f"Se esperaba una anatomía para sub-{sujeto}; "
            f"se encontraron {len(candidatos)}."
        )
    return candidatos.iloc[0]


def _metricas_solapamiento_coreg(bold_mask, t1):
    """Calcula Dice y cobertura de la máscara EPI dentro del cerebro T1w."""

    epi = bold_mask.numpy() > 0
    cerebro = t1.numpy() > 0

    if epi.shape != cerebro.shape:
        raise RuntimeError("Las máscaras EPI y T1w no comparten dimensiones.")
    if not epi.any() or not cerebro.any():
        raise RuntimeError("Alguna de las máscaras de corregistro está vacía.")

    interseccion = np.count_nonzero(epi & cerebro)
    dice = 2 * interseccion / (epi.sum() + cerebro.sum())
    epi_dentro = interseccion / epi.sum()
    cerebro_cubierto = interseccion / cerebro.sum()
    return dice, epi_dentro, cerebro_cubierto


def _bordes_funcionales(image_path):
    """Genera bordes del BOLD medio sin colorear el fondo."""

    imagen = nib.load(image_path)
    datos = np.asarray(imagen.dataobj, dtype=np.float32)
    positivos = datos[np.isfinite(datos) & (datos > 0)]
    if positivos.size == 0:
        raise RuntimeError(f"La imagen está vacía: {image_path}")

    p01, p99 = np.percentile(positivos, (1, 99))
    escala = max(float(p99 - p01), np.finfo(np.float32).eps)
    normalizada = np.clip((datos - p01) / escala, 0, 1)
    gradiente = gaussian_gradient_magnitude(normalizada, sigma=1.0)
    validos = gradiente[datos > 0]
    umbral = float(np.percentile(validos, 82))
    bordes = np.where(
        (datos > 0) & (gradiente >= umbral),
        gradiente,
        0,
    ).astype(np.float32)

    header = imagen.header.copy()
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(bordes, imagen.affine, header)


def coregistrar_bold_t1_lote(
    resultados_stc,
    resultados_segmentacion,
    overwrite=False,
    verbose=False,
):
    """Estima BOLD→T1w con el BOLD medio y conserva el 4D en espacio nativo."""

    if resultados_stc.empty:
        raise ValueError("No hay BOLD con slice timing para corregistrar.")
    if resultados_segmentacion.empty:
        raise ValueError("No hay segmentaciones anatómicas disponibles.")

    resultados = []

    for numero, fila in enumerate(
        resultados_stc.itertuples(index=False),
        start=1,
    ):
        anatomia = _fila_anatomica_correspondiente(
            fila,
            resultados_segmentacion,
        )
        bold_path = Path(fila.stc_path)
        t1_path = Path(anatomia.input_path)
        sufijo = "_desc-mocoStc_bold.nii.gz"
        if not bold_path.name.endswith(sufijo):
            raise ValueError(f"No se reconoce el BOLD STC: {bold_path.name}")

        prefijo = bold_path.name[: -len(sufijo)]
        output_dir = bold_path.parent
        boldref_path = output_dir / f"{prefijo}_space-T1w_boldref.nii.gz"
        epi_mask_path = (
            output_dir / f"{prefijo}_space-T1w_desc-brain_mask.nii.gz"
        )
        transform_dir = output_dir.parent / "transforms"
        transform_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = (
            transform_dir / f"{prefijo}_from-bold_to-T1w_xfm.json"
        )
        transform_prefix = transform_dir / f"{prefijo}_from-bold_to-T1w_"
        productos = (
            boldref_path,
            epi_mask_path,
            manifest_path,
        )

        print(
            f"[{numero}/{len(resultados_stc)}] "
            f"sub-{fila.subject}: BOLD → T1w"
        )

        if all(ruta.exists() for ruta in productos) and not overwrite:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            forward = [
                manifest_path.parent / nombre
                for nombre in manifest["forward_transforms"]
            ]
            faltantes = [ruta for ruta in forward if not ruta.exists()]
            if faltantes:
                raise RuntimeError(
                    "Faltan transformaciones de corregistro: "
                    + ", ".join(ruta.name for ruta in faltantes)
                )
            estado = "reutilizado"
            segundos = 0.0
        else:
            existentes = [ruta for ruta in productos if ruta.exists()]
            if existentes and not overwrite:
                raise RuntimeError(
                    f"Hay productos de corregistro incompletos para "
                    f"sub-{fila.subject}. Usa overwrite=True."
                )

            inicio = perf_counter()
            bold = ants.image_read(str(bold_path))
            t1 = ants.image_read(str(t1_path))
            if bold.dimension != 4 or t1.dimension != 3:
                raise ValueError("El corregistro requiere BOLD 4D y T1w 3D.")

            boldref = ants.get_average_of_timeseries(bold)
            opciones_registro = {"verbose": True} if verbose else {}
            registro = ants.registration(
                fixed=t1,
                moving=boldref,
                type_of_transform="Rigid",
                outprefix=str(transform_prefix),
                **opciones_registro,
            )
            forward = [Path(ruta) for ruta in registro["fwdtransforms"]]
            boldref_t1 = registro["warpedmovout"]
            epi_mask = ants.get_mask(boldref_t1, cleanup=2)

            if np.count_nonzero(epi_mask.numpy()) == 0:
                raise RuntimeError(
                    f"No se obtuvo una máscara EPI para sub-{fila.subject}."
                )

            ants.image_write(boldref_t1, str(boldref_path))
            ants.image_write(epi_mask, str(epi_mask_path))
            manifest = {
                "algorithm": ALGORITMO_CORREGISTRO,
                "fixed_t1w": str(t1_path.resolve()),
                "moving_bold": str(bold_path.resolve()),
                "forward_transforms": [ruta.name for ruta in forward],
            }
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            estado = "procesado"
            segundos = round(perf_counter() - inicio, 1)

        t1 = ants.image_read(str(t1_path))
        epi_mask = ants.image_read(str(epi_mask_path))
        dice, epi_dentro, cerebro_cubierto = _metricas_solapamiento_coreg(
            epi_mask,
            t1,
        )

        resultado = fila._asdict()
        resultado.update(
            {
                "t1_path": t1_path,
                "segmentation_path": Path(anatomia.segmentation_path),
                "gm_path": Path(anatomia.gm_path),
                "wm_path": Path(anatomia.wm_path),
                "bold_native_path": bold_path,
                "boldref_t1_path": boldref_path,
                "epi_mask_t1_path": epi_mask_path,
                "bold_to_t1_transforms": forward,
                "coreg_status": estado,
                "coreg_seconds": segundos,
                "dice_epi_t1": round(float(dice), 3),
                "epi_inside_t1_pct": round(float(epi_dentro * 100), 1),
                "t1_covered_by_epi_pct": round(
                    float(cerebro_cubierto * 100),
                    1,
                ),
            }
        )
        resultados.append(resultado)
        print(
            f"    {estado} | Dice: {resultado['dice_epi_t1']:.3f} | "
            f"EPI dentro del cerebro: "
            f"{resultado['epi_inside_t1_pct']:.1f}%"
        )

    return pd.DataFrame(resultados)


def mostrar_qc_coregistro(resultados_coregistro):
    """Muestra bordes funcionales y tejidos anatómicos en espacio T1w."""

    if resultados_coregistro.empty:
        raise ValueError("No hay corregistraciones para visualizar.")

    opciones = []
    for posicion, fila in enumerate(resultados_coregistro.itertuples()):
        etiqueta = f"sub-{fila.subject}"
        if fila.session:
            etiqueta += f" | ses-{fila.session}"
        opciones.append((etiqueta, posicion))

    selector_sujeto = widgets.Dropdown(
        options=opciones,
        description="Sujeto:",
    )
    panel = widgets.Output()

    def actualizar(change=None):
        fila = resultados_coregistro.iloc[selector_sujeto.value]

        with panel:
            clear_output(wait=True)
            print(
                f"Dice EPI–T1w: {fila['dice_epi_t1']:.3f} | "
                f"EPI dentro del cerebro: "
                f"{fila['epi_inside_t1_pct']:.1f}% | "
                f"cerebro cubierto: {fila['t1_covered_by_epi_pct']:.1f}%"
            )
            print("Gris: T1w BET | Cian: bordes del BOLD medio")
            bordes = _bordes_funcionales(fila["boldref_t1_path"])
            display(
                plotting.view_img(
                    bordes,
                    bg_img=str(fila["t1_path"]),
                    cmap=BORDES_FUNC_CMAP,
                    symmetric_cmap=False,
                    threshold=1e-6,
                    colorbar=False,
                    black_bg=True,
                    draw_cross=True,
                    dim=0,
                    opacity=0.9,
                    resampling_interpolation="nearest",
                    width_view=700,
                    title="Bordes del BOLD medio sobre el T1w",
                )
            )

            print(
                "Fondo: BOLD medio | Azul: CSF | "
                "Naranja: GM | Verde: WM"
            )
            display(
                plotting.view_img(
                    str(fila["segmentation_path"]),
                    bg_img=str(fila["boldref_t1_path"]),
                    cmap=TEJIDOS_FUNC_CMAP,
                    symmetric_cmap=False,
                    threshold=0.5,
                    vmin=1,
                    vmax=3,
                    colorbar=False,
                    black_bg=True,
                    draw_cross=True,
                    dim=0,
                    opacity=0.5,
                    resampling_interpolation="nearest",
                    width_view=700,
                    title="Tejidos anatómicos sobre el BOLD corregistrado",
                )
            )

    selector_sujeto.observe(actualizar, names="value")
    display(selector_sujeto, panel)
    actualizar()


# -----------------------------------------------------------------------------
# Normalización funcional a MNI152
# -----------------------------------------------------------------------------


def _fila_normalizacion_correspondiente(
    fila_funcional,
    resultados_normalizacion,
):
    """Selecciona la normalización anatómica del mismo sujeto y sesión."""

    sujeto = _entidad_limpia(fila_funcional.subject, "sub-")
    sesion = _entidad_limpia(fila_funcional.session, "ses-")
    candidatos = resultados_normalizacion.copy()
    candidatos = candidatos[
        candidatos["subject"].map(
            lambda valor: _entidad_limpia(valor, "sub-") == sujeto
        )
    ]

    if sesion is not None:
        candidatos = candidatos[
            candidatos["session"].map(
                lambda valor: _entidad_limpia(valor, "ses-") == sesion
            )
        ]

    if len(candidatos) != 1:
        raise ValueError(
            f"Se esperaba una normalización T1w para sub-{sujeto}; "
            f"se encontraron {len(candidatos)}."
        )
    return candidatos.iloc[0]


def _aplicar_transformaciones_4d_mni(
    bold_path,
    template_path,
    transformaciones,
    output_path,
    mean_path,
    tr,
):
    """Normaliza un BOLD por volumen usando una sola composición espacial."""

    bold = ants.image_read(str(bold_path))
    template = ants.image_read(str(template_path))
    if bold.dimension != 4 or template.dimension != 3:
        raise ValueError("Se esperaba un BOLD 4D y una plantilla MNI 3D.")

    volumenes = int(bold.shape[3])
    shape_salida = tuple(int(x) for x in template.shape) + (volumenes,)
    output_path = Path(output_path)

    with tempfile.TemporaryDirectory(
        prefix="bold_mni_",
        dir=output_path.parent,
    ) as temporal:
        memmap_path = Path(temporal) / "bold_mni_float32.dat"
        datos_salida = np.memmap(
            memmap_path,
            dtype=np.float32,
            mode="w+",
            shape=shape_salida,
        )
        suma = np.zeros(template.shape, dtype=np.float64)

        for indice in range(volumenes):
            volumen = ants.slice_image(bold, axis=3, idx=indice)
            volumen_mni = ants.apply_transforms(
                fixed=template,
                moving=volumen,
                transformlist=[str(ruta) for ruta in transformaciones],
                interpolator="linear",
            )
            datos = volumen_mni.numpy().astype(np.float32, copy=False)
            datos_salida[..., indice] = datos
            suma += datos

            if (indice + 1) % 25 == 0 or indice + 1 == volumenes:
                print(f"        volumen {indice + 1}/{volumenes}")

        datos_salida.flush()
        template_nib = nib.load(template_path)
        header = template_nib.header.copy()
        header.set_data_dtype(np.float32)
        header.set_data_shape(shape_salida)
        header.set_zooms(
            tuple(float(x) for x in template_nib.header.get_zooms()[:3])
            + (float(tr),)
        )
        salida = nib.Nifti1Image(
            datos_salida,
            template_nib.affine,
            header,
        )
        nib.save(salida, output_path)

        media = (suma / volumenes).astype(np.float32)
        mean_header = template_nib.header.copy()
        mean_header.set_data_dtype(np.float32)
        nib.save(
            nib.Nifti1Image(media, template_nib.affine, mean_header),
            mean_path,
        )
        del salida
        del datos_salida


def _metricas_solapamiento_mni(mask_path, template_path):
    """Calcula la cobertura de la máscara funcional sobre MNI152."""

    mascara = np.asarray(nib.load(mask_path).dataobj) > 0
    template = np.asarray(nib.load(template_path).dataobj) > 0
    if mascara.shape != template.shape:
        raise RuntimeError("La máscara funcional no coincide con MNI152.")
    if not mascara.any() or not template.any():
        raise RuntimeError("La máscara funcional o la plantilla está vacía.")

    interseccion = np.count_nonzero(mascara & template)
    dice = 2 * interseccion / (mascara.sum() + template.sum())
    epi_dentro = interseccion / mascara.sum()
    mni_cubierto = interseccion / template.sum()
    return dice, epi_dentro, mni_cubierto


def normalizar_bold_mni_lote(
    resultados_coregistro,
    resultados_normalizacion,
    overwrite=False,
):
    """Compone BOLD→T1w y T1w→MNI y normaliza cada serie una sola vez."""

    if resultados_coregistro.empty:
        raise ValueError("No hay corregistraciones funcionales.")
    if resultados_normalizacion.empty:
        raise ValueError("No hay normalizaciones anatómicas.")

    resultados = []

    for numero, fila in enumerate(
        resultados_coregistro.itertuples(index=False),
        start=1,
    ):
        anat = _fila_normalizacion_correspondiente(
            fila,
            resultados_normalizacion,
        )
        bold_path = Path(fila.bold_native_path)
        template_path = Path(anat.template_path)
        sufijo = "_desc-mocoStc_bold.nii.gz"
        if not bold_path.name.endswith(sufijo):
            raise ValueError(f"No se reconoce el BOLD nativo: {bold_path.name}")

        prefijo = bold_path.name[: -len(sufijo)]
        output_dir = bold_path.parent
        normalized_path = (
            output_dir / f"{prefijo}_space-MNI152_desc-preproc_bold.nii.gz"
        )
        mean_path = (
            output_dir / f"{prefijo}_space-MNI152_desc-mean_bold.nii.gz"
        )
        mask_path = (
            output_dir / f"{prefijo}_space-MNI152_desc-brain_mask.nii.gz"
        )
        productos = (normalized_path, mean_path, mask_path)

        transformaciones_t1_mni = [
            Path(ruta) for ruta in anat.forward_transforms
        ]
        transformaciones_bold_t1 = [
            Path(ruta) for ruta in fila.bold_to_t1_transforms
        ]
        transformaciones = (
            transformaciones_t1_mni + transformaciones_bold_t1
        )
        faltantes = [ruta for ruta in transformaciones if not ruta.exists()]
        if faltantes:
            raise FileNotFoundError(
                "Faltan transformaciones: "
                + ", ".join(ruta.name for ruta in faltantes)
            )

        print(
            f"[{numero}/{len(resultados_coregistro)}] "
            f"sub-{fila.subject}: BOLD → MNI152"
        )

        if all(ruta.exists() for ruta in productos) and not overwrite:
            estado = "reutilizado"
            segundos = 0.0
        else:
            existentes = [ruta for ruta in productos if ruta.exists()]
            if existentes and not overwrite:
                raise RuntimeError(
                    f"Hay productos MNI incompletos para sub-{fila.subject}. "
                    "Usa overwrite=True."
                )

            inicio = perf_counter()
            _aplicar_transformaciones_4d_mni(
                bold_path=bold_path,
                template_path=template_path,
                transformaciones=transformaciones,
                output_path=normalized_path,
                mean_path=mean_path,
                tr=fila.TR_s,
            )
            mean_mni = ants.image_read(str(mean_path))
            mask_mni = ants.get_mask(mean_mni, cleanup=2)
            if np.count_nonzero(mask_mni.numpy()) == 0:
                raise RuntimeError(
                    f"La máscara MNI quedó vacía para sub-{fila.subject}."
                )
            ants.image_write(mask_mni, str(mask_path))
            estado = "procesado"
            segundos = round(perf_counter() - inicio, 1)

        salida_img = nib.load(normalized_path)
        template_img = nib.load(template_path)
        if salida_img.shape[:3] != template_img.shape:
            raise RuntimeError(
                f"La salida MNI de sub-{fila.subject} no coincide "
                "con la plantilla."
            )
        dice, epi_dentro, mni_cubierto = _metricas_solapamiento_mni(
            mask_path,
            template_path,
        )

        resultado = fila._asdict()
        resultado.update(
            {
                "mni_template_path": template_path,
                "normalized_bold_path": normalized_path,
                "normalized_mean_path": mean_path,
                "normalized_mask_path": mask_path,
                "composed_transforms": transformaciones,
                "normalization_status": estado,
                "normalization_seconds": segundos,
                "dice_epi_mni": round(float(dice), 3),
                "epi_inside_mni_pct": round(float(epi_dentro * 100), 1),
                "mni_covered_by_epi_pct": round(
                    float(mni_cubierto * 100),
                    1,
                ),
            }
        )
        resultados.append(resultado)
        print(
            f"    {estado} | Dice: {resultado['dice_epi_mni']:.3f} | "
            f"EPI dentro de MNI: {resultado['epi_inside_mni_pct']:.1f}%"
        )

    return pd.DataFrame(resultados)


def _bold_medio_sin_fondo(mean_path, mask_path):
    """Aplica la máscara funcional al BOLD medio para QC visual."""

    media_img = nib.load(mean_path)
    media = np.asarray(media_img.dataobj, dtype=np.float32)
    mascara = np.asarray(nib.load(mask_path).dataobj) > 0
    salida = np.where(mascara, media, 0).astype(np.float32)
    header = media_img.header.copy()
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(salida, media_img.affine, header)


def mostrar_qc_normalizacion_bold(resultados_normalizados):
    """Muestra el BOLD medio y sus bordes sobre la plantilla MNI152."""

    if resultados_normalizados.empty:
        raise ValueError("No hay BOLD normalizados para visualizar.")

    opciones = []
    for posicion, fila in enumerate(resultados_normalizados.itertuples()):
        etiqueta = f"sub-{fila.subject}"
        if fila.session:
            etiqueta += f" | ses-{fila.session}"
        opciones.append((etiqueta, posicion))

    selector_sujeto = widgets.Dropdown(
        options=opciones,
        description="Sujeto:",
    )
    panel = widgets.Output()

    def actualizar(change=None):
        fila = resultados_normalizados.iloc[selector_sujeto.value]

        with panel:
            clear_output(wait=True)
            print(
                f"Dice EPI–MNI: {fila['dice_epi_mni']:.3f} | "
                f"EPI dentro de MNI: {fila['epi_inside_mni_pct']:.1f}% | "
                f"MNI cubierto: {fila['mni_covered_by_epi_pct']:.1f}%"
            )

            print("Gris: MNI152 | Color: BOLD medio normalizado")
            bold_medio = _bold_medio_sin_fondo(
                fila["normalized_mean_path"],
                fila["normalized_mask_path"],
            )
            display(
                plotting.view_img(
                    bold_medio,
                    bg_img=str(fila["mni_template_path"]),
                    cmap="autumn",
                    symmetric_cmap=False,
                    threshold=1e-6,
                    colorbar=False,
                    black_bg=True,
                    draw_cross=True,
                    dim=0,
                    opacity=0.5,
                    resampling_interpolation="nearest",
                    width_view=700,
                    title="BOLD medio sobre MNI152",
                )
            )

            print("Gris: MNI152 | Cian: bordes del BOLD medio")
            bordes = _bordes_funcionales(fila["normalized_mean_path"])
            display(
                plotting.view_img(
                    bordes,
                    bg_img=str(fila["mni_template_path"]),
                    cmap=BORDES_FUNC_CMAP,
                    symmetric_cmap=False,
                    threshold=1e-6,
                    colorbar=False,
                    black_bg=True,
                    draw_cross=True,
                    dim=0,
                    opacity=0.9,
                    resampling_interpolation="nearest",
                    width_view=700,
                    title="Bordes del BOLD medio sobre MNI152",
                )
            )

    selector_sujeto.observe(actualizar, names="value")
    display(selector_sujeto, panel)
    actualizar()


# -----------------------------------------------------------------------------
# Extracción de componentes de ruido con aCompCor
# -----------------------------------------------------------------------------


def _fila_segmentacion_correspondiente(
    fila_funcional,
    resultados_segmentacion,
):
    """Selecciona la segmentación anatómica del mismo sujeto y sesión."""

    sujeto = _entidad_limpia(fila_funcional.subject, "sub-")
    sesion = _entidad_limpia(fila_funcional.session, "ses-")
    candidatos = resultados_segmentacion.copy()
    candidatos = candidatos[
        candidatos["subject"].map(
            lambda valor: _entidad_limpia(valor, "sub-") == sujeto
        )
    ]

    if sesion is not None:
        candidatos = candidatos[
            candidatos["session"].map(
                lambda valor: _entidad_limpia(valor, "ses-") == sesion
            )
        ]

    if len(candidatos) != 1:
        raise ValueError(
            f"Se esperaba una segmentación T1w para sub-{sujeto}; "
            f"se encontraron {len(candidatos)}."
        )
    return candidatos.iloc[0]


def _crear_mascara_acompcor_mni(
    probability_path,
    template_path,
    functional_mask_path,
    forward_transforms,
    output_path,
    probability_threshold,
    erosion_iterations,
    tissue,
):
    """Transforma una probabilidad tisular a MNI y conserva su interior."""

    template = ants.image_read(str(template_path))
    probability = ants.image_read(str(probability_path))
    probability_mni = ants.apply_transforms(
        fixed=template,
        moving=probability,
        transformlist=[str(ruta) for ruta in forward_transforms],
        interpolator="linear",
    )

    datos = probability_mni.numpy()
    mascara_funcional = (
        np.asarray(nib.load(functional_mask_path).dataobj) > 0
    )
    mascara = (
        np.isfinite(datos)
        & (datos >= probability_threshold)
        & mascara_funcional
    )

    if erosion_iterations > 0:
        mascara = binary_erosion(
            mascara,
            iterations=erosion_iterations,
            border_value=0,
        )

    voxeles = int(np.count_nonzero(mascara))
    if voxeles == 0:
        raise RuntimeError(
            f"La máscara aCompCor de {tissue} quedó vacía. "
            "Reduce probability_threshold o usa erosion_iterations=0."
        )

    referencia = nib.load(template_path)
    header = referencia.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(
        nib.Nifti1Image(
            mascara.astype(np.uint8),
            referencia.affine,
            header,
        ),
        output_path,
    )
    return voxeles


def _extraer_series_acompcor(
    bold_path,
    wm_mask_path,
    csf_mask_path,
    block_size=25,
):
    """Lee el BOLD por bloques y extrae WM y CSF sin cargarlo completo."""

    bold = nib.load(bold_path)
    if len(bold.shape) != 4:
        raise ValueError("El BOLD normalizado debe ser 4D.")

    wm = np.asarray(nib.load(wm_mask_path).dataobj) > 0
    csf = np.asarray(nib.load(csf_mask_path).dataobj) > 0
    if wm.shape != bold.shape[:3] or csf.shape != bold.shape[:3]:
        raise RuntimeError("Las máscaras aCompCor no coinciden con el BOLD.")

    indices_wm = np.flatnonzero(wm.ravel())
    indices_csf = np.flatnonzero(csf.ravel())
    volumenes = int(bold.shape[3])
    series_wm = np.empty((indices_wm.size, volumenes), dtype=np.float32)
    series_csf = np.empty((indices_csf.size, volumenes), dtype=np.float32)

    for inicio in range(0, volumenes, block_size):
        fin = min(inicio + block_size, volumenes)
        bloque = np.asarray(
            bold.dataobj[..., inicio:fin],
            dtype=np.float32,
        ).reshape(-1, fin - inicio)
        series_wm[:, inicio:fin] = bloque[indices_wm]
        series_csf[:, inicio:fin] = bloque[indices_csf]
        del bloque

    return series_wm, series_csf


def _pca_temporal(series, n_components, tissue):
    """Calcula componentes temporales y su fracción de varianza."""

    series = np.asarray(series, dtype=np.float32)
    validas = np.isfinite(series).all(axis=1)
    series = series[validas]
    if series.shape[0] < n_components:
        raise RuntimeError(
            f"Solo hay {series.shape[0]} vóxeles válidos de {tissue}; "
            f"se pidieron {n_components} componentes."
        )

    series = detrend(series, axis=1, type="linear")
    desviacion = series.std(axis=1, ddof=1)
    series = series[desviacion > np.finfo(np.float32).eps]
    desviacion = desviacion[desviacion > np.finfo(np.float32).eps]
    if series.shape[0] < n_components:
        raise RuntimeError(
            f"No hay suficientes vóxeles variables de {tissue} para PCA."
        )

    series /= desviacion[:, None]
    # La matriz temporal es pequeña (T x T) y evita una SVD costosa de V x T.
    matriz_temporal = series.T @ series
    valores_propios, vectores_propios = np.linalg.eigh(matriz_temporal)
    orden = np.argsort(valores_propios)[::-1]
    valores_propios = np.clip(valores_propios[orden], 0, None)
    vectores_temporales = vectores_propios[:, orden]
    varianza = valores_propios
    varianza /= varianza.sum()

    componentes = vectores_temporales[:, :n_components]
    for indice in range(componentes.shape[1]):
        pico = int(np.argmax(np.abs(componentes[:, indice])))
        if componentes[pico, indice] < 0:
            componentes[:, indice] *= -1
    componentes -= componentes.mean(axis=0, keepdims=True)
    componentes /= np.maximum(
        componentes.std(axis=0, ddof=1, keepdims=True),
        np.finfo(np.float32).eps,
    )
    return componentes.astype(np.float32), varianza.astype(float)


def _n_componentes_para_varianza(varianza, umbral=0.50):
    """Número mínimo de componentes que alcanza una varianza acumulada."""

    alcanzan = np.flatnonzero(np.cumsum(varianza) >= umbral)
    return int(alcanzan[0] + 1) if alcanzan.size else int(len(varianza))


def extraer_acompcor_lote(
    resultados_bold_mni,
    resultados_segmentacion,
    resultados_normalizacion,
    n_components=5,
    probability_threshold=0.80,
    erosion_iterations=1,
    overwrite=False,
):
    """Extrae componentes aCompCor separados de WM y CSF."""

    if resultados_bold_mni.empty:
        raise ValueError("No hay BOLD normalizados para aCompCor.")

    n_components = int(n_components)
    erosion_iterations = int(erosion_iterations)
    if n_components < 1:
        raise ValueError("n_components debe ser al menos 1.")
    if not 0 < probability_threshold < 1:
        raise ValueError("probability_threshold debe estar entre 0 y 1.")
    if erosion_iterations < 0:
        raise ValueError("erosion_iterations no puede ser negativo.")

    resultados = []

    for numero, fila in enumerate(
        resultados_bold_mni.itertuples(index=False),
        start=1,
    ):
        segmentacion = _fila_segmentacion_correspondiente(
            fila,
            resultados_segmentacion,
        )
        normalizacion = _fila_normalizacion_correspondiente(
            fila,
            resultados_normalizacion,
        )
        forward = [Path(ruta) for ruta in normalizacion.forward_transforms]
        faltantes = [ruta for ruta in forward if not ruta.exists()]
        if faltantes:
            raise FileNotFoundError(
                "Faltan transformaciones anatómicas: "
                + ", ".join(ruta.name for ruta in faltantes)
            )

        bold_path = Path(fila.normalized_bold_path)
        sufijo = "_space-MNI152_desc-preproc_bold.nii.gz"
        if not bold_path.name.endswith(sufijo):
            raise ValueError(f"No se reconoce el BOLD MNI: {bold_path.name}")
        prefijo = bold_path.name[: -len(sufijo)]
        output_dir = bold_path.parent
        wm_mask_path = output_dir / (
            f"{prefijo}_space-MNI152_label-WM_desc-acompcor_mask.nii.gz"
        )
        csf_mask_path = output_dir / (
            f"{prefijo}_space-MNI152_label-CSF_desc-acompcor_mask.nii.gz"
        )
        confounds_path = output_dir / (
            f"{prefijo}_desc-acompcor_timeseries.tsv"
        )
        metadata_path = output_dir / (
            f"{prefijo}_desc-acompcor_timeseries.json"
        )
        productos = (
            wm_mask_path,
            csf_mask_path,
            confounds_path,
            metadata_path,
        )

        print(
            f"[{numero}/{len(resultados_bold_mni)}] "
            f"sub-{fila.subject}: aCompCor"
        )
        inicio = perf_counter()

        if all(ruta.exists() for ruta in productos) and not overwrite:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            configuracion = metadata["configuration"]
            esperado = {
                "n_components": n_components,
                "probability_threshold": float(probability_threshold),
                "erosion_iterations": erosion_iterations,
            }
            if configuracion != esperado:
                raise RuntimeError(
                    f"La salida aCompCor de sub-{fila.subject} usa otros "
                    "parámetros. Ejecuta una vez con overwrite=True."
                )
            estado = "reutilizado"
            segundos = 0.0
        else:
            existentes = [ruta for ruta in productos if ruta.exists()]
            if existentes and not overwrite:
                raise RuntimeError(
                    f"Hay productos aCompCor incompletos para "
                    f"sub-{fila.subject}. Usa overwrite=True."
                )

            wm_voxels = _crear_mascara_acompcor_mni(
                probability_path=segmentacion.wm_path,
                template_path=fila.mni_template_path,
                functional_mask_path=fila.normalized_mask_path,
                forward_transforms=forward,
                output_path=wm_mask_path,
                probability_threshold=probability_threshold,
                erosion_iterations=erosion_iterations,
                tissue="WM",
            )
            csf_voxels = _crear_mascara_acompcor_mni(
                probability_path=segmentacion.csf_path,
                template_path=fila.mni_template_path,
                functional_mask_path=fila.normalized_mask_path,
                forward_transforms=forward,
                output_path=csf_mask_path,
                probability_threshold=probability_threshold,
                erosion_iterations=erosion_iterations,
                tissue="CSF",
            )
            series_wm, series_csf = _extraer_series_acompcor(
                bold_path,
                wm_mask_path,
                csf_mask_path,
            )
            componentes_wm, varianza_wm = _pca_temporal(
                series_wm,
                n_components,
                "WM",
            )
            componentes_csf, varianza_csf = _pca_temporal(
                series_csf,
                n_components,
                "CSF",
            )
            del series_wm, series_csf

            columnas_wm = [
                f"wm_comp_cor_{indice:02d}"
                for indice in range(n_components)
            ]
            columnas_csf = [
                f"csf_comp_cor_{indice:02d}"
                for indice in range(n_components)
            ]
            confounds = pd.DataFrame(
                np.column_stack([componentes_wm, componentes_csf]),
                columns=columnas_wm + columnas_csf,
            )
            confounds.to_csv(confounds_path, sep="\t", index=False)

            metadata = {
                "algorithm": ALGORITMO_ACOMPCOR,
                "configuration": {
                    "n_components": n_components,
                    "probability_threshold": float(probability_threshold),
                    "erosion_iterations": erosion_iterations,
                },
                "wm": {
                    "mask_voxels": wm_voxels,
                    "variance_ratio": varianza_wm.tolist(),
                    "variance_first_n": float(
                        varianza_wm[:n_components].sum()
                    ),
                    "components_for_50pct": _n_componentes_para_varianza(
                        varianza_wm
                    ),
                },
                "csf": {
                    "mask_voxels": csf_voxels,
                    "variance_ratio": varianza_csf.tolist(),
                    "variance_first_n": float(
                        varianza_csf[:n_components].sum()
                    ),
                    "components_for_50pct": _n_componentes_para_varianza(
                        varianza_csf
                    ),
                },
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            estado = "procesado"
            segundos = round(perf_counter() - inicio, 1)

        resultado = fila._asdict()
        resultado.update(
            {
                "normalized_t1_path": Path(normalizacion.normalized_path),
                "wm_acompcor_mask_path": wm_mask_path,
                "csf_acompcor_mask_path": csf_mask_path,
                "acompcor_path": confounds_path,
                "acompcor_metadata_path": metadata_path,
                "acompcor_status": estado,
                "acompcor_seconds": segundos,
                "wm_mask_voxels": int(metadata["wm"]["mask_voxels"]),
                "csf_mask_voxels": int(metadata["csf"]["mask_voxels"]),
                "wm_variance_first_n_pct": round(
                    100 * float(metadata["wm"]["variance_first_n"]),
                    1,
                ),
                "csf_variance_first_n_pct": round(
                    100 * float(metadata["csf"]["variance_first_n"]),
                    1,
                ),
                "wm_components_for_50pct": int(
                    metadata["wm"]["components_for_50pct"]
                ),
                "csf_components_for_50pct": int(
                    metadata["csf"]["components_for_50pct"]
                ),
            }
        )
        resultados.append(resultado)
        print(
            f"    {estado} | WM: {resultado['wm_mask_voxels']} vóxeles | "
            f"CSF: {resultado['csf_mask_voxels']} vóxeles"
        )

    return pd.DataFrame(resultados)


def _mapa_mascaras_acompcor(csf_path, wm_path):
    """Crea un mapa etiquetado: 1=CSF y 2=WM."""

    csf_img = nib.load(csf_path)
    csf = np.asarray(csf_img.dataobj) > 0
    wm = np.asarray(nib.load(wm_path).dataobj) > 0
    etiquetas = np.zeros(csf.shape, dtype=np.uint8)
    etiquetas[csf] = 1
    etiquetas[wm] = 2
    header = csf_img.header.copy()
    header.set_data_dtype(np.uint8)
    return nib.Nifti1Image(etiquetas, csf_img.affine, header)


def mostrar_qc_acompcor(resultados_acompcor):
    """Muestra máscaras, varianza acumulada y primeros componentes."""

    if resultados_acompcor.empty:
        raise ValueError("No hay resultados aCompCor para revisar.")

    opciones = []
    for posicion, fila in enumerate(resultados_acompcor.itertuples()):
        etiqueta = f"sub-{fila.subject}"
        if fila.session:
            etiqueta += f" | ses-{fila.session}"
        opciones.append((etiqueta, posicion))

    selector = widgets.Dropdown(options=opciones, description="Sujeto:")
    panel = widgets.Output()

    def actualizar(change=None):
        fila = resultados_acompcor.iloc[selector.value]
        metadata = json.loads(
            Path(fila["acompcor_metadata_path"]).read_text(encoding="utf-8")
        )
        confounds = pd.read_csv(fila["acompcor_path"], sep="\t")
        n_componentes = metadata["configuration"]["n_components"]

        with panel:
            clear_output(wait=True)
            print(
                f"WM: {fila['wm_mask_voxels']} vóxeles; "
                f"primeros {n_componentes}: "
                f"{fila['wm_variance_first_n_pct']:.1f}% de varianza"
            )
            print(
                f"CSF: {fila['csf_mask_voxels']} vóxeles; "
                f"primeros {n_componentes}: "
                f"{fila['csf_variance_first_n_pct']:.1f}% de varianza"
            )
            display(
                HTML(
                    '<span style="color:#00B8D9">■</span> CSF &nbsp; '
                    '<span style="color:#F4A261">■</span> WM'
                )
            )
            display(
                plotting.view_img(
                    _mapa_mascaras_acompcor(
                        fila["csf_acompcor_mask_path"],
                        fila["wm_acompcor_mask_path"],
                    ),
                    bg_img=str(fila["normalized_t1_path"]),
                    cmap=COMPCOR_CMAP,
                    symmetric_cmap=False,
                    threshold=0.5,
                    vmin=0.5,
                    vmax=2.5,
                    colorbar=False,
                    black_bg=True,
                    draw_cross=True,
                    opacity=0.65,
                    resampling_interpolation="nearest",
                    width_view=700,
                    title="Máscaras erosionadas usadas por aCompCor",
                )
            )

            varianza_wm = np.asarray(metadata["wm"]["variance_ratio"])
            varianza_csf = np.asarray(metadata["csf"]["variance_ratio"])
            limite = min(25, len(varianza_wm), len(varianza_csf))
            eje_x = np.arange(1, limite + 1)

            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            axes[0].plot(
                eje_x,
                np.cumsum(varianza_wm)[:limite],
                marker="o",
                markersize=3,
                label="WM",
                color="#F4A261",
            )
            axes[0].plot(
                eje_x,
                np.cumsum(varianza_csf)[:limite],
                marker="o",
                markersize=3,
                label="CSF",
                color="#00B8D9",
            )
            axes[0].axhline(0.50, color="grey", linestyle=":")
            axes[0].axvline(n_componentes, color="black", linestyle="--")
            axes[0].set_xlabel("Número de componentes")
            axes[0].set_ylabel("Varianza acumulada")
            axes[0].set_title("Varianza explicada por aCompCor")
            axes[0].legend(frameon=False)

            axes[1].plot(
                confounds["wm_comp_cor_00"],
                color="#F4A261",
                linewidth=0.9,
                label="WM 1",
            )
            axes[1].plot(
                confounds["csf_comp_cor_00"],
                color="#00B8D9",
                linewidth=0.9,
                label="CSF 1",
            )
            axes[1].set_xlabel("Volumen")
            axes[1].set_ylabel("Unidades típicas")
            axes[1].set_title("Primer componente de cada tejido")
            axes[1].legend(frameon=False)
            plt.tight_layout()
            plt.show()

    selector.observe(actualizar, names="value")
    display(selector, panel)
    actualizar()


# -----------------------------------------------------------------------------
# Denoising temporal
# -----------------------------------------------------------------------------


def _extraer_series_mascara(
    bold_path,
    mask_path,
    block_size=100,
):
    """Extrae los vóxeles cerebrales leyendo el BOLD por bloques temporales."""

    bold = nib.load(bold_path)
    mascara = np.asarray(nib.load(mask_path).dataobj) > 0
    if len(bold.shape) != 4 or mascara.shape != bold.shape[:3]:
        raise RuntimeError("El BOLD y su máscara funcional no coinciden.")

    indices = np.flatnonzero(mascara.ravel())
    volumenes = int(bold.shape[3])
    series = np.empty((indices.size, volumenes), dtype=np.float32)

    for inicio in range(0, volumenes, block_size):
        fin = min(inicio + block_size, volumenes)
        bloque = np.asarray(
            bold.dataobj[..., inicio:fin],
            dtype=np.float32,
        ).reshape(-1, fin - inicio)
        series[:, inicio:fin] = bloque[indices]
        del bloque

    return bold, indices, series


def _regresores_24hmp(motion_path):
    """Construye 24 parámetros: movimiento, derivadas y cuadrados."""

    columnas = [
        "trans_x_mm",
        "trans_y_mm",
        "trans_z_mm",
        "rot_x_rad",
        "rot_y_rad",
        "rot_z_rad",
    ]
    movimiento = pd.read_csv(motion_path, sep="\t")
    base = movimiento[columnas].fillna(0.0).to_numpy(dtype=np.float32)
    derivadas = np.vstack(
        [np.zeros((1, base.shape[1]), dtype=np.float32), np.diff(base, axis=0)]
    )
    regresores = np.column_stack(
        [base, derivadas, base**2, derivadas**2]
    ).astype(np.float32)
    nombres = (
        columnas
        + [f"{nombre}_derivative1" for nombre in columnas]
        + [f"{nombre}_power2" for nombre in columnas]
        + [f"{nombre}_derivative1_power2" for nombre in columnas]
    )
    return regresores, nombres


def _filtrar_banda_en_sitio(
    series,
    tr,
    low_pass,
    high_pass,
    block_size=20000,
):
    """Aplica un Butterworth paso banda por bloques de vóxeles."""

    nyquist = 0.5 / float(tr)
    if not 0 < high_pass < low_pass < nyquist:
        raise ValueError(
            f"La banda debe cumplir 0 < high_pass < low_pass < {nyquist:.3f}."
        )

    sos = butter(
        2,
        [high_pass / nyquist, low_pass / nyquist],
        btype="bandpass",
        output="sos",
    )
    padlen = min(
        int(3.0 / (high_pass * float(tr))),
        series.shape[1] - 1,
    )

    for inicio in range(0, series.shape[0], block_size):
        fin = min(inicio + block_size, series.shape[0])
        series[inicio:fin] = sosfiltfilt(
            sos,
            series[inicio:fin],
            axis=1,
            padlen=padlen,
        ).astype(np.float32)
    return series


def _dvars_por_bloques(
    series,
    volume_indices=None,
    block_size=20000,
):
    """Calcula DVARS sin crear una matriz de diferencias completa."""

    if volume_indices is None:
        numero_volumenes = series.shape[1]
    else:
        volume_indices = np.asarray(volume_indices, dtype=int)
        numero_volumenes = len(volume_indices)

    suma = np.zeros(numero_volumenes - 1, dtype=np.float64)
    total_voxeles = 0
    for inicio in range(0, series.shape[0], block_size):
        fin = min(inicio + block_size, series.shape[0])
        bloque = series[inicio:fin]
        if volume_indices is not None:
            bloque = bloque[:, volume_indices]
        diferencias = np.diff(bloque, axis=1)
        suma += np.square(diferencias, dtype=np.float64).sum(axis=0)
        total_voxeles += fin - inicio
    return np.sqrt(suma / total_voxeles)


def _correlacion_segura(x, y):
    """Correlación de Pearson o NaN cuando no puede estimarse."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    validos = np.isfinite(x) & np.isfinite(y)
    if validos.sum() < 3:
        return np.nan
    x = x[validos]
    y = y[validos]
    if np.std(x) <= 0 or np.std(y) <= 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _gcor_muestra(series):
    """Correlación media entre todos los pares de una muestra de vóxeles."""

    z = np.asarray(series, dtype=np.float32)
    z = z - z.mean(axis=1, keepdims=True)
    z /= np.maximum(
        np.linalg.norm(z, axis=1, keepdims=True),
        np.finfo(np.float32).eps,
    )
    correlaciones = z @ z.T
    n = correlaciones.shape[0]
    if n < 2:
        return np.nan
    return float(
        (correlaciones.sum() - np.trace(correlaciones)) / (n * (n - 1))
    )


def _guardar_bold_desruido(
    bold,
    indices_mascara,
    series_limpias,
    output_path,
    tr,
    temporal_dir,
    block_size=20000,
):
    """Reconstruye el NIfTI 4D desde las series cerebrales limpias."""

    shape = tuple(int(x) for x in bold.shape[:3]) + (
        int(series_limpias.shape[1]),
    )
    datos_path = Path(temporal_dir) / "denoised_bold_float32.dat"
    datos = np.memmap(datos_path, mode="w+", dtype=np.float32, shape=shape)
    datos[:] = 0
    datos_planos = datos.reshape(-1, shape[3])

    for inicio in range(0, len(indices_mascara), block_size):
        fin = min(inicio + block_size, len(indices_mascara))
        datos_planos[indices_mascara[inicio:fin]] = series_limpias[inicio:fin]
    datos.flush()

    header = bold.header.copy()
    header.set_data_dtype(np.float32)
    header.set_data_shape(shape)
    header.set_zooms(
        tuple(float(x) for x in bold.header.get_zooms()[:3]) + (float(tr),)
    )
    salida = nib.Nifti1Image(datos, bold.affine, header)
    nib.save(salida, output_path)
    del salida, datos_planos, datos


def _procesar_denoising(
    fila,
    output_path,
    censoring_path,
    metadata_path,
    qc_path,
    high_pass,
    low_pass,
    fd_threshold,
    gschange_threshold,
    n_acompcor,
    fast_mode,
):
    """Filtra y regresa confounds con memoria acotada."""

    bold_path = Path(fila.normalized_bold_path)
    mask_path = Path(fila.normalized_mask_path)
    tr = float(fila.TR_s)
    print("    1/4 Leyendo únicamente los vóxeles de la máscara...")
    bold, indices_mascara, series = _extraer_series_mascara(
        bold_path,
        mask_path,
    )
    volumenes = int(series.shape[1])

    movimiento, nombres_movimiento = _regresores_24hmp(fila.motion_path)
    acompcor = pd.read_csv(fila.acompcor_path, sep="\t")
    columnas_wm = [
        nombre for nombre in acompcor.columns if nombre.startswith("wm_comp_cor_")
    ][:n_acompcor]
    columnas_csf = [
        nombre
        for nombre in acompcor.columns
        if nombre.startswith("csf_comp_cor_")
    ][:n_acompcor]
    if len(columnas_wm) != n_acompcor or len(columnas_csf) != n_acompcor:
        raise RuntimeError(
            f"Se necesitan {n_acompcor} componentes de WM y CSF."
        )
    componentes = acompcor[columnas_wm + columnas_csf].to_numpy(
        dtype=np.float32
    )
    if len(movimiento) != volumenes or len(componentes) != volumenes:
        raise RuntimeError(
            "El BOLD, el movimiento y aCompCor no tienen la misma longitud."
        )

    metricas_movimiento = _cargar_metricas_movimiento(
        fila.motion_path,
        fila.fd_path,
    )
    fd = metricas_movimiento["fd_caja_mm"].to_numpy(dtype=float)
    senal_global = series.mean(axis=0, dtype=np.float64)
    cambio_global = np.concatenate([[0.0], np.abs(np.diff(senal_global))])
    gschange_std = _escalar_robusto(cambio_global)
    censurados = (fd > fd_threshold) | (gschange_std > gschange_threshold)
    validos = ~censurados
    indices_validos = np.flatnonzero(validos)

    regresores = np.column_stack([movimiento, componentes]).astype(np.float32)
    nombres_regresores = nombres_movimiento + columnas_wm + columnas_csf
    print("    2/4 Filtrando BOLD y regresores en la misma banda...")
    _filtrar_banda_en_sitio(
        regresores.T,
        tr,
        low_pass,
        high_pass,
    )
    _filtrar_banda_en_sitio(
        series,
        tr,
        low_pass,
        high_pass,
    )

    regresores_validos = regresores[validos]
    desviacion = regresores_validos.std(axis=0, ddof=1)
    columnas_validas = desviacion > np.finfo(np.float32).eps
    regresores_validos = regresores_validos[:, columnas_validas]
    nombres_regresores = [
        nombre
        for nombre, conservar in zip(nombres_regresores, columnas_validas)
        if conservar
    ]
    regresores_validos -= regresores_validos.mean(axis=0, keepdims=True)
    regresores_validos /= regresores_validos.std(axis=0, ddof=1, keepdims=True)
    diseno = np.column_stack(
        [
            regresores_validos,
            np.ones(len(indices_validos), dtype=np.float32),
        ]
    ).astype(np.float32)
    if len(indices_validos) <= diseno.shape[1] + 10:
        raise RuntimeError(
            "Quedaron muy pocos volúmenes después del censurado para ajustar "
            "el modelo de denoising."
        )
    pseudoinversa = np.linalg.pinv(diseno).astype(np.float32)

    print("    3/4 Regresando movimiento y aCompCor por bloques...")
    with tempfile.TemporaryDirectory(
        prefix="denoising_",
        dir=Path(output_path).parent,
    ) as temporal:
        shape_limpia = (series.shape[0], len(indices_validos))
        if fast_mode:
            limpias = np.lib.format.open_memmap(
                output_path,
                mode="w+",
                dtype=np.float32,
                shape=shape_limpia,
            )
        else:
            limpias_path = Path(temporal) / "clean_signals_float32.dat"
            limpias = np.memmap(
                limpias_path,
                mode="w+",
                dtype=np.float32,
                shape=shape_limpia,
            )
        bloque_voxeles = 20000
        for inicio in range(0, series.shape[0], bloque_voxeles):
            fin = min(inicio + bloque_voxeles, series.shape[0])
            y = series[inicio:fin, validos].T
            limpias[inicio:fin] = (
                y - diseno @ (pseudoinversa @ y)
            ).T.astype(np.float32)
        limpias.flush()

        rng = np.random.default_rng(2026)
        muestra = np.sort(
            rng.choice(
                series.shape[0],
                size=min(500, series.shape[0]),
                replace=False,
            )
        )
        antes_muestra = np.asarray(series[muestra][:, validos], dtype=np.float32)
        despues_muestra = np.asarray(limpias[muestra], dtype=np.float32)

        # Para QC basta una muestra reproducible. Calcular DVARS sobre todos los
        # vóxeles duplicaba tiempo sin cambiar la lectura comparativa del panel.
        dvars_antes = _dvars_por_bloques(antes_muestra)
        dvars_despues = _dvars_por_bloques(despues_muestra)
        adyacentes = np.diff(indices_validos) == 1
        fd_pares = fd[indices_validos[1:]]
        corr_antes = _correlacion_segura(
            fd_pares[adyacentes],
            dvars_antes[adyacentes],
        )
        corr_despues = _correlacion_segura(
            fd_pares[adyacentes],
            dvars_despues[adyacentes],
        )
        std_antes = float(
            np.mean(np.std(antes_muestra, axis=1, ddof=1))
        )
        std_despues = float(
            np.mean(np.std(despues_muestra, axis=1, ddof=1))
        )
        gcor_antes = _gcor_muestra(antes_muestra)
        gcor_despues = _gcor_muestra(despues_muestra)

        print("    4/4 Guardando resultados y métricas QC...")
        if not fast_mode:
            _guardar_bold_desruido(
                bold,
                indices_mascara,
                limpias,
                output_path,
                tr,
                temporal,
            )
        np.savez_compressed(
            qc_path,
            before=antes_muestra,
            after=despues_muestra,
            retained_indices=indices_validos,
            fd=fd[indices_validos],
            dvars_before=np.concatenate([[0.0], dvars_antes]),
            dvars_after=np.concatenate([[0.0], dvars_despues]),
            tr=np.asarray(tr),
        )
        del limpias

    tabla_censurado = pd.DataFrame(
        {
            "original_volume_index": np.arange(volumenes),
            "retained": validos,
            "fd_box_mm": fd,
            "gschange_std": gschange_std,
        }
    )
    tabla_censurado.to_csv(censoring_path, sep="\t", index=False)

    fraccion_banda = (low_pass - high_pass) / (0.5 / tr)
    dof = (len(indices_validos) - diseno.shape[1]) * fraccion_banda
    metadata = {
        "algorithm": ALGORITMO_DENOISING,
        "configuration": {
            "high_pass_hz": float(high_pass),
            "low_pass_hz": float(low_pass),
            "fd_threshold_mm": float(fd_threshold),
            "gschange_threshold": float(gschange_threshold),
            "n_acompcor_per_tissue": int(n_acompcor),
            "fast_mode": bool(fast_mode),
        },
        "regressors": nombres_regresores,
        "original_volumes": volumenes,
        "retained_volumes": int(validos.sum()),
        "censored_volumes": int(censurados.sum()),
        "effective_dof": float(dof),
        "corr_fd_dvars_before": corr_antes,
        "corr_fd_dvars_after": corr_despues,
        "gcor_before": gcor_antes,
        "gcor_after": gcor_despues,
        "temporal_std_retained_pct": float(
            100 * std_despues / max(std_antes, np.finfo(float).eps)
        ),
        "output_format": (
            "masked_voxels_by_time_npy" if fast_mode else "nifti_4d"
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


def aplicar_denoising_lote(
    resultados_acompcor,
    high_pass=0.008,
    low_pass=0.09,
    fd_threshold=0.5,
    gschange_threshold=3.0,
    n_acompcor=5,
    fast_mode=True,
    overwrite=False,
):
    """Aplica el denoising adoptado por el notebook largo."""

    if resultados_acompcor.empty:
        raise ValueError("No hay componentes aCompCor para el denoising.")

    resultados = []
    configuracion = {
        "high_pass_hz": float(high_pass),
        "low_pass_hz": float(low_pass),
        "fd_threshold_mm": float(fd_threshold),
        "gschange_threshold": float(gschange_threshold),
        "n_acompcor_per_tissue": int(n_acompcor),
        "fast_mode": bool(fast_mode),
    }

    for numero, fila in enumerate(
        resultados_acompcor.itertuples(index=False),
        start=1,
    ):
        bold_path = Path(fila.normalized_bold_path)
        sufijo = "_space-MNI152_desc-preproc_bold.nii.gz"
        if not bold_path.name.endswith(sufijo):
            raise ValueError(f"No se reconoce el BOLD MNI: {bold_path.name}")
        prefijo = bold_path.name[: -len(sufijo)]
        output_dir = bold_path.parent
        if fast_mode:
            output_path = output_dir / (
                f"{prefijo}_space-MNI152_desc-denoised_timeseries.npy"
            )
        else:
            output_path = output_dir / (
                f"{prefijo}_space-MNI152_desc-denoised_bold.nii.gz"
            )
        censoring_path = output_dir / (
            f"{prefijo}_desc-censoring_timeseries.tsv"
        )
        metadata_path = output_dir / f"{prefijo}_desc-denoising.json"
        qc_path = output_dir / f"{prefijo}_desc-denoising_qc.npz"
        productos = (output_path, censoring_path, metadata_path, qc_path)

        print(
            f"[{numero}/{len(resultados_acompcor)}] "
            f"sub-{fila.subject}: denoising"
        )
        if all(ruta.exists() for ruta in productos) and not overwrite:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["configuration"] != configuracion:
                raise RuntimeError(
                    f"El denoising existente de sub-{fila.subject} usa otros "
                    "parámetros. Ejecuta una vez con overwrite=True."
                )
            estado = "reutilizado"
            segundos = 0.0
        else:
            existentes = [ruta for ruta in productos if ruta.exists()]
            if existentes and not overwrite:
                raise RuntimeError(
                    f"Hay productos de denoising incompletos para "
                    f"sub-{fila.subject}. Usa overwrite=True."
                )
            inicio = perf_counter()
            metadata = _procesar_denoising(
                fila,
                output_path,
                censoring_path,
                metadata_path,
                qc_path,
                high_pass,
                low_pass,
                fd_threshold,
                gschange_threshold,
                n_acompcor,
                fast_mode,
            )
            estado = "procesado"
            segundos = round(perf_counter() - inicio, 1)

        resultado = fila._asdict()
        resultado.update(
            {
                "denoised_path": output_path,
                "censoring_path": censoring_path,
                "denoising_metadata_path": metadata_path,
                "denoising_qc_path": qc_path,
                "denoised_format": metadata["output_format"],
                "denoising_status": estado,
                "denoising_seconds": segundos,
                "retained_volumes": int(metadata["retained_volumes"]),
                "censored_volumes": int(metadata["censored_volumes"]),
                "effective_dof": round(float(metadata["effective_dof"]), 1),
                "corr_fd_dvars_before": round(
                    float(metadata["corr_fd_dvars_before"]), 3
                ),
                "corr_fd_dvars_after": round(
                    float(metadata["corr_fd_dvars_after"]), 3
                ),
                "gcor_before": round(float(metadata["gcor_before"]), 4),
                "gcor_after": round(float(metadata["gcor_after"]), 4),
                "temporal_std_retained_pct": round(
                    float(metadata["temporal_std_retained_pct"]), 1
                ),
            }
        )
        resultados.append(resultado)
        print(
            f"    {estado} | retenidos: {resultado['retained_volumes']} | "
            f"censurados: {resultado['censored_volumes']} | "
            f"corr FD–DVARS: {resultado['corr_fd_dvars_before']:.3f} → "
            f"{resultado['corr_fd_dvars_after']:.3f}"
        )

    return pd.DataFrame(resultados)


def _estandarizar_filas(datos):
    """Centra y escala cada vóxel para mostrar un carpet plot."""

    datos = np.asarray(datos, dtype=np.float32)
    salida = datos - datos.mean(axis=1, keepdims=True)
    salida /= np.maximum(
        salida.std(axis=1, keepdims=True),
        np.finfo(np.float32).eps,
    )
    return np.clip(salida, -3, 3)


def mostrar_qc_denoising(resultados_denoising):
    """Compara carpet plots, espectro y FD–DVARS antes/después."""

    if resultados_denoising.empty:
        raise ValueError("No hay resultados de denoising para revisar.")

    opciones = []
    for posicion, fila in enumerate(resultados_denoising.itertuples()):
        etiqueta = f"sub-{fila.subject}"
        if fila.session:
            etiqueta += f" | ses-{fila.session}"
        opciones.append((etiqueta, posicion))
    selector = widgets.Dropdown(options=opciones, description="Sujeto:")
    panel = widgets.Output()

    def actualizar(change=None):
        fila = resultados_denoising.iloc[selector.value]
        qc = np.load(fila["denoising_qc_path"])
        antes = qc["before"]
        despues = qc["after"]
        fd = qc["fd"]
        dvars_antes = qc["dvars_before"]
        dvars_despues = qc["dvars_after"]
        tr = float(qc["tr"])

        with panel:
            clear_output(wait=True)
            resumen = pd.DataFrame(
                {
                    "Volúmenes retenidos": [fila["retained_volumes"]],
                    "Volúmenes censurados": [fila["censored_volumes"]],
                    "DOF efectivos": [fila["effective_dof"]],
                    "corr FD–DVARS antes": [fila["corr_fd_dvars_before"]],
                    "corr FD–DVARS después": [fila["corr_fd_dvars_after"]],
                    "GCOR antes": [fila["gcor_before"]],
                    "GCOR después": [fila["gcor_after"]],
                    "Variabilidad conservada (%)": [
                        fila["temporal_std_retained_pct"]
                    ],
                }
            )
            display(resumen)

            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            axes[0, 0].imshow(
                _estandarizar_filas(antes),
                aspect="auto",
                cmap="gray",
                vmin=-2,
                vmax=2,
                interpolation="nearest",
            )
            axes[0, 0].set_title("Carpet plot — antes de regresar confounds")
            axes[0, 0].set_ylabel("Muestra de vóxeles")

            axes[0, 1].imshow(
                _estandarizar_filas(despues),
                aspect="auto",
                cmap="gray",
                vmin=-2,
                vmax=2,
                interpolation="nearest",
            )
            axes[0, 1].set_title("Carpet plot — después del denoising")

            fs = 1.0 / tr
            frecuencias, potencia_antes = welch(
                antes,
                fs=fs,
                axis=1,
                nperseg=min(128, antes.shape[1]),
            )
            _, potencia_despues = welch(
                despues,
                fs=fs,
                axis=1,
                nperseg=min(128, despues.shape[1]),
            )
            axes[1, 0].semilogy(
                frecuencias,
                potencia_antes.mean(axis=0),
                label="Antes",
            )
            axes[1, 0].semilogy(
                frecuencias,
                potencia_despues.mean(axis=0),
                label="Después",
            )
            axes[1, 0].axvspan(0.008, 0.09, color="tab:green", alpha=0.15)
            axes[1, 0].set_xlabel("Frecuencia (Hz)")
            axes[1, 0].set_ylabel("Potencia media")
            axes[1, 0].set_title("Espectro temporal; verde = banda conservada")
            axes[1, 0].legend(frameon=False)

            dvars_antes_z = _escalar_robusto(dvars_antes)
            dvars_despues_z = _escalar_robusto(dvars_despues)
            axes[1, 1].plot(fd, color="black", linewidth=0.8, label="FD (mm)")
            axes[1, 1].axhline(0.5, color="black", linestyle=":", linewidth=0.8)
            eje_dvars = axes[1, 1].twinx()
            eje_dvars.plot(
                dvars_antes_z,
                color="tab:blue",
                linewidth=0.8,
                alpha=0.8,
                label="DVARS antes",
            )
            eje_dvars.plot(
                dvars_despues_z,
                color="tab:orange",
                linewidth=0.8,
                alpha=0.8,
                label="DVARS después",
            )
            axes[1, 1].set_xlabel("Volumen retenido")
            axes[1, 1].set_ylabel("FD (mm)")
            eje_dvars.set_ylabel("DVARS robustamente estandarizado")
            axes[1, 1].set_title("Movimiento y cambios de señal")
            lineas_1, etiquetas_1 = axes[1, 1].get_legend_handles_labels()
            lineas_2, etiquetas_2 = eje_dvars.get_legend_handles_labels()
            eje_dvars.legend(
                lineas_1 + lineas_2,
                etiquetas_1 + etiquetas_2,
                frameon=False,
                fontsize=8,
            )

            plt.tight_layout()
            plt.show()

    selector.observe(actualizar, names="value")
    display(selector, panel)
    actualizar()


# -----------------------------------------------------------------------------
# Atlas Schaefer y extracción de series regionales
# -----------------------------------------------------------------------------


def _texto_atlas(valor):
    """Convierte etiquetas del atlas a texto uniforme."""

    if isinstance(valor, bytes):
        return valor.decode("utf-8")
    return str(valor)


def _entidades_etiqueta_schaefer(etiqueta):
    """Obtiene hemisferio y red de una etiqueta de Schaefer."""

    partes = etiqueta.split("_")
    hemisferio = partes[1] if len(partes) > 1 else "NA"
    red = partes[2] if len(partes) > 2 else "Unknown"
    return hemisferio, red


def preparar_atlas_schaefer(
    atlas_root,
    template_path,
    n_rois=100,
    yeo_networks=7,
    resolution_mm=2,
):
    """Descarga Schaefer y lo ajusta exactamente a la rejilla MNI usada."""

    atlas_root = Path(atlas_root)
    template_path = Path(template_path)
    atlas_root.mkdir(parents=True, exist_ok=True)
    nombre = f"Schaefer{n_rois}_{yeo_networks}Networks_{resolution_mm}mm"
    maps_path = atlas_root / f"{nombre}_space-projectMNI_dseg.nii.gz"
    labels_path = atlas_root / f"{nombre}_labels.tsv"

    if maps_path.exists() and labels_path.exists():
        return {
            "name": nombre,
            "maps": maps_path,
            "labels_path": labels_path,
            "template_path": template_path,
            "n_rois": int(n_rois),
            "yeo_networks": int(yeo_networks),
            "resolution_mm": int(resolution_mm),
        }

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=int(n_rois),
        yeo_networks=int(yeo_networks),
        resolution_mm=int(resolution_mm),
        data_dir=str(atlas_root / "nilearn_cache"),
        verbose=1,
    )
    atlas_img = nib.load(str(atlas.maps))
    template_img = nib.load(template_path)

    if (
        atlas_img.shape != template_img.shape
        or not np.allclose(atlas_img.affine, template_img.affine, atol=1e-4)
    ):
        atlas_img = image.resample_to_img(
            atlas_img,
            template_img,
            interpolation="nearest",
        )

    datos = np.rint(np.asarray(atlas_img.dataobj)).astype(np.uint16)
    ids = np.unique(datos)
    ids = ids[ids > 0].astype(int)
    etiquetas = [_texto_atlas(valor) for valor in atlas.labels]
    if len(etiquetas) == len(ids) + 1 and etiquetas[0].lower().startswith(
        "background"
    ):
        etiquetas = etiquetas[1:]
    if len(etiquetas) != len(ids):
        etiquetas = [f"ROI_{roi_id:03d}" for roi_id in ids]

    header = template_img.header.copy()
    header.set_data_dtype(np.uint16)
    nib.save(
        nib.Nifti1Image(datos, template_img.affine, header),
        maps_path,
    )

    filas = []
    for roi_id, etiqueta in zip(ids, etiquetas):
        hemisferio, red = _entidades_etiqueta_schaefer(etiqueta)
        filas.append(
            {
                "roi_id": int(roi_id),
                "label": etiqueta,
                "hemisphere": hemisferio,
                "network": red,
                "atlas_voxels": int(np.count_nonzero(datos == roi_id)),
            }
        )
    pd.DataFrame(filas).to_csv(labels_path, sep="\t", index=False)

    return {
        "name": nombre,
        "maps": maps_path,
        "labels_path": labels_path,
        "template_path": template_path,
        "n_rois": int(n_rois),
        "yeo_networks": int(yeo_networks),
        "resolution_mm": int(resolution_mm),
    }


def _matriz_desruida_en_mascara(fila):
    """Abre la salida rápida o extrae la matriz desde una salida NIfTI."""

    ruta = Path(fila.denoised_path)
    mascara = np.asarray(nib.load(fila.normalized_mask_path).dataobj) > 0
    indices = np.flatnonzero(mascara.ravel())

    if ruta.suffix == ".npy":
        series = np.load(ruta, mmap_mode="r")
    elif ruta.name.endswith((".nii", ".nii.gz")):
        _, indices_nifti, series = _extraer_series_mascara(
            ruta,
            fila.normalized_mask_path,
        )
        if not np.array_equal(indices, indices_nifti):
            raise RuntimeError("La máscara del BOLD denoised cambió.")
    else:
        raise ValueError(f"Formato denoised no reconocido: {ruta.name}")

    if series.ndim != 2 or series.shape[0] != len(indices):
        raise RuntimeError(
            "La matriz denoised no coincide con la máscara funcional."
        )
    return series, indices


def extraer_series_roi_lote(
    resultados_denoising,
    atlas_info,
    min_coverage=0.50,
    min_voxels=10,
    overwrite=False,
):
    """Promedia la señal denoised dentro de cada parcela cubierta."""

    if resultados_denoising.empty:
        raise ValueError("No hay resultados de denoising.")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage debe estar entre 0 y 1.")

    atlas_path = Path(atlas_info["maps"])
    etiquetas = pd.read_csv(atlas_info["labels_path"], sep="\t")
    atlas_img = nib.load(atlas_path)
    atlas_data = np.rint(np.asarray(atlas_img.dataobj)).astype(np.int32)
    configuracion = {
        "atlas": atlas_info["name"],
        "min_coverage": float(min_coverage),
        "min_voxels": int(min_voxels),
    }
    resultados = []
    coberturas = []

    for numero, fila in enumerate(
        resultados_denoising.itertuples(index=False),
        start=1,
    ):
        mascara_img = nib.load(fila.normalized_mask_path)
        if (
            mascara_img.shape != atlas_img.shape
            or not np.allclose(mascara_img.affine, atlas_img.affine, atol=1e-4)
        ):
            raise RuntimeError(
                f"El atlas no coincide con la rejilla de sub-{fila.subject}."
            )

        denoised_path = Path(fila.denoised_path)
        if denoised_path.suffix == ".npy":
            sufijo = "_space-MNI152_desc-denoised_timeseries.npy"
        else:
            sufijo = "_space-MNI152_desc-denoised_bold.nii.gz"
        if not denoised_path.name.endswith(sufijo):
            raise ValueError(
                f"No se reconoce la salida denoised: {denoised_path.name}"
            )
        prefijo = denoised_path.name[: -len(sufijo)]
        output_dir = denoised_path.parent
        timeseries_path = output_dir / (
            f"{prefijo}_atlas-Schaefer{atlas_info['n_rois']}_timeseries.tsv"
        )
        coverage_path = output_dir / (
            f"{prefijo}_atlas-Schaefer{atlas_info['n_rois']}_coverage.tsv"
        )
        metadata_path = output_dir / (
            f"{prefijo}_atlas-Schaefer{atlas_info['n_rois']}_timeseries.json"
        )
        productos = (timeseries_path, coverage_path, metadata_path)

        print(
            f"[{numero}/{len(resultados_denoising)}] "
            f"sub-{fila.subject}: series Schaefer"
        )
        inicio = perf_counter()

        if all(ruta.exists() for ruta in productos) and not overwrite:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["configuration"] != configuracion:
                raise RuntimeError(
                    f"Las series ROI existentes de sub-{fila.subject} usan "
                    "otros parámetros. Ejecuta una vez con overwrite=True."
                )
            cobertura = pd.read_csv(coverage_path, sep="\t")
            estado = "reutilizado"
            segundos = 0.0
        else:
            existentes = [ruta for ruta in productos if ruta.exists()]
            if existentes and not overwrite:
                raise RuntimeError(
                    f"Hay productos ROI incompletos para sub-{fila.subject}. "
                    "Usa overwrite=True."
                )

            series, indices_mascara = _matriz_desruida_en_mascara(fila)
            atlas_en_mascara = atlas_data.ravel()[indices_mascara]
            roi_series = {}
            filas_cobertura = []

            for etiqueta in etiquetas.itertuples(index=False):
                roi_id = int(etiqueta.roi_id)
                posiciones = np.flatnonzero(atlas_en_mascara == roi_id)
                disponibles = int(len(posiciones))
                esperados = int(etiqueta.atlas_voxels)
                fraccion = disponibles / esperados if esperados else 0.0
                valida = (
                    fraccion >= min_coverage and disponibles >= int(min_voxels)
                )
                columna = f"roi_{roi_id:03d}"
                if valida:
                    roi_series[columna] = np.asarray(
                        series[posiciones].mean(axis=0),
                        dtype=np.float32,
                    )
                else:
                    roi_series[columna] = np.full(
                        series.shape[1],
                        np.nan,
                        dtype=np.float32,
                    )
                filas_cobertura.append(
                    {
                        "subject": fila.subject,
                        "roi_id": roi_id,
                        "label": etiqueta.label,
                        "hemisphere": etiqueta.hemisphere,
                        "network": etiqueta.network,
                        "atlas_voxels": esperados,
                        "available_voxels": disponibles,
                        "coverage_pct": round(100 * fraccion, 1),
                        "valid": bool(valida),
                    }
                )

            pd.DataFrame(roi_series).to_csv(
                timeseries_path,
                sep="\t",
                index=False,
            )
            cobertura = pd.DataFrame(filas_cobertura)
            cobertura.to_csv(coverage_path, sep="\t", index=False)
            metadata = {
                "algorithm": ALGORITMO_PARCELACION,
                "configuration": configuracion,
                "timepoints": int(series.shape[1]),
                "valid_rois": int(cobertura["valid"].sum()),
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            estado = "procesado"
            segundos = round(perf_counter() - inicio, 1)

        cobertura = cobertura.copy()
        cobertura["subject"] = fila.subject
        coberturas.append(cobertura)
        resultado = fila._asdict()
        resultado.update(
            {
                "atlas_name": atlas_info["name"],
                "roi_timeseries_path": timeseries_path,
                "roi_coverage_path": coverage_path,
                "roi_metadata_path": metadata_path,
                "roi_status": estado,
                "roi_seconds": segundos,
                "valid_rois": int(cobertura["valid"].sum()),
                "mean_roi_coverage_pct": round(
                    float(cobertura["coverage_pct"].mean()),
                    1,
                ),
            }
        )
        resultados.append(resultado)
        print(
            f"    {estado} | regiones válidas: "
            f"{resultado['valid_rois']}/{len(cobertura)}"
        )

    resultados = pd.DataFrame(resultados)
    cobertura_grupo = pd.concat(coberturas, ignore_index=True)
    comunes = cobertura_grupo.groupby("roi_id")["valid"].all()
    ids_comunes = comunes[comunes].index.astype(int).tolist()
    cobertura_grupo["common_valid"] = cobertura_grupo["roi_id"].isin(
        ids_comunes
    )
    resultados["common_valid_rois"] = len(ids_comunes)
    resultados["common_roi_ids"] = [ids_comunes] * len(resultados)
    return resultados, cobertura_grupo


def _mapa_rois_validas(atlas_path, roi_ids):
    """Conserva en el atlas únicamente las regiones válidas seleccionadas."""

    atlas_img = nib.load(atlas_path)
    datos = np.rint(np.asarray(atlas_img.dataobj)).astype(np.int32)
    salida = np.where(np.isin(datos, roi_ids), datos, 0).astype(np.float32)
    header = atlas_img.header.copy()
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(salida, atlas_img.affine, header)


def mostrar_qc_series_roi(
    resultados_roi,
    cobertura_roi,
    atlas_info,
):
    """Muestra cobertura anatómica y series regionales por sujeto."""

    if resultados_roi.empty or cobertura_roi.empty:
        raise ValueError("No hay series ROI para revisar.")

    opciones = []
    for posicion, fila in enumerate(resultados_roi.itertuples()):
        etiqueta = f"sub-{fila.subject}"
        if fila.session:
            etiqueta += f" | ses-{fila.session}"
        opciones.append((etiqueta, posicion))
    selector = widgets.Dropdown(options=opciones, description="Sujeto:")
    panel = widgets.Output()

    def actualizar(change=None):
        fila = resultados_roi.iloc[selector.value]
        metadata = json.loads(
            Path(fila["roi_metadata_path"]).read_text(encoding="utf-8")
        )
        umbral_cobertura = 100 * float(
            metadata["configuration"]["min_coverage"]
        )
        cobertura = cobertura_roi[
            cobertura_roi["subject"].astype(str) == str(fila["subject"])
        ].sort_values("roi_id")
        comunes = cobertura.loc[cobertura["common_valid"], "roi_id"].astype(
            int
        )
        if comunes.empty:
            seleccion = cobertura.loc[cobertura["valid"], "roi_id"].astype(int)
            tipo_seleccion = "válidas para este sujeto"
        else:
            seleccion = comunes
            tipo_seleccion = "válidas en todos los sujetos"

        with panel:
            clear_output(wait=True)
            print(
                f"ROIs válidas del sujeto: {int(cobertura['valid'].sum())}/"
                f"{len(cobertura)} | ROIs comunes: {len(comunes)}"
            )
            print(f"Visor: regiones {tipo_seleccion}")
            display(
                plotting.view_img(
                    _mapa_rois_validas(atlas_info["maps"], seleccion),
                    bg_img=str(fila["normalized_mean_path"]),
                    cmap="gist_ncar",
                    symmetric_cmap=False,
                    threshold=0.5,
                    vmin=0.5,
                    vmax=float(atlas_info["n_rois"]) + 0.5,
                    colorbar=False,
                    black_bg=True,
                    draw_cross=True,
                    opacity=0.55,
                    resampling_interpolation="nearest",
                    width_view=700,
                    title="Cobertura del atlas Schaefer sobre el BOLD medio",
                )
            )

            redes = sorted(cobertura["network"].fillna("Unknown").unique())
            cmap_redes = plt.get_cmap("tab20", max(len(redes), 1))
            colores_red = {
                red: cmap_redes(indice)
                for indice, red in enumerate(redes)
            }
            colores = [
                colores_red[red]
                for red in cobertura["network"].fillna("Unknown")
            ]
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
            axes[0].bar(
                cobertura["roi_id"],
                cobertura["coverage_pct"],
                color=colores,
                width=0.9,
            )
            axes[0].axhline(
                umbral_cobertura,
                color="black",
                linestyle="--",
                linewidth=0.9,
            )
            axes[0].set_ylim(0, 105)
            axes[0].set_xlabel("ROI")
            axes[0].set_ylabel("Cobertura (%)")
            axes[0].set_title("Cobertura funcional de cada parcela")
            leyenda = [
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="none",
                    color=colores_red[red],
                    label=red,
                    markersize=6,
                )
                for red in redes
            ]
            axes[0].legend(
                handles=leyenda,
                ncol=min(6, len(redes)),
                fontsize=7,
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.20),
            )

            series_roi = pd.read_csv(fila["roi_timeseries_path"], sep="\t")
            columnas = [f"roi_{roi_id:03d}" for roi_id in seleccion]
            columnas = [col for col in columnas if col in series_roi.columns]
            datos = series_roi[columnas].to_numpy(dtype=np.float32).T
            datos = datos[np.isfinite(datos).all(axis=1)]
            if datos.size:
                axes[1].imshow(
                    _estandarizar_filas(datos),
                    aspect="auto",
                    cmap="gray",
                    vmin=-2,
                    vmax=2,
                    interpolation="nearest",
                )
                axes[1].set_ylabel("ROI")
                axes[1].set_xlabel("Volumen retenido")
                axes[1].set_title("Series regionales estandarizadas")
            else:
                axes[1].text(
                    0.5,
                    0.5,
                    "No hay ROIs con cobertura suficiente",
                    ha="center",
                    va="center",
                )
                axes[1].axis("off")
            plt.tight_layout()
            plt.show()

    selector.observe(actualizar, names="value")
    display(selector, panel)
    actualizar()


# -----------------------------------------------------------------------------
# Conectividad funcional entre regiones Schaefer
# -----------------------------------------------------------------------------


def calcular_conectividad_lote(
    resultados_roi,
    cobertura_roi,
    atlas_info,
    overwrite=False,
):
    """Calcula una matriz de correlación con las mismas ROIs en el grupo."""

    if resultados_roi.empty or cobertura_roi.empty:
        raise ValueError("No hay series ROI para calcular conectividad.")

    ids_comunes = sorted(
        cobertura_roi.loc[
            cobertura_roi["common_valid"].astype(bool),
            "roi_id",
        ]
        .astype(int)
        .unique()
        .tolist()
    )
    if len(ids_comunes) < 2:
        raise RuntimeError(
            "Se necesitan al menos dos ROIs válidas en todos los sujetos."
        )

    etiquetas = pd.read_csv(atlas_info["labels_path"], sep="\t")
    etiquetas = (
        etiquetas[etiquetas["roi_id"].isin(ids_comunes)]
        .sort_values("roi_id")
        .reset_index(drop=True)
    )
    if etiquetas["roi_id"].astype(int).tolist() != ids_comunes:
        raise RuntimeError("No coinciden las ROIs comunes y sus etiquetas.")

    columnas = [f"roi_{roi_id:03d}" for roi_id in ids_comunes]
    nombres = etiquetas["label"].astype(str).tolist()
    configuracion = {
        "algorithm": ALGORITMO_CONECTIVIDAD,
        "atlas": atlas_info["name"],
        "roi_ids": ids_comunes,
    }
    resultados = []

    for numero, fila in enumerate(
        resultados_roi.itertuples(index=False),
        start=1,
    ):
        series_path = Path(fila.roi_timeseries_path)
        sufijo = (
            f"_atlas-Schaefer{atlas_info['n_rois']}_timeseries.tsv"
        )
        if not series_path.name.endswith(sufijo):
            raise ValueError(
                f"No se reconoce la salida ROI: {series_path.name}"
            )

        prefijo = series_path.name[: -len(sufijo)]
        matriz_path = series_path.parent / (
            f"{prefijo}_atlas-Schaefer{atlas_info['n_rois']}_"
            "desc-correlation_connectivity.tsv"
        )
        metadata_path = series_path.parent / (
            f"{prefijo}_atlas-Schaefer{atlas_info['n_rois']}_"
            "desc-correlation_connectivity.json"
        )

        print(
            f"[{numero}/{len(resultados_roi)}] "
            f"sub-{fila.subject}: conectividad"
        )
        inicio = perf_counter()

        if matriz_path.exists() and metadata_path.exists() and not overwrite:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("configuration") != configuracion:
                raise RuntimeError(
                    f"La conectividad existente de sub-{fila.subject} usa "
                    "otra selección de ROIs. Ejecuta con overwrite=True."
                )
            matriz = pd.read_csv(
                matriz_path,
                sep="\t",
                index_col=0,
            ).to_numpy(dtype=np.float64)
            estado = "reutilizado"
            segundos = 0.0
        else:
            if (matriz_path.exists() or metadata_path.exists()) and not overwrite:
                raise RuntimeError(
                    f"Hay productos de conectividad incompletos para "
                    f"sub-{fila.subject}. Usa overwrite=True."
                )

            series = pd.read_csv(series_path, sep="\t")
            faltantes = [col for col in columnas if col not in series.columns]
            if faltantes:
                raise RuntimeError(
                    f"Faltan {len(faltantes)} series ROI en "
                    f"sub-{fila.subject}."
                )
            datos = series[columnas].to_numpy(dtype=np.float64)
            if datos.shape[0] < 3 or not np.isfinite(datos).all():
                raise RuntimeError(
                    f"Las series comunes de sub-{fila.subject} no son válidas."
                )
            desviacion = datos.std(axis=0)
            if np.any(desviacion <= np.finfo(np.float64).eps):
                raise RuntimeError(
                    f"Hay series ROI constantes en sub-{fila.subject}."
                )

            matriz = np.corrcoef(datos, rowvar=False)
            matriz = np.clip((matriz + matriz.T) / 2.0, -1.0, 1.0)
            np.fill_diagonal(matriz, 1.0)
            pd.DataFrame(
                matriz,
                index=nombres,
                columns=nombres,
            ).to_csv(matriz_path, sep="\t", index=True)

            metadata = {
                "configuration": configuracion,
                "subject": str(fila.subject),
                "timepoints": int(datos.shape[0]),
                "n_rois": int(len(ids_comunes)),
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            estado = "procesado"
            segundos = round(perf_counter() - inicio, 1)

        triangular = matriz[np.triu_indices_from(matriz, k=1)]
        simetria_error = float(np.max(np.abs(matriz - matriz.T)))
        diagonal_error = float(np.max(np.abs(np.diag(matriz) - 1.0)))
        qc_valido = bool(
            np.isfinite(matriz).all()
            and simetria_error < 1e-8
            and diagonal_error < 1e-8
        )
        resultado = fila._asdict()
        resultado.update(
            {
                "connectivity_path": matriz_path,
                "connectivity_metadata_path": metadata_path,
                "connectivity_status": estado,
                "connectivity_seconds": segundos,
                "connectivity_rois": int(len(ids_comunes)),
                "mean_absolute_r": round(
                    float(np.mean(np.abs(triangular))),
                    3,
                ),
                "connectivity_qc_valid": qc_valido,
            }
        )
        resultados.append(resultado)
        print(
            f"    {estado} | {len(ids_comunes)} ROIs | "
            f"QC: {'OK' if qc_valido else 'REVISAR'}"
        )

    return pd.DataFrame(resultados)


def mostrar_qc_conectividad(resultados_conectividad, atlas_info):
    """Muestra la matriz parcelaria, el resumen por red y su distribución."""

    if resultados_conectividad.empty:
        raise ValueError("No hay matrices de conectividad para revisar.")

    etiquetas_atlas = pd.read_csv(atlas_info["labels_path"], sep="\t")
    opciones = []
    for posicion, fila in enumerate(
        resultados_conectividad.itertuples(index=False)
    ):
        etiqueta = f"sub-{fila.subject}"
        if fila.session:
            etiqueta += f" | ses-{fila.session}"
        opciones.append((etiqueta, posicion))

    selector = widgets.Dropdown(options=opciones, description="Sujeto:")
    panel = widgets.Output()

    def actualizar(change=None):
        fila = resultados_conectividad.iloc[selector.value]
        matriz_df = pd.read_csv(
            fila["connectivity_path"],
            sep="\t",
            index_col=0,
        )
        matriz = matriz_df.to_numpy(dtype=np.float64)
        metadata = json.loads(
            Path(fila["connectivity_metadata_path"]).read_text(
                encoding="utf-8"
            )
        )
        roi_ids = metadata["configuration"]["roi_ids"]
        etiquetas = (
            etiquetas_atlas[etiquetas_atlas["roi_id"].isin(roi_ids)]
            .sort_values("roi_id")
            .reset_index(drop=True)
        )
        redes = etiquetas["network"].fillna("Unknown").astype(str)
        redes_unicas = list(dict.fromkeys(redes.tolist()))
        matriz_redes = np.empty(
            (len(redes_unicas), len(redes_unicas)),
            dtype=np.float64,
        )
        for i, red_i in enumerate(redes_unicas):
            indices_i = np.flatnonzero(redes.to_numpy() == red_i)
            for j, red_j in enumerate(redes_unicas):
                indices_j = np.flatnonzero(redes.to_numpy() == red_j)
                bloque = matriz[np.ix_(indices_i, indices_j)]
                if i == j:
                    bloque = bloque[
                        ~np.eye(len(indices_i), dtype=bool)
                    ]
                matriz_redes[i, j] = (
                    float(np.mean(bloque)) if bloque.size else np.nan
                )

        triangular = matriz[np.triu_indices_from(matriz, k=1)]
        cambios = np.flatnonzero(redes.to_numpy()[1:] != redes.to_numpy()[:-1])
        limites = cambios + 0.5

        with panel:
            clear_output(wait=True)
            print(
                f"ROIs comunes: {matriz.shape[0]} | "
                f"media |r|: {np.mean(np.abs(triangular)):.3f} | "
                f"QC estructural: "
                f"{'OK' if fila['connectivity_qc_valid'] else 'REVISAR'}"
            )
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            imagen = axes[0].imshow(
                matriz,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                interpolation="nearest",
            )
            for limite in limites:
                axes[0].axhline(limite, color="black", linewidth=0.25)
                axes[0].axvline(limite, color="black", linewidth=0.25)
            axes[0].set_title("Conectividad entre ROIs")
            axes[0].set_xlabel("ROI")
            axes[0].set_ylabel("ROI")
            fig.colorbar(imagen, ax=axes[0], fraction=0.046, label="r")

            imagen_red = axes[1].imshow(
                matriz_redes,
                cmap="coolwarm",
                vmin=-0.5,
                vmax=0.5,
                interpolation="nearest",
            )
            axes[1].set_xticks(range(len(redes_unicas)))
            axes[1].set_yticks(range(len(redes_unicas)))
            axes[1].set_xticklabels(redes_unicas, rotation=90, fontsize=7)
            axes[1].set_yticklabels(redes_unicas, fontsize=7)
            axes[1].set_title("Correlación media por red")
            fig.colorbar(imagen_red, ax=axes[1], fraction=0.046, label="r")

            axes[2].hist(
                triangular,
                bins=40,
                color="#4C78A8",
                alpha=0.85,
            )
            axes[2].axvline(0, color="black", linestyle="--", linewidth=0.8)
            axes[2].set_xlim(-1, 1)
            axes[2].set_xlabel("Correlación r")
            axes[2].set_ylabel("Conexiones")
            axes[2].set_title("Distribución de conexiones")

            plt.tight_layout()
            plt.show()

    selector.observe(actualizar, names="value")
    display(selector, panel)
    actualizar()


def _prefijo_bold(ruta):
    """Retira el sufijo BOLD de un nombre NIfTI."""

    nombre = Path(ruta).name

    for sufijo in ("_bold.nii.gz", "_bold.nii"):
        if nombre.endswith(sufijo):
            return nombre[: -len(sufijo)]

    raise ValueError(f"No se reconoce el nombre BOLD: {nombre}")


def _directorio_funcional(output_root, sujeto, sesion=None):
    """Construye la carpeta funcional de derivados."""

    output_dir = Path(output_root) / f"sub-{sujeto}"

    if sesion:
        output_dir = output_dir / f"ses-{sesion}"

    output_dir = output_dir / "func"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def descartar_volumenes_lote(
    inventario,
    output_root,
    n_descartar=5,
    overwrite=False,
):
    """Descarta los primeros volúmenes de todos los BOLD del inventario."""

    if inventario.empty:
        raise ValueError("No hay BOLD para procesar.")

    n_descartar = int(n_descartar)

    if n_descartar < 0:
        raise ValueError("n_descartar no puede ser negativo.")

    resultados = []

    for numero, fila in enumerate(inventario.itertuples(), start=1):
        bold_path = Path(fila.ruta)
        output_dir = _directorio_funcional(
            output_root,
            fila.sujeto,
            fila.sesion,
        )
        prefijo = _prefijo_bold(bold_path)
        output_path = output_dir / f"{prefijo}_desc-trim_bold.nii.gz"

        bold = nib.load(bold_path)
        volumenes_originales = int(bold.shape[3])
        volumenes_finales = volumenes_originales - n_descartar

        if volumenes_finales < 20:
            raise ValueError(
                f"{bold_path.name}: quedarían solo {volumenes_finales} "
                "volúmenes después del descarte."
            )

        if output_path.exists() and not overwrite:
            salida = nib.load(output_path)

            if salida.shape[3] != volumenes_finales:
                raise RuntimeError(
                    f"La salida existente no coincide con n_descartar="
                    f"{n_descartar}: {output_path}"
                )

            estado = "reutilizado"
        else:
            salida = image.index_img(
                bold,
                slice(n_descartar, volumenes_originales),
            )
            nib.save(salida, output_path)
            estado = "procesado"

        resultado = {
            "subject": fila.sujeto,
            "session": fila.sesion,
            "task": fila.tarea,
            "run": fila.corrida,
            "input_path": bold_path,
            "trimmed_path": output_path,
            "TR_s": float(fila.TR_s),
            "discarded_volumes": n_descartar,
            "original_volumes": volumenes_originales,
            "remaining_volumes": volumenes_finales,
            "duration_min": round(
                volumenes_finales * float(fila.TR_s) / 60,
                2,
            ),
            "status": estado,
        }
        resultados.append(resultado)

        print(
            f"[{numero}/{len(inventario)}] sub-{fila.sujeto}: "
            f"{volumenes_originales} → {volumenes_finales} volúmenes "
            f"({estado})"
        )

    return pd.DataFrame(resultados)


def mostrar_qc_descarte(resultados):
    """Muestra el BOLD medio después del descarte y el conteo de volúmenes."""

    if resultados.empty:
        raise ValueError("No hay resultados del descarte para visualizar.")

    opciones = []
    for posicion, fila in enumerate(resultados.itertuples()):
        etiqueta = f"sub-{fila.subject}"
        if fila.session:
            etiqueta += f" | ses-{fila.session}"
        opciones.append((etiqueta, posicion))

    selector = widgets.Dropdown(options=opciones, description="BOLD:")
    panel = widgets.Output()

    def actualizar(change=None):
        fila = resultados.iloc[selector.value]
        bold_medio = image.mean_img(str(fila["trimmed_path"]))

        with panel:
            clear_output(wait=True)
            print(f"Salida: {Path(fila['trimmed_path']).name}")
            print(
                f"Volúmenes: {fila['original_volumes']} → "
                f"{fila['remaining_volumes']} | "
                f"Descartados: {fila['discarded_volumes']} | "
                f"Duración útil: {fila['duration_min']} min"
            )
            display(
                plotting.view_img(
                    bold_medio,
                    bg_img=False,
                    cmap="gray",
                    symmetric_cmap=False,
                    threshold=None,
                    colorbar=False,
                    black_bg=True,
                    draw_cross=True,
                    width_view=700,
                    title="QC del descarte: BOLD medio",
                )
            )

    selector.observe(actualizar, names="value")
    display(selector, panel)
    actualizar()


def _parametros_desde_transformaciones(transformaciones):
    """Convierte transformaciones rígidas de ANTs en seis series numéricas."""

    filas = []

    for transformacion in transformaciones:
        if transformacion == "NA":
            filas.append([np.nan] * 6)
            continue

        ruta = transformacion
        if isinstance(transformacion, (list, tuple)):
            ruta = transformacion[0]

        transform = ants.read_transform(str(ruta))
        parametros = np.asarray(transform.parameters, dtype=float)

        if parametros.size < 12:
            raise RuntimeError(
                "ANTs devolvió una transformación que no es rígida 3D."
            )

        matriz = parametros[:9].reshape(3, 3)
        traslacion = parametros[9:12]
        rotacion = Rotation.from_matrix(matriz).as_euler("xyz", degrees=False)
        filas.append([*traslacion, *rotacion])

    columnas = [
        "trans_x_mm",
        "trans_y_mm",
        "trans_z_mm",
        "rot_x_rad",
        "rot_y_rad",
        "rot_z_rad",
    ]
    tabla = pd.DataFrame(filas, columns=columnas)
    return tabla.interpolate(limit_direction="both").fillna(0.0)


def _matriz_rotacion(rx, ry, rz):
    """Construye una matriz rígida desde rotaciones en radianes."""

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    return (
        np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    )


def _fd_caja_envolvente(parametros):
    """Calcula FD usando los seis puntos de una caja cerebral."""

    posiciones = np.stack(
        [
            PUNTOS_CONTROL @ _matriz_rotacion(*fila[3:]).T + fila[:3]
            for fila in parametros
        ]
    )
    desplazamientos = np.linalg.norm(
        np.diff(posiciones, axis=0),
        axis=2,
    )
    return np.concatenate([[0.0], desplazamientos.max(axis=1)])


def _fd_power(parametros, radio_mm=50.0):
    """Calcula FD de Power a partir de derivadas de movimiento."""

    diferencias = np.vstack(
        [np.zeros((1, 6)), np.abs(np.diff(parametros, axis=0))]
    )
    diferencias[:, 3:] *= float(radio_mm)
    return diferencias.sum(axis=1)


def _escalar_robusto(serie):
    """Escala una serie mediante mediana y rango intercuartílico."""

    serie = np.asarray(serie, dtype=float)
    iqr = float(np.subtract(*np.percentile(serie, [75, 25])))

    if iqr <= np.finfo(float).eps:
        return np.zeros_like(serie)

    return (serie - np.median(serie)) / (0.74 * iqr)


def _series_senal(corrected_path):
    """Calcula señal global y DVARS cargando el NIfTI comprimido una sola vez."""

    bold = nib.load(corrected_path)
    bold_medio = image.mean_img(bold)
    mask_img = masking.compute_epi_mask(bold_medio)
    mask_array = np.asarray(mask_img.dataobj) > 0

    if not mask_array.any():
        raise RuntimeError(f"No se pudo calcular la máscara EPI: {corrected_path}")

    # Leer un volumen por vez desde .nii.gz obliga a descomprimir el archivo
    # repetidamente. Esta carga única usa más memoria, pero reduce mucho el tiempo.
    data = np.asarray(bold.dataobj, dtype=np.float32)
    volumenes = int(data.shape[3])
    senal_global = np.zeros(volumenes, dtype=float)
    dvars = np.zeros(volumenes, dtype=float)
    anterior = None

    for indice in range(volumenes):
        volumen = data[..., indice][mask_array]
        volumen = np.nan_to_num(volumen, copy=False)
        senal_global[indice] = float(volumen.mean())

        if anterior is not None:
            dvars[indice] = float(
                np.sqrt(np.mean(np.square(volumen - anterior)))
            )

        anterior = volumen

    referencia_dvars = float(np.median(dvars[1:]))
    if referencia_dvars > np.finfo(float).eps:
        dvars_normalizado = dvars / referencia_dvars
    else:
        dvars_normalizado = np.zeros_like(dvars)

    gschange = np.concatenate(
        [[0.0], np.abs(np.diff(senal_global))]
    )
    gschange_std = _escalar_robusto(gschange)

    return senal_global, dvars, dvars_normalizado, gschange, gschange_std


def _ruta_series_qc(corrected_path, incluir_senal):
    """Construye la ruta de las métricas QC por volumen."""

    corrected_path = Path(corrected_path)
    sufijo = "_desc-moco_bold.nii.gz"

    if not corrected_path.name.endswith(sufijo):
        raise ValueError(
            f"No se reconoce la salida de movimiento: {corrected_path.name}"
        )

    prefijo = corrected_path.name[: -len(sufijo)]
    modo = "qcFull" if incluir_senal else "qcMotion"
    return corrected_path.parent / f"{prefijo}_desc-{modo}_timeseries.tsv"


def _cargar_series_qc(fila, incluir_senal=False):
    """Calcula o reutiliza métricas rápidas o completas de QC."""

    corrected_path = Path(fila["corrected_path"])
    motion_path = Path(fila["motion_path"])
    fd_path = Path(fila["fd_path"])
    qc_path = _ruta_series_qc(corrected_path, incluir_senal)
    entradas = (motion_path, fd_path)
    if incluir_senal:
        entradas = (corrected_path, motion_path, fd_path)

    actualizado = (
        qc_path.exists()
        and qc_path.stat().st_mtime
        >= max(ruta.stat().st_mtime for ruta in entradas)
    )

    if actualizado:
        return pd.read_csv(qc_path, sep="\t"), qc_path

    series = _cargar_metricas_movimiento(motion_path, fd_path)

    if incluir_senal:
        (
            senal_global,
            dvars,
            dvars_normalizado,
            gschange,
            gschange_std,
        ) = _series_senal(corrected_path)

        if len(senal_global) != len(series):
            raise RuntimeError(
                "Las longitudes del BOLD y las métricas no coinciden."
            )

        series["global_signal"] = senal_global
        series["dvars"] = dvars
        series["dvars_normalizado"] = dvars_normalizado
        series["gschange"] = gschange
        series["gschange_std"] = gschange_std
    series.to_csv(qc_path, sep="\t", index=False)
    return series, qc_path


def _cargar_metricas_movimiento(motion_path, fd_path):
    """Carga parámetros rígidos y calcula las tres variantes de FD."""

    movimiento = pd.read_csv(motion_path, sep="\t")
    columnas = [
        "trans_x_mm",
        "trans_y_mm",
        "trans_z_mm",
        "rot_x_rad",
        "rot_y_rad",
        "rot_z_rad",
    ]
    parametros = movimiento[columnas].fillna(0.0).to_numpy(dtype=float)
    fd_ants = pd.read_csv(fd_path, sep="\t")[
        "framewise_displacement_mm"
    ].fillna(0.0).to_numpy(dtype=float)
    longitudes = {
        len(parametros),
        len(fd_ants),
    }
    if len(longitudes) != 1:
        raise RuntimeError(
            "Las longitudes del BOLD, movimiento, FD y señal no coinciden."
        )

    series = movimiento.copy()
    series["fd_ants_mm"] = fd_ants
    series["fd_caja_mm"] = _fd_caja_envolvente(parametros)
    series["fd_power_mm"] = _fd_power(parametros)

    return series


def _resumen_series_qc(
    fila,
    series_antes,
    series_despues,
    umbral_fd,
    umbral_gschange,
    umbral_dvars,
    incluir_senal,
):
    """Resume las métricas del notebook largo para un BOLD."""

    fd_antes = series_antes["fd_caja_mm"].to_numpy()
    fd_despues = series_despues["fd_caja_mm"].to_numpy()
    atipicos_mov_antes = fd_antes > umbral_fd
    atipicos_mov_despues = fd_despues > umbral_fd
    atipicos = atipicos_mov_antes.copy()

    if incluir_senal:
        atipicos |= (
            series_antes["gschange_std"].to_numpy() > umbral_gschange
        )

    validos = ~atipicos
    atipicos_power_antes = (
        series_antes["fd_power_mm"].to_numpy() > umbral_fd
    )
    atipicos_power_despues = (
        series_despues["fd_power_mm"].to_numpy() > umbral_fd
    )

    if incluir_senal:
        atipicos_power_antes |= (
            series_antes["dvars_normalizado"].to_numpy() > umbral_dvars
        )

    def media_validos(nombre):
        if not validos.any():
            return np.nan
        return round(float(series_antes.loc[validos, nombre].mean()), 3)

    resumen = {
        "Sujeto": f"sub-{fila['subject']}",
        "Sesión": fila["session"],
        "Tarea": fila["task"],
        "Corrida": fila["run"],
        "Volúmenes": len(series_antes),
        "MaxMotion_antes_mm": round(
            float(fd_antes.max()),
            3,
        ),
        "MaxMotion_después_mm": round(float(fd_despues.max()), 3),
        "MeanMotion_antes_mm": media_validos("fd_caja_mm"),
        "MeanMotion_después_mm": round(float(fd_despues.mean()), 3),
        "InvalidScans_antes": int(atipicos_mov_antes.sum()),
        "InvalidScans_después": int(atipicos_mov_despues.sum()),
        "PVS_antes": round(float((~atipicos_mov_antes).mean()), 3),
        "PVS_después": round(float((~atipicos_mov_despues).mean()), 3),
        "InvalidScans_Power_antes": int(atipicos_power_antes.sum()),
        "InvalidScans_Power_después": int(atipicos_power_despues.sum()),
        "FD_ANTs_antes_medio": round(
            float(series_antes["fd_ants_mm"].mean()),
            3,
        ),
        "FD_ANTs_después_medio": round(
            float(series_despues["fd_ants_mm"].mean()),
            3,
        ),
        "FD_Power_antes_medio": round(
            float(series_antes["fd_power_mm"].mean()),
            3,
        ),
        "FD_Power_después_medio": round(
            float(series_despues["fd_power_mm"].mean()),
            3,
        ),
        "FD_caja_antes_medio": round(float(fd_antes.mean()), 3),
        "FD_caja_después_medio": round(float(fd_despues.mean()), 3),
    }

    if incluir_senal:
        fd = fd_antes
        dvars = series_antes["dvars"].to_numpy()
        correlacion = np.nan

        if len(fd) > 2 and np.std(fd[1:]) > 0 and np.std(dvars[1:]) > 0:
            correlacion = round(
                float(np.corrcoef(fd[1:], dvars[1:])[0, 1]),
                3,
            )

        resumen["MeanGSchange"] = media_validos("gschange_std")
        resumen["InvalidScans_Power_DVARS"] = int(
            atipicos_power_antes.sum()
        )
        resumen["corr_FD_DVARS"] = correlacion

    return resumen


def corregir_movimiento_lote(
    resultados_descarte,
    umbral_fd=0.5,
    overwrite=False,
    verbose=True,
):
    """Corrige movimiento rígido y guarda parámetros y FD por volumen."""

    if resultados_descarte.empty:
        raise ValueError("No hay BOLD recortados para corregir.")

    resultados = []

    for numero, fila in enumerate(
        resultados_descarte.itertuples(),
        start=1,
    ):
        trimmed_path = Path(fila.trimmed_path)
        output_dir = trimmed_path.parent
        prefijo = trimmed_path.name.removesuffix("_desc-trim_bold.nii.gz")
        corrected_path = output_dir / f"{prefijo}_desc-moco_bold.nii.gz"
        motion_path = output_dir / f"{prefijo}_desc-motion_timeseries.tsv"
        fd_path = output_dir / f"{prefijo}_desc-fd_timeseries.tsv"
        transform_dir = output_dir / f"{prefijo}_motion_transforms"
        productos = (corrected_path, motion_path, fd_path)

        print(
            f"[{numero}/{len(resultados_descarte)}] "
            f"sub-{fila.subject}: corrección de movimiento"
        )

        if all(ruta.exists() for ruta in productos) and not overwrite:
            parametros = pd.read_csv(motion_path, sep="\t")
            fd = pd.read_csv(fd_path, sep="\t")["framewise_displacement_mm"]
            estado = "reutilizado"
            segundos = 0.0
        else:
            existentes = [ruta for ruta in productos if ruta.exists()]
            if existentes and not overwrite:
                raise RuntimeError(
                    f"Hay productos de movimiento incompletos para "
                    f"sub-{fila.subject}. Usa overwrite=True."
                )

            inicio = perf_counter()
            transform_dir.mkdir(parents=True, exist_ok=True)
            bold = ants.image_read(str(trimmed_path))

            if bold.dimension != 4:
                raise ValueError(f"El BOLD no es 4D: {trimmed_path}")

            referencia = ants.get_average_of_timeseries(bold)
            correccion = ants.motion_correction(
                bold,
                fixed=referencia,
                type_of_transform="BOLDRigid",
                fdOffset=50,
                outprefix=str(transform_dir / "vol"),
                verbose=verbose,
            )

            corregido = correccion["motion_corrected"]
            fd = pd.Series(
                np.asarray(correccion["FD"], dtype=float).reshape(-1),
                name="framewise_displacement_mm",
            )
            parametros = _parametros_desde_transformaciones(
                correccion["motion_parameters"]
            )

            if len(fd) != bold.shape[3] or len(parametros) != bold.shape[3]:
                raise RuntimeError(
                    "Las longitudes de BOLD, parámetros y FD no coinciden."
                )

            ants.image_write(corregido, str(corrected_path))
            parametros.to_csv(motion_path, sep="\t", index=False)
            fd.to_frame().to_csv(fd_path, sep="\t", index=False)
            estado = "procesado"
            segundos = round(perf_counter() - inicio, 1)

        fd_array = np.asarray(fd, dtype=float)
        resultado = {
            "subject": fila.subject,
            "session": fila.session,
            "task": fila.task,
            "run": fila.run,
            "input_path": trimmed_path,
            "corrected_path": corrected_path,
            "motion_path": motion_path,
            "fd_path": fd_path,
            "TR_s": float(fila.TR_s),
            "volumes": int(fila.remaining_volumes),
            "status": estado,
            "seconds": segundos,
            "fd_mean_mm": round(float(np.mean(fd_array)), 3),
            "fd_max_mm": round(float(np.max(fd_array)), 3),
            "fd_above_threshold": int(np.sum(fd_array > umbral_fd)),
            "fd_above_percent": round(
                100 * float(np.mean(fd_array > umbral_fd)),
                1,
            ),
        }
        resultados.append(resultado)

        print(
            f"    FD medio: {resultado['fd_mean_mm']} mm | "
            f"FD máximo: {resultado['fd_max_mm']} mm | "
            f"> {umbral_fd} mm: {resultado['fd_above_threshold']} "
            f"({resultado['fd_above_percent']}%)"
        )

    return pd.DataFrame(resultados)


def estimar_movimiento_residual_lote(
    resultados_movimiento,
    overwrite=False,
    verbose=False,
):
    """Reestima movimiento sobre el BOLD corregido para medir el residual."""

    if resultados_movimiento.empty:
        raise ValueError("No hay BOLD corregidos para evaluar.")

    resultados = []

    for numero, fila in enumerate(
        resultados_movimiento.itertuples(index=False),
        start=1,
    ):
        corrected_path = Path(fila.corrected_path)
        sufijo = "_desc-moco_bold.nii.gz"

        if not corrected_path.name.endswith(sufijo):
            raise ValueError(
                f"No se reconoce el BOLD corregido: {corrected_path.name}"
            )

        prefijo = corrected_path.name[: -len(sufijo)]
        output_dir = corrected_path.parent
        residual_motion_path = (
            output_dir / f"{prefijo}_desc-residualMotion_timeseries.tsv"
        )
        residual_fd_path = (
            output_dir / f"{prefijo}_desc-residualFD_timeseries.tsv"
        )
        transform_dir = output_dir / f"{prefijo}_residual_motion_transforms"
        productos = (residual_motion_path, residual_fd_path)

        print(
            f"[{numero}/{len(resultados_movimiento)}] "
            f"sub-{fila.subject}: movimiento residual"
        )

        if all(ruta.exists() for ruta in productos) and not overwrite:
            parametros = pd.read_csv(residual_motion_path, sep="\t")
            fd = pd.read_csv(residual_fd_path, sep="\t")[
                "framewise_displacement_mm"
            ]
            estado = "reutilizado"
            segundos = 0.0
        else:
            existentes = [ruta for ruta in productos if ruta.exists()]
            if existentes and not overwrite:
                raise RuntimeError(
                    f"Hay métricas residuales incompletas para "
                    f"sub-{fila.subject}. Usa overwrite=True."
                )

            inicio = perf_counter()
            transform_dir.mkdir(parents=True, exist_ok=True)
            bold_corregido = ants.image_read(str(corrected_path))

            if bold_corregido.dimension != 4:
                raise ValueError(
                    f"El BOLD corregido no es 4D: {corrected_path}"
                )

            referencia = ants.get_average_of_timeseries(bold_corregido)
            estimacion = ants.motion_correction(
                bold_corregido,
                fixed=referencia,
                type_of_transform="BOLDRigid",
                fdOffset=50,
                outprefix=str(transform_dir / "vol"),
                verbose=verbose,
            )

            fd = pd.Series(
                np.asarray(estimacion["FD"], dtype=float).reshape(-1),
                name="framewise_displacement_mm",
            )
            parametros = _parametros_desde_transformaciones(
                estimacion["motion_parameters"]
            )

            if (
                len(fd) != bold_corregido.shape[3]
                or len(parametros) != bold_corregido.shape[3]
            ):
                raise RuntimeError(
                    "Las longitudes del BOLD corregido y las métricas "
                    "residuales no coinciden."
                )

            parametros.to_csv(residual_motion_path, sep="\t", index=False)
            fd.to_frame().to_csv(residual_fd_path, sep="\t", index=False)
            estado = "procesado"
            segundos = round(perf_counter() - inicio, 1)

        resultado = fila._asdict()
        resultado.update(
            {
                "residual_motion_path": residual_motion_path,
                "residual_fd_path": residual_fd_path,
                "residual_status": estado,
                "residual_seconds": segundos,
            }
        )
        resultados.append(resultado)

        metricas = _cargar_metricas_movimiento(
            residual_motion_path,
            residual_fd_path,
        )
        print(
            f"    FD residual medio: "
            f"{metricas['fd_caja_mm'].mean():.3f} mm | "
            f"máximo: {metricas['fd_caja_mm'].max():.3f} mm"
        )

    return pd.DataFrame(resultados)


def mostrar_qc_movimiento(
    resultados,
    umbral_fd=0.5,
    umbral_gschange=3.0,
    umbral_dvars=1.5,
    incluir_senal=False,
):
    """Reproduce el QC completo de movimiento y señal del notebook largo."""

    if resultados.empty:
        raise ValueError("No hay correcciones de movimiento para visualizar.")

    columnas_residuales = {
        "residual_motion_path",
        "residual_fd_path",
    }
    if not columnas_residuales.issubset(resultados.columns):
        raise ValueError(
            "Falta estimar el movimiento residual. Ejecuta primero "
            "estimar_movimiento_residual_lote(resultados_movimiento)."
        )

    series_qc = {}
    resumen = []
    opciones = []

    modo = "completo" if incluir_senal else "rápido"
    print(f"Calculando o reutilizando QC {modo} por volumen...")

    for posicion, (_, fila) in enumerate(resultados.iterrows()):
        series_antes, qc_path = _cargar_series_qc(
            fila,
            incluir_senal=incluir_senal,
        )
        series_despues = _cargar_metricas_movimiento(
            fila["residual_motion_path"],
            fila["residual_fd_path"],
        )

        if len(series_antes) != len(series_despues):
            raise RuntimeError(
                "Las series de movimiento antes y después no coinciden."
            )

        series_qc[posicion] = {
            "antes": series_antes,
            "después": series_despues,
        }
        resumen.append(
            _resumen_series_qc(
                fila,
                series_antes,
                series_despues,
                umbral_fd,
                umbral_gschange,
                umbral_dvars,
                incluir_senal,
            )
        )

        etiqueta = f"sub-{fila['subject']}"
        if fila["session"]:
            etiqueta += f" | ses-{fila['session']}"
        if fila["task"]:
            etiqueta += f" | task-{fila['task']}"
        if fila["run"]:
            etiqueta += f" | run-{fila['run']}"
        opciones.append((etiqueta, posicion))

    resumen = pd.DataFrame(resumen)
    print("\nResumen de movimiento y estabilidad de señal:")
    display(resumen)
    if incluir_senal:
        print(
            f"Umbrales: FD caja > {umbral_fd} mm o "
            f"GSchange > {umbral_gschange}; comparación Power/DVARS > "
            f"{umbral_dvars}."
        )
    else:
        print(
            f"Umbral rápido: FD caja > {umbral_fd} mm. "
            "DVARS y GSchange quedan omitidos."
        )
    print("PVS < 0.75 requiere revisión especial.")
    print("Después = movimiento residual reestimado en el BOLD corregido.")

    longitud = min(
        len(conjunto["antes"])
        for conjunto in series_qc.values()
    )
    matriz_fd_antes = np.vstack(
        [
            series_qc[posicion]["antes"]["fd_caja_mm"]
            .to_numpy()[:longitud]
            for posicion in range(len(resultados))
        ]
    )
    matriz_fd_despues = np.vstack(
        [
            series_qc[posicion]["después"]["fd_caja_mm"]
            .to_numpy()[:longitud]
            for posicion in range(len(resultados))
        ]
    )
    limite_color = float(
        np.percentile(
            np.concatenate(
                [matriz_fd_antes.ravel(), matriz_fd_despues.ravel()]
            ),
            99,
        )
    )
    if limite_color <= 0:
        limite_color = 1.0

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 2.2 + 0.45 * len(resultados)),
        sharex=True,
        sharey=True,
    )
    mapas = []
    for ax, matriz, titulo in zip(
        axes,
        (matriz_fd_antes, matriz_fd_despues),
        ("FD antes", "FD residual después"),
    ):
        mapas.append(
            ax.imshow(
                matriz,
                aspect="auto",
                cmap="magma",
                vmin=0,
                vmax=limite_color,
            )
        )
        ax.set_yticks(range(len(opciones)))
        ax.set_yticklabels([etiqueta for etiqueta, _ in opciones])
        ax.set_xlabel("Volumen")
        ax.set_title(titulo)
    fig.colorbar(
        mapas[-1],
        ax=axes,
        fraction=0.025,
        label="FD caja (mm)",
    )
    plt.tight_layout()
    plt.show()

    selector = widgets.Dropdown(options=opciones, description="BOLD:")
    panel = widgets.Output()

    def actualizar(change=None):
        fila = resultados.iloc[selector.value]
        series_antes = series_qc[selector.value]["antes"]
        series_despues = series_qc[selector.value]["después"]
        fd = series_antes["fd_caja_mm"].to_numpy()
        fd_despues = series_despues["fd_caja_mm"].to_numpy()
        atipicos = fd > umbral_fd
        atipicos_despues = fd_despues > umbral_fd

        if incluir_senal:
            dvars = series_antes["dvars_normalizado"].to_numpy()
            gschange = series_antes["gschange_std"].to_numpy()
            atipicos |= gschange > umbral_gschange
        rotaciones_grados = np.rad2deg(
            series_antes[
                ["rot_x_rad", "rot_y_rad", "rot_z_rad"]
            ].to_numpy()
        )
        traslaciones = series_antes[
            ["trans_x_mm", "trans_y_mm", "trans_z_mm"]
        ].to_numpy()
        traslaciones_despues = series_despues[
            ["trans_x_mm", "trans_y_mm", "trans_z_mm"]
        ].to_numpy()
        rotaciones_despues = np.rad2deg(
            series_despues[
                ["rot_x_rad", "rot_y_rad", "rot_z_rad"]
            ].to_numpy()
        )
        volumenes = np.arange(len(fd))

        with panel:
            clear_output(wait=True)
            print(
                f"Antes → MaxMotion: {fd.max():.3f} mm | "
                f"atípicos: {atipicos.sum()}/{len(fd)} | "
                f"PVS: {(~atipicos).mean():.3f}"
            )
            print(
                f"Después → MaxMotion: {fd_despues.max():.3f} mm | "
                f"atípicos: {atipicos_despues.sum()}/{len(fd_despues)} | "
                f"PVS: {(~atipicos_despues).mean():.3f}"
            )

            fig, ax_fd = plt.subplots(figsize=(11, 4))
            ax_fd.plot(
                volumenes,
                fd,
                color="tab:blue",
                linewidth=0.9,
                label="Antes: FD estimado",
            )
            ax_fd.plot(
                volumenes,
                fd_despues,
                color="tab:orange",
                linewidth=1.2,
                linestyle="--",
                label="Después: FD residual",
            )
            ax_fd.axhline(
                umbral_fd,
                color="tab:blue",
                linestyle=":",
            )
            ax_fd.set_ylabel("FD caja (mm)", color="tab:blue")
            ax_fd.tick_params(axis="y", labelcolor="tab:blue")

            for indice in np.flatnonzero(atipicos):
                ax_fd.axvspan(
                    indice - 0.5,
                    indice + 0.5,
                    color="grey",
                    alpha=0.25,
                    linewidth=0,
                )

            if incluir_senal:
                ax_senal = ax_fd.twinx()
                ax_senal.plot(
                    volumenes,
                    dvars,
                    color="tab:red",
                    linewidth=0.8,
                    alpha=0.8,
                    label="DVARS normalizado",
                )
                ax_senal.plot(
                    volumenes,
                    gschange,
                    color="tab:green",
                    linewidth=0.8,
                    alpha=0.8,
                    label="GSchange",
                )
                ax_senal.axhline(
                    umbral_gschange,
                    color="tab:green",
                    linestyle=":",
                )
                ax_senal.set_ylabel("Unidades estandarizadas")
                lineas = ax_fd.get_lines()[:2] + ax_senal.get_lines()[:2]
                ax_fd.legend(
                    lineas,
                    [linea.get_label() for linea in lineas],
                    ncol=2,
                    frameon=False,
                    loc="upper right",
                )
            else:
                ax_fd.legend(frameon=False, loc="upper right")
            ax_fd.set_xlabel("Volumen")
            titulo = "FD antes y después de aplicar la corrección"
            if incluir_senal:
                titulo = (
                    "FD antes/después y cambios de señal; "
                    "gris = volumen atípico"
                )
            ax_fd.set_title(titulo)
            plt.tight_layout()
            plt.show()

            fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
            axes[0, 0].plot(
                volumenes,
                traslaciones,
            )
            axes[0, 0].set_title("Traslaciones — antes")
            axes[0, 0].set_ylabel("mm")
            axes[0, 0].legend(
                ["X", "Y", "Z"],
                ncol=3,
                frameon=False,
            )

            axes[0, 1].plot(volumenes, traslaciones_despues)
            axes[0, 1].set_title("Traslaciones — después")
            axes[0, 1].set_ylabel("mm")
            axes[0, 1].set_ylim(axes[0, 0].get_ylim())
            axes[0, 1].legend(
                ["X", "Y", "Z"],
                ncol=3,
                frameon=False,
            )

            axes[1, 0].plot(volumenes, rotaciones_grados)
            axes[1, 0].set_title("Rotaciones — antes")
            axes[1, 0].set_ylabel("grados")
            axes[1, 0].set_xlabel("Volumen")
            axes[1, 0].legend(
                ["X", "Y", "Z"],
                ncol=3,
                frameon=False,
            )

            axes[1, 1].plot(volumenes, rotaciones_despues)
            axes[1, 1].set_title("Rotaciones — después")
            axes[1, 1].set_ylabel("grados")
            axes[1, 1].set_xlabel("Volumen")
            axes[1, 1].set_ylim(axes[1, 0].get_ylim())
            axes[1, 1].legend(
                ["X", "Y", "Z"],
                ncol=3,
                frameon=False,
            )
            plt.tight_layout()
            plt.show()

            maximo_fd = max(
                float(fd.max()),
                float(fd_despues.max()),
                umbral_fd,
                1e-6,
            )
            bins_fd = np.linspace(0, maximo_fd * 1.05, 41)

            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
            axes[0].hist(
                fd,
                bins=bins_fd,
                color="tab:blue",
                alpha=0.70,
                label="Antes",
            )
            axes[0].hist(
                fd_despues,
                bins=bins_fd,
                color="tab:orange",
                alpha=0.70,
                label="Después",
            )
            axes[0].axvline(umbral_fd, color="black", linestyle=":")
            axes[0].set_title("Histograma de FD")
            axes[0].set_xlabel("FD caja (mm)")
            axes[0].set_ylabel("Volúmenes")
            axes[0].legend(frameon=False)

            axes[1].boxplot(
                [fd, fd_despues],
                showfliers=True,
            )
            axes[1].set_xticks([1, 2])
            axes[1].set_xticklabels(["Antes", "Después"])
            axes[1].axhline(umbral_fd, color="black", linestyle=":")
            axes[1].set_title("Boxplot de FD")
            axes[1].set_ylabel("FD caja (mm)")
            plt.tight_layout()
            plt.show()

            print(
                "Después = movimiento residual reestimado mediante "
                "BOLDRigid sobre la serie corregida."
            )

            if incluir_senal:
                bold_medio = image.mean_img(str(fila["corrected_path"]))
                display(
                    plotting.view_img(
                        bold_medio,
                        bg_img=False,
                        cmap="gray",
                        symmetric_cmap=False,
                        threshold=None,
                        colorbar=False,
                        black_bg=True,
                        draw_cross=True,
                        width_view=700,
                        title="BOLD medio después de corregir movimiento",
                    )
                )

    selector.observe(actualizar, names="value")
    display(selector, panel)
    actualizar()

    return resumen


def mostrar_antes_despues_movimiento(resultados):
    """Compara el volumen con mayor FD antes y después de corregirlo."""

    if resultados.empty:
        raise ValueError("No hay correcciones de movimiento para comparar.")

    opciones = []
    for posicion, fila in enumerate(resultados.itertuples()):
        etiqueta = f"sub-{fila.subject}"
        if fila.session:
            etiqueta += f" | ses-{fila.session}"
        opciones.append((etiqueta, posicion))

    selector = widgets.Dropdown(options=opciones, description="BOLD:")
    panel = widgets.Output()

    def actualizar(change=None):
        fila = resultados.iloc[selector.value]
        series, _ = _cargar_series_qc(fila, incluir_senal=False)
        indice = int(series["fd_caja_mm"].to_numpy().argmax())
        fd_maximo = float(series.loc[indice, "fd_caja_mm"])

        antes_4d = nib.load(fila["input_path"])
        despues_4d = nib.load(fila["corrected_path"])
        antes = np.asarray(
            antes_4d.dataobj[..., indice],
            dtype=np.float32,
        )
        despues = np.asarray(
            despues_4d.dataobj[..., indice],
            dtype=np.float32,
        )

        if antes.shape != despues.shape:
            raise RuntimeError(
                "El BOLD anterior y posterior no comparten dimensiones."
            )

        valores = np.concatenate(
            [
                antes[np.isfinite(antes) & (antes > 0)],
                despues[np.isfinite(despues) & (despues > 0)],
            ]
        )
        if valores.size == 0:
            raise RuntimeError("Los volúmenes seleccionados están vacíos.")

        vmin, vmax = np.percentile(valores, (1, 99))
        antes_img = nib.Nifti1Image(
            antes,
            antes_4d.affine,
            antes_4d.header.copy(),
        )
        despues_img = nib.Nifti1Image(
            despues,
            despues_4d.affine,
            despues_4d.header.copy(),
        )

        diferencia = np.abs(despues - antes).astype(np.float32)
        diferencias_positivas = diferencia[diferencia > 0]
        umbral_diferencia = (
            float(np.percentile(diferencias_positivas, 80))
            if diferencias_positivas.size
            else 1e-6
        )
        diferencia_img = nib.Nifti1Image(
            diferencia,
            despues_4d.affine,
            despues_4d.header.copy(),
        )

        with panel:
            clear_output(wait=True)
            print(
                f"Volumen seleccionado: {indice} | "
                f"FD caja: {fd_maximo:.3f} mm"
            )
            print("Se usa automáticamente el volumen con mayor movimiento.")

            display(
                plotting.view_img(
                    antes_img,
                    bg_img=False,
                    cmap="gray",
                    symmetric_cmap=False,
                    threshold=None,
                    vmin=float(vmin),
                    vmax=float(vmax),
                    colorbar=False,
                    black_bg=True,
                    draw_cross=True,
                    width_view=700,
                    title="Antes: volumen BOLD sin corregir",
                )
            )
            display(
                plotting.view_img(
                    despues_img,
                    bg_img=False,
                    cmap="gray",
                    symmetric_cmap=False,
                    threshold=None,
                    vmin=float(vmin),
                    vmax=float(vmax),
                    colorbar=False,
                    black_bg=True,
                    draw_cross=True,
                    width_view=700,
                    title="Después: el mismo volumen corregido",
                )
            )
            display(
                plotting.view_img(
                    diferencia_img,
                    bg_img=despues_img,
                    cmap="hot",
                    symmetric_cmap=False,
                    threshold=umbral_diferencia,
                    colorbar=False,
                    black_bg=True,
                    draw_cross=True,
                    opacity=0.65,
                    width_view=700,
                    title="Cambios producidos por la corrección",
                )
            )

    selector.observe(actualizar, names="value")
    display(selector, panel)
    actualizar()
