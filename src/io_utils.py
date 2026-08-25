from pathlib import Path

import nibabel as nib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATHS = {
    "project": PROJECT_ROOT,
    "bids": PROJECT_ROOT / "data" / "BIDS",
    "derivatives": PROJECT_ROOT / "derivatives",
    "external": PROJECT_ROOT / "data" / "external",
    "antsxnet_cache": PROJECT_ROOT / "data" / "external" / "ANTsXNet",
}


def preparar_rutas():
    """Verifica las entradas y crea las carpetas de salida."""

    if not PATHS["bids"].is_dir():
        raise FileNotFoundError(
            f"No se encontró la carpeta BIDS: {PATHS['bids']}"
        )

    for nombre in ("derivatives", "external", "antsxnet_cache"):
        PATHS[nombre].mkdir(parents=True, exist_ok=True)

    return PATHS


def inventariar_t1w():
    """Localiza y verifica un T1w tridimensional por sujeto y sesión."""

    t1_paths = sorted(PATHS["bids"].rglob("*_T1w.nii.gz"))
    filas = []

    for ruta in t1_paths:
        sujeto = next(
            (p for p in ruta.parts if p.startswith("sub-")),
            "desconocido",
        )
        sesion = next(
            (p for p in ruta.parts if p.startswith("ses-")),
            "sin-sesion",
        )

        img = nib.load(ruta)

        filas.append({
            "sujeto": sujeto,
            "sesion": sesion,
            "dimensiones": img.shape,
            "voxel_mm": tuple(
                round(x, 3) for x in img.header.get_zooms()[:3]
            ),
            "orientacion": "".join(nib.aff2axcodes(img.affine)),
            "tamaño_MB": round(ruta.stat().st_size / 1024**2, 1),
            "ruta": ruta,
        })

    inventario = pd.DataFrame(filas)

    if inventario.empty:
        raise FileNotFoundError("No se encontró ningún archivo T1w.")

    if not inventario["dimensiones"].apply(lambda x: len(x) == 3).all():
        raise ValueError("Todos los T1w deben ser tridimensionales.")

    duplicados = inventario.groupby(["sujeto", "sesion"]).size()
    duplicados = duplicados[duplicados > 1]

    if not duplicados.empty:
        raise ValueError(
            f"Hay múltiples T1w por sujeto/sesión:\n{duplicados}"
        )

    return inventario