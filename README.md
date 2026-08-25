# Workshop: fMRI preprocessing basics

Taller práctico de preprocesamiento anatómico y funcional de resonancia
magnética funcional en Python. Incluye preparación de imágenes T1w y BOLD,
corrección de movimiento, registro a espacio MNI, segmentación, denoising,
extracción de series temporales y métricas básicas de control de calidad.

## Contenido

- `notebooks/workshop.ipynb`: notebook principal del taller, basado en ANTsPy.
- `notebooks/rsfmri_preproc_large_version_COLAB.ipynb`: versión extendida para
  Colab/Linux, con MRIQC, fMRIPrep y Neurodesk.
- `src/anat_preproc.py`: preprocesamiento anatómico.
- `src/func_preproc.py`: preprocesamiento funcional y conectividad.
- `src/io_utils.py`: rutas e inventario de datos.
- `data/`: datos BIDS de entrada; no se incluyen en el repositorio.
- `derivatives/`: resultados generados; no se incluyen en el repositorio.

## Instalación

Se recomienda Python 3.11. Desde la raíz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Linux o macOS, la activación es:

```bash
source .venv/bin/activate
```

Para abrir el taller:

```bash
python -m jupyter lab
```

Luego abra `notebooks/workshop.ipynb` y ejecute las celdas en orden.

## Estructura mínima de los datos

El notebook principal espera un dataset BIDS dentro de `data/BIDS`, por ejemplo:

```text
data/BIDS/
└── sub-s003/
    └── ses-open/
        ├── anat/
        │   ├── sub-s003_ses-open_T1w.nii.gz
        │   └── sub-s003_ses-open_T1w.json
        └── func/
            ├── sub-s003_ses-open_task-rest_bold.nii.gz
            └── sub-s003_ses-open_task-rest_bold.json
```

Los productos se escriben en `derivatives/`.

## Notas

- La primera ejecución de ANTsPyNet puede descargar pesos de modelos y requiere
  conexión a internet.
- El notebook principal usa ANTs mediante `antspyx`; no requiere instalar los
  binarios de ANTs por separado.
- La versión extendida utiliza comandos de Linux y contenedores de Neurodesk para
  MRIQC y fMRIPrep. Se recomienda ejecutarla en Google Colab, Linux o WSL con
  suficiente RAM y espacio en disco.
- `google.colab` no aparece en `requirements.txt` porque ya viene incluido en
  Google Colab.
- MRIQC, fMRIPrep, AFNI, FSL y FreeSurfer no son simples dependencias de `pip` en
  este proyecto: el notebook extendido los ejecuta mediante contenedores.
- No suba `data/`, `derivatives/`, `.venv/`, imágenes NIfTI ni resultados del
  procesamiento al repositorio.
