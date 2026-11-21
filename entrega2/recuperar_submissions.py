"""Recupera en el computador local los archivos submission generados en Colab.

La última celda del notebook, ejecutada en Colab, empaqueta los archivos ./datos/submission_*.csv en su
salida. Como VS Code guarda el notebook en el computador local con sus salidas, este script los extrae
de ahí y los escribe en ./datos (reemplazando los que ya existan con el mismo nombre).

Uso, después de ejecutar esa celda y guardar el notebook:  python recuperar_submissions.py
"""

import base64
import io
import json
import re
import sys
import zipfile
from pathlib import Path

carpeta = Path(__file__).parent
notebook = json.loads((carpeta / "modelo_entrega_2.ipynb").read_text(encoding="utf-8"))

salidas = "".join(
    "".join(salida.get("text", ""))
    for celda in notebook["cells"]
    for salida in celda.get("outputs", [])
)
paquetes = re.findall(r"<<<SUBMISSIONS ([A-Za-z0-9+/=]+)>>>", salidas)
if not paquetes:
    sys.exit(
        "No se encontraron archivos en las salidas del notebook: ejecuta en Colab la celda que "
        "empaqueta los archivos submission y guarda el notebook antes de correr este script."
    )

with zipfile.ZipFile(io.BytesIO(base64.b64decode(paquetes[-1]))) as archivo_zip:
    archivo_zip.extractall(carpeta)
    for nombre in archivo_zip.namelist():
        print(f"Recuperado: {carpeta / nombre}")
