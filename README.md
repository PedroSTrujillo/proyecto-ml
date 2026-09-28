# Clasificador de reseñas

Proyecto de machine learning para clasificar reseñas. El trabajo se desarrolla en un cuaderno de Jupyter usando pandas, scikit-learn, imbalanced-learn, matplotlib y seaborn.

## Instalación de dependencias

Las dependencias del proyecto están en [requirements.txt](requirements.txt). Se recomienda instalarlas dentro de un entorno virtual.

1. Clona el repositorio y entra en la carpeta del proyecto:

   ```bash
   cd proyecto-ml
   ```

2. Crea un entorno virtual:

   ```bash
   python3 -m venv .venv
   ```

3. Actívalo:

   - macOS / Linux:

     ```bash
     source .venv/bin/activate
     ```

   - Windows (PowerShell):

     ```powershell
     .venv\Scripts\Activate.ps1
     ```

4. Actualiza `pip` e instala las dependencias:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Abre el cuaderno (por ejemplo desde VS Code o con Jupyter) y selecciona el kernel del entorno `.venv`.

### Notas

- Las versiones están fijadas, por lo que se requiere una versión reciente de Python (las versiones de numpy 2.5 y pandas 3.0 necesitan Python 3.11 o superior).
- El paquete `appnope` solo aplica a macOS; si la instalación falla en otro sistema operativo, elimina esa línea de `requirements.txt`.
