"""Genera el ZIP de entrega con todo lo necesario para reproducir y servir el proyecto.

Uso, desde la raíz del repositorio:

    python src/empaquetar.py                 # → entregas/proyecto_diabetes.zip
    python src/empaquetar.py --nombre PC2    # → entregas/PC2.zip

En el PC1 el ZIP llegó sin `data/processed/` ni los artefactos. Este script incluye
explícitamente cada bloque y **falla** si falta alguno, en lugar de dejar fuera en
silencio una carpeta imprescindible.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "entregas"

# (patrón glob relativo a la raíz, ¿obligatorio?)
CONTENIDO = [
    ("README.md", True),
    ("requirements.txt", True),
    ("proy_diabetes_dev.csv", True),
    ("src/*.py", True),
    ("notebooks/*.ipynb", True),
    ("data/processed/**/*.parquet", True),
    ("data/processed/*.csv", True),
    ("models/preprocesadores/*.joblib", True),
    ("artifacts/*.joblib", True),
    ("artifacts/*.json", True),
    ("reports/figures/*.png", True),
    ("reports/*.json", True),
    ("tests/*.py", True),
    ("ejemplos/*.json", True),
]


def recopilar() -> list[Path]:
    archivos, faltan = [], []
    for patron, obligatorio in CONTENIDO:
        encontrados = sorted(p for p in RAIZ.glob(patron) if p.is_file())
        if obligatorio and not encontrados:
            faltan.append(patron)
        archivos.extend(encontrados)
    if faltan:
        raise SystemExit(
            "ERROR: faltan bloques obligatorios en el ZIP: " + ", ".join(faltan)
            + "\nEjecutar antes los notebooks 02/03 y `python src/entrenar.py clasificacion`."
        )
    return list(dict.fromkeys(archivos))


def empaquetar(nombre: str) -> Path:
    archivos = recopilar()
    DESTINO.mkdir(exist_ok=True)
    ruta_zip = DESTINO / f"{nombre}.zip"
    with zipfile.ZipFile(ruta_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for archivo in archivos:
            zf.write(archivo, Path(nombre) / archivo.relative_to(RAIZ))

    with zipfile.ZipFile(ruta_zip) as zf:
        nombres = zf.namelist()
    for bloque in ("data/processed/", "artifacts/", "models/preprocesadores/", "src/servicio.py"):
        assert any(f"/{bloque}" in n for n in nombres), f"El ZIP no contiene {bloque}"

    print(f"[OK] {ruta_zip.relative_to(RAIZ)}: {len(nombres)} archivos, {ruta_zip.stat().st_size / 1e6:.1f} MB")
    carpetas = sorted({"/".join(n.split("/")[1:-1]) or "(raíz)" for n in nombres})
    for carpeta in carpetas:
        n = sum(1 for x in nombres if ("/".join(x.split("/")[1:-1]) or "(raíz)") == carpeta)
        print(f"     {carpeta:<35} {n:>3} archivos")
    return ruta_zip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nombre", default="proyecto_diabetes", help="nombre del ZIP (sin extensión)")
    empaquetar(parser.parse_args().nombre)


if __name__ == "__main__":
    main()
