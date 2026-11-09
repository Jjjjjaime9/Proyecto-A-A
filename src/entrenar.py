"""Entrena el modelo final de una línea de trabajo y guarda el artefacto que sirve la API.

Uso, desde la raíz del repositorio:

    python src/entrenar.py clasificacion

Reajusta `modelado.CONFIG_FINAL[linea]` (decidida y validada en el notebook 03) con
**todo** el conjunto de desarrollo: la configuración ya está cerrada, así que el test
no se necesita para nada más y añadirlo solo aporta datos. Genera:

- `artifacts/proy_modelo_<linea>.joblib`: el `Pipeline` completo (datos crudos → predicción),
  que es lo que carga `src/servicio.py`.
- `artifacts/proy_modelo_<linea>.json`: metadatos (versión, fecha, columnas, configuración y
  métricas de validación y test del notebook).

Se ejecuta como script (no como `python -m src.entrenar`) a propósito: así `src/` es
`sys.path[0]` y el artefacto referencia `preprocesamiento.*` y `modelado.*` como módulos
de primer nivel, igual que los resuelve el servicio desde cualquier directorio.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelado as md  # noqa: E402
import preprocesamiento as pp  # noqa: E402

RUTA_ARTEFACTOS = pp.RUTA_PROYECTO / "artifacts"
PREFIJO = "proy_modelo"
RUTA_METRICAS = pp.RUTA_PROYECTO / "reports" / "metricas_{linea}.json"

# Nombre de la línea en el contrato de la API → nombre interno en `preprocesamiento`
LINEA_INTERNA = {"clasificacion": "clasificacion", "regresion": "regresion", "clusterizacion": "clustering"}


def entrenar(linea: str) -> Path:
    interna = LINEA_INTERNA[linea]
    crudo = pp.cargar_datos()
    X, y = pp.separar_target(crudo, interna)
    columnas = pp.columnas_crudas(interna)
    X = X[columnas]

    modelo = md.construir_final(interna).fit(X, y)

    # Comprobación mínima antes de guardar: contrato de columnas y una predicción por fila
    assert list(modelo.feature_names_in_) == columnas
    proba = modelo.predict_proba(X.head(50))[:, 1]
    assert len(proba) == 50 and ((proba >= 0) & (proba <= 1)).all()

    RUTA_ARTEFACTOS.mkdir(parents=True, exist_ok=True)
    ruta_modelo = RUTA_ARTEFACTOS / f"{PREFIJO}_{linea}.joblib"
    joblib.dump(modelo, ruta_modelo, compress=3)

    ruta_metricas = Path(str(RUTA_METRICAS).format(linea=interna))
    metadatos = {
        "linea": linea,
        "modelo_version": md.VERSION_MODELO,
        "entrenado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_entrenamiento": int(len(y)),
        "prevalencia_entrenamiento": round(float(y.mean()), 4),
        "columnas_entrada": columnas,
        "configuracion": md.CONFIG_FINAL[interna],
        "evaluacion_notebook": json.loads(ruta_metricas.read_text(encoding="utf-8")) if ruta_metricas.is_file() else None,
        "entorno": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
    }
    ruta_json = ruta_modelo.with_suffix(".json")
    ruta_json.write_text(json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8")

    tam = ruta_modelo.stat().st_size / 1024
    print(f"[OK] {ruta_modelo.relative_to(pp.RUTA_PROYECTO)} ({tam:,.0f} KB) - {len(y):,} pacientes - {len(columnas)} columnas")
    print(f"[OK] {ruta_json.relative_to(pp.RUTA_PROYECTO)}")
    return ruta_modelo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("linea", choices=sorted(LINEA_INTERNA), help="línea de trabajo (nombre del contrato de la API)")
    args = parser.parse_args()
    entrenar(args.linea)


if __name__ == "__main__":
    main()
