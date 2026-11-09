"""Configuración común de las pruebas: rutas, datos crudos y cliente del servicio."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(SRC))

import preprocesamiento as pp  # noqa: E402

ARTEFACTO = RAIZ / "artifacts" / "proy_modelo_clasificacion.joblib"


def a_instancias(df: pd.DataFrame) -> list[dict]:
    """Filas → lista de dicts serializable en JSON, como la enviaría un cliente real (NaN → null)."""
    return json.loads(df.to_json(orient="records"))


@pytest.fixture(scope="session")
def crudo() -> pd.DataFrame:
    return pp.cargar_datos()


@pytest.fixture(scope="session")
def test_crudo(crudo) -> pd.DataFrame:
    """Pacientes de test del notebook 03, con todas sus columnas crudas salvo el objetivo."""
    _, X_test, _, _ = pp.dividir(crudo, "clasificacion")
    return X_test


@pytest.fixture(scope="session")
def cliente():
    if not ARTEFACTO.is_file():
        pytest.skip("Falta el artefacto: ejecutar antes `python src/entrenar.py clasificacion`")
    from fastapi.testclient import TestClient

    from src.servicio import app

    return TestClient(app)


@pytest.fixture(scope="session")
def modelo():
    import joblib

    return joblib.load(ARTEFACTO)
