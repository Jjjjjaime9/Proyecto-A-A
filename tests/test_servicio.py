"""Pruebas del servicio (`src/servicio.py`) contra el contrato de la API común.

Ejecutar desde la raíz:  python -m pytest tests -q
"""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import ARTEFACTO, SRC, a_instancias

import modelado as md
import preprocesamiento as pp

UMBRAL = md.CONFIG_FINAL["clasificacion"]["umbral"]


def pedir(cliente, instancias, linea="clasificacion"):
    return cliente.post("/predict", json={"linea": linea, "instancias": instancias})


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_declara_la_linea_servida(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["estado"] == "ok"
    assert cuerpo["lineas"] == ["clasificacion"]
    assert cuerpo["modelo_version"]


# ---------------------------------------------------------------------------
# /predict: caso normal
# ---------------------------------------------------------------------------


def test_una_prediccion_valida_por_instancia(cliente, test_crudo):
    """200 instancias tal y como salen del CSV (con PII, id, gestión y el otro objetivo)."""
    instancias = a_instancias(test_crudo.head(200))
    r = pedir(cliente, instancias)
    assert r.status_code == 200, r.text
    cuerpo = r.json()
    assert cuerpo["linea"] == "clasificacion"
    assert len(cuerpo["predicciones"]) == len(instancias)
    for p in cuerpo["predicciones"]:
        assert set(p) == {"clase", "probabilidad"}
        assert p["clase"] in (0, 1)
        assert 0.0 <= p["probabilidad"] <= 1.0
        assert p["clase"] == int(p["probabilidad"] >= UMBRAL)


def test_la_api_devuelve_lo_mismo_que_el_modelo(cliente, modelo, test_crudo):
    muestra = test_crudo.head(100)
    r = pedir(cliente, a_instancias(muestra))
    proba_api = np.array([p["probabilidad"] for p in r.json()["predicciones"]])
    proba_directa = modelo.predict_proba(muestra[list(modelo.feature_names_in_)])[:, 1]
    np.testing.assert_allclose(proba_api, proba_directa, atol=1e-9)


def test_el_objetivo_en_la_peticion_se_ignora(cliente, crudo):
    """Si llegan filas completas (con `diabetes`), el objetivo no influye en la predicción."""
    filas = crudo.head(50)
    con_objetivo = pedir(cliente, a_instancias(filas)).json()["predicciones"]
    sin_objetivo = pedir(cliente, a_instancias(filas.drop(columns=pp.TARGET_CLASIF))).json()["predicciones"]
    assert con_objetivo == sin_objetivo


def test_no_exige_el_objetivo_de_la_otra_linea(cliente, test_crudo):
    """Revisión del PC1: `dias_salud_fisica_mala` no es necesaria para clasificar."""
    instancias = a_instancias(test_crudo.head(20).drop(columns=pp.TARGET_REGR))
    assert pedir(cliente, instancias).status_code == 200


def test_basta_con_las_columnas_minimas(cliente, test_crudo):
    instancias = a_instancias(test_crudo.head(20)[pp.columnas_crudas("clasificacion")])
    assert pedir(cliente, instancias).status_code == 200


# ---------------------------------------------------------------------------
# /predict: datos sucios (la API nunca debe descartar una instancia)
# ---------------------------------------------------------------------------


def test_codificaciones_nulos_y_valores_imposibles(cliente, test_crudo):
    base = a_instancias(test_crudo.head(1))[0]
    sucias = [
        {**base, "fumador": "Sí", "actividad_fisica": "YES", "salud_general": " Muy Buena ", "sexo": "MUJER"},
        {**base, "presion_alta": True, "colesterol_alto": 1.0, "ictus": "False", "dificultad_caminar": 0},
        {**base, "imc": 250, "dias_salud_mental_mala": -3, "grupo_edad": "edad desconocida"},
        {clave: None for clave in base},  # paciente sin ningún dato
    ]
    r = pedir(cliente, sucias)
    assert r.status_code == 200, r.text
    probas = [p["probabilidad"] for p in r.json()["predicciones"]]
    assert len(probas) == len(sucias)
    assert all(0.0 <= p <= 1.0 for p in probas)


def test_instancia_a_la_que_le_falta_una_clave(cliente, test_crudo):
    """Una instancia sin `imc` (las demás sí lo traen) se trata como dato ausente."""
    instancias = a_instancias(test_crudo.head(3))
    del instancias[1]["imc"]
    r = pedir(cliente, instancias)
    assert r.status_code == 200, r.text
    assert len(r.json()["predicciones"]) == 3


# ---------------------------------------------------------------------------
# /predict: errores del contrato (422)
# ---------------------------------------------------------------------------


def test_columna_requerida_ausente_en_todas_las_instancias(cliente, test_crudo):
    r = pedir(cliente, a_instancias(test_crudo.head(5).drop(columns="imc")))
    assert r.status_code == 422
    assert "imc" in r.json()["detail"]


@pytest.mark.parametrize("linea", ["regresion", "clusterizacion"])
def test_linea_no_servida(cliente, test_crudo, linea):
    r = pedir(cliente, a_instancias(test_crudo.head(2)), linea=linea)
    assert r.status_code == 422
    assert "no está servida" in r.json()["detail"]


@pytest.mark.parametrize(
    "cuerpo",
    [
        {"linea": "clasificacion", "instancias": []},
        {"linea": "diabetes", "instancias": [{"imc": 30}]},
        {"instancias": [{"imc": 30}]},
    ],
    ids=["sin-instancias", "linea-desconocida", "sin-linea"],
)
def test_peticion_mal_formada(cliente, cuerpo):
    assert cliente.post("/predict", json=cuerpo).status_code == 422


# ---------------------------------------------------------------------------
# Artefacto
# ---------------------------------------------------------------------------


def test_el_artefacto_carga_desde_cualquier_directorio(tmp_path):
    """Proceso limpio, otro directorio de trabajo y solo `src/` en el path (como el servicio)."""
    codigo = (
        "import sys, joblib, pandas as pd; "
        f"sys.path.append({str(SRC)!r}); "
        f"m = joblib.load({str(ARTEFACTO)!r}); "
        "print(len(m.feature_names_in_))"
    )
    salida = subprocess.run([sys.executable, "-c", codigo], cwd=tmp_path, capture_output=True, text=True)
    assert salida.returncode == 0, salida.stderr
    assert salida.stdout.strip() == "18"


def test_artefacto_pide_solo_columnas_crudas_sin_pii_ni_objetivos(modelo):
    columnas = list(modelo.feature_names_in_)
    assert columnas == pp.columnas_crudas("clasificacion")
    prohibidas = set(pp.COLS_PII + pp.COLS_GESTION + [pp.COL_ID, pp.TARGET_CLASIF, pp.TARGET_REGR])
    assert not prohibidas & set(columnas)
