"""
UIE · Aprendizaje Automático · Curso 2026-27
Profesor: Borja González Seoane

Servicio
========

Plantilla de servicio HTTP para los modelos del proyecto de la asignatura.

Implementa el contrato de API común descrito en el enunciado del proyecto (`Proyecto.pdf`).

Para levantar el servicio, desde la raíz del repositorio:

    uvicorn src.servicio:app --host 0.0.0.0 --port 8000

Y para comprobarlo:

    curl http://localhost:8000/health
"""

import os
import sys
from pathlib import Path
from typing import Any, Literal, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODELO_VERSION = "1.0.0"

DIR_ARTEFACTOS = Path(__file__).resolve().parents[1] / "artifacts"


sys.path.append(str(Path(__file__).resolve().parent))

# Prefijo de los archivos de artefacto. Cada equipo puede dejar el que viene por defecto y nombrar así sus artefactos, o apuntar a los suyos sin tocar el código:
#
#     PROY_PREFIJO_ARTEFACTOS=proy_2026_27_modelo uvicorn src.servicio:app
PREFIJO = os.environ.get("PROY_PREFIJO_ARTEFACTOS", "proy_modelo")

# Mapeo de línea de trabajo a archivo de artefacto. Las líneas cuyo artefacto no exista no se sirven, de modo que un equipo puede desplegar sólo las que tenga operativas
ARTEFACTOS = {
    linea: f"{PREFIJO}_{linea}.joblib"
    for linea in ("clasificacion", "regresion", "clusterizacion")
}

Linea = Literal["clasificacion", "regresion", "clusterizacion"]


def _cargar_modelos() -> dict[str, Any]:
    """Carga los artefactos disponibles en el arranque del servicio."""
    modelos = {}
    for linea, archivo in ARTEFACTOS.items():
        ruta = DIR_ARTEFACTOS / archivo
        if ruta.is_file():
            modelos[linea] = joblib.load(ruta)
    return modelos


MODELOS = _cargar_modelos()

app = FastAPI(
    title="Servicio Proyecto de Aprendizaje Automático - UIE",
    version=MODELO_VERSION,
)


class PeticionPrediccion(BaseModel):
    linea: Linea
    instancias: list[dict[str, Any]] = Field(min_length=1)


class PrediccionClasificacion(BaseModel):
    clase: int
    probabilidad: float = Field(ge=0.0, le=1.0)


class RespuestaSalud(BaseModel):
    estado: str
    lineas: list[str]
    modelo_version: str


class RespuestaPrediccion(BaseModel):
    linea: str
    modelo_version: str
    predicciones: list[Union[PrediccionClasificacion, float, int]]


def _preparar_marco(instancias: list[dict[str, Any]], modelo: Any) -> pd.DataFrame:
    """Construye el `DataFrame` de entrada que espera el artefacto.

    Las instancias llegan con los nombres de columna originales del conjunto de
    datos y sin preprocesar. Se descartan las columnas que el modelo no utiliza
    y se comprueba que no falte ninguna de las que sí necesita. Fijar el orden
    de las columnas es imprescindible: Scikit-Learn valida por nombre, pero
    muchos artefactos personalizados asumen posición.
    """
    marco = pd.DataFrame(instancias)

    columnas_esperadas = getattr(modelo, "feature_names_in_", None)
    if columnas_esperadas is None:
        return marco

    faltantes = [c for c in columnas_esperadas if c not in marco.columns]
    if faltantes:
        raise HTTPException(
            status_code=422,
            detail=f"Faltan columnas requeridas por el modelo: {sorted(faltantes)}",
        )

    return marco[list(columnas_esperadas)]


@app.get("/health", response_model=RespuestaSalud)
def salud() -> RespuestaSalud:
    """Declara que el servicio está vivo y qué líneas de trabajo sirve."""
    return RespuestaSalud(
        estado="ok",
        lineas=sorted(MODELOS.keys()),
        modelo_version=MODELO_VERSION,
    )


@app.post("/predict", response_model=RespuestaPrediccion)
def predecir(peticion: PeticionPrediccion) -> RespuestaPrediccion:
    """Devuelve las predicciones de una de las líneas de trabajo."""
    modelo = MODELOS.get(peticion.linea)
    if modelo is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"La línea «{peticion.linea}» no está servida por este equipo. "
                f"Líneas disponibles: {sorted(MODELOS.keys())}"
            ),
        )

    marco = _preparar_marco(peticion.instancias, modelo)

    try:
        crudas = modelo.predict(marco)

        if peticion.linea == "clasificacion":
            if not hasattr(modelo, "predict_proba"):
                raise ValueError(
                    "el modelo de clasificación no implementa `predict_proba`, "
                    "necesario para devolver la probabilidad que exige el contrato"
                )
            probabilidades = modelo.predict_proba(marco)[:, 1]
            predicciones = [
                PrediccionClasificacion(
                    clase=int(clase), probabilidad=float(probabilidad)
                )
                for clase, probabilidad in zip(crudas, probabilidades)
            ]
        elif peticion.linea == "regresion":
            predicciones = [float(valor) for valor in crudas]
        else:
            predicciones = [int(valor) for valor in crudas]
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"No se pudo predecir: {error}"
        ) from error

    if len(predicciones) != len(peticion.instancias):
        raise HTTPException(
            status_code=422,
            detail=(
                f"El modelo devolvió {len(predicciones)} predicciones para "
                f"{len(peticion.instancias)} instancias. El contrato exige una "
                "predicción por instancia: hay que imputar los ausentes, no descartar filas"
            ),
        )

    return RespuestaPrediccion(
        linea=peticion.linea,
        modelo_version=MODELO_VERSION,
        predicciones=predicciones,
    )
