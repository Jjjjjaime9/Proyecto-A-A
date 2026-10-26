"""Preprocesamiento del conjunto de datos de diabetes del consorcio.

El módulo separa dos tipos de transformaciones:

1. **Limpieza determinista** (`limpiar`, `anadir_features`): no aprende nada de los
   datos, por lo que se puede aplicar igual al conjunto de desarrollo, a la partición
   oculta del profesor o a una petición de la API.
2. **Transformaciones ajustadas** (`construir_preprocesador`): imputación y escalado.
   Se ajustan *solo* sobre el conjunto de entrenamiento para evitar fuga de información
   y se guardan como artefacto (`joblib`) junto al modelo.

El preprocesador final es un `Pipeline` de sklearn que recibe los datos **crudos**
(tal y como salen del CSV) y devuelve la matriz lista para el modelo.
"""

from __future__ import annotations

import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# ---------------------------------------------------------------------------
# Definición de columnas
# ---------------------------------------------------------------------------

RUTA_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_DATOS_CRUDOS = RUTA_PROYECTO / "proy_diabetes_dev.csv"
RUTA_PROCESADOS = RUTA_PROYECTO / "data" / "processed"
RUTA_PREPROCESADORES = RUTA_PROYECTO / "models" / "preprocesadores"

SEMILLA = 42

COL_ID = "id_paciente_proy"

# Datos personales ofuscados: no aportan nada y están sujetos a protección de datos.
COLS_PII = [
    "dni",
    "nombre",
    "apellidos",
    "direccion",
    "telefono",
    "email",
    "num_historia_clinica",
]

# Columnas de gestión interna sin valor predictivo (ver EDA, sección 2).
COLS_GESTION = ["usuario_alta_datos"]

TARGET_CLASIF = "diabetes"
TARGET_REGR = "dias_salud_fisica_mala"

# Variables sí/no del archivo original (algunas con codificaciones mezcladas).
COLS_BINARIAS_ORIGINALES = [
    "presion_alta",
    "colesterol_alto",
    "chequeo_colesterol_5a",
    "fumador",
    "ictus",
    "enfermedad_cardiaca",
    "actividad_fisica",
    "consume_fruta",
    "consume_verdura",
    "consumo_alcohol_alto",
    "dificultad_caminar",
    TARGET_CLASIF,
]

ORDEN_SALUD_GENERAL = {"excelente": 1, "muy buena": 2, "buena": 3, "regular": 4, "mala": 5}
ORDEN_EDAD = {
    "18-24": 1,
    "25-29": 2,
    "30-34": 3,
    "35-39": 4,
    "40-44": 5,
    "45-49": 6,
    "50-54": 7,
    "55-59": 8,
    "60-64": 9,
    "65-69": 10,
    "70-74": 11,
    "75-79": 12,
    "80+": 13,
}
ORDEN_EDUCACION = {
    "sin estudios": 1,
    "primaria": 2,
    "secundaria incompleta": 3,
    "secundaria": 4,
    "formacion superior parcial": 5,
    "titulado superior": 6,
}
ORDEN_INGRESOS = {
    "<10k": 1,
    "10-15k": 2,
    "15-20k": 3,
    "20-25k": 4,
    "25-35k": 5,
    "35-50k": 6,
    "50-75k": 7,
    ">75k": 8,
}
MAPAS_ORDINALES = {
    "grupo_edad": ORDEN_EDAD,
    "salud_general": ORDEN_SALUD_GENERAL,
    "nivel_educativo": ORDEN_EDUCACION,
    "nivel_ingresos": ORDEN_INGRESOS,
}

# Categorías de IMC de la OMS (límite inferior de cada categoría).
CATEGORIAS_IMC = {
    1: "Bajo peso (<18,5)",
    2: "Normal (18,5-25)",
    3: "Sobrepeso (25-30)",
    4: "Obesidad I (30-35)",
    5: "Obesidad II (35-40)",
    6: "Obesidad III (≥40)",
}
_CORTES_IMC = [-np.inf, 18.5, 25, 30, 35, 40, np.inf]

# Rangos plausibles: lo que caiga fuera se considera error de registro y pasa a nulo.
RANGO_IMC = (10, 100)
RANGO_DIAS = (0, 30)

# Columnas tras la limpieza, agrupadas por tipo.
COLS_CONTINUAS = ["imc", "dias_salud_mental_mala"]
COLS_ORDINALES = list(MAPAS_ORDINALES)
COLS_BINARIAS = [c for c in COLS_BINARIAS_ORIGINALES if c != TARGET_CLASIF] + ["es_hombre"]
COLS_FEATURES_NUEVAS_ORD = ["imc_categoria", "n_factores_riesgo_cv"]
COLS_FEATURES_NUEVAS_BIN = ["falta_habitos", "falta_imc"]

# Variables usadas para clusterizar: perfil de salud y hábitos. Las socioeconómicas
# (sexo, educación, ingresos) se dejan fuera a propósito para describir los clústeres
# a posteriori, no para formarlos.
COLS_CLUSTERING = [
    "grupo_edad",
    "imc",
    "salud_general",
    "dias_salud_fisica_mala",
    "dias_salud_mental_mala",
    "presion_alta",
    "colesterol_alto",
    "diabetes",
    "enfermedad_cardiaca",
    "dificultad_caminar",
    "actividad_fisica",
    "fumador",
]

LINEAS = ("clasificacion", "regresion", "clustering")

# ---------------------------------------------------------------------------
# Limpieza determinista
# ---------------------------------------------------------------------------

_VERDADERO = {"si", "s", "yes", "y", "true", "verdadero", "1", "1.0"}
_FALSO = {"no", "n", "false", "falso", "0", "0.0"}


def cargar_datos(ruta: str | Path = RUTA_DATOS_CRUDOS) -> pd.DataFrame:
    """Lee el CSV crudo sin aplicar ninguna transformación."""
    return pd.read_csv(ruta)


def _normalizar_texto(serie: pd.Series) -> pd.Series:
    """Minúsculas, sin tildes y sin espacios sobrantes; los nulos se conservan."""

    def _limpiar(valor):
        if pd.isna(valor):
            return np.nan
        texto = unicodedata.normalize("NFKD", str(valor)).encode("ascii", "ignore").decode()
        return " ".join(texto.strip().lower().split())

    return serie.map(_limpiar)


def _avisar_desconocidos(serie: pd.Series, convertida: pd.Series) -> None:
    desconocidos = serie[serie.notna() & convertida.isna()].unique()
    if len(desconocidos):
        warnings.warn(
            f"'{serie.name}': valores no reconocidos convertidos a nulo: {list(desconocidos)[:10]}",
            stacklevel=3,
        )


def normalizar_binaria(serie: pd.Series) -> pd.Series:
    """Unifica las distintas codificaciones de una variable sí/no en 1.0 / 0.0 / NaN.

    Acepta `si/no`, `SI/NO`, `True/False`, `1.0/0.0`, etc. Un valor no reconocido
    se convierte en nulo (y se avisa) para que lo trate el imputador, en lugar de
    romper la predicción de una instancia nueva.
    """
    texto = _normalizar_texto(serie)
    convertida = texto.map(lambda v: 1.0 if v in _VERDADERO else 0.0 if v in _FALSO else np.nan)
    _avisar_desconocidos(serie, convertida)
    return convertida.astype(float)


def codificar_ordinal(serie: pd.Series, mapa: dict[str, int]) -> pd.Series:
    """Convierte una variable ordinal en texto a su código entero (NaN si falta)."""
    texto = _normalizar_texto(serie)
    convertida = texto.map(mapa)
    _avisar_desconocidos(serie, convertida)
    return convertida.astype(float)


def _numerica_en_rango(serie: pd.Series, rango: tuple[float, float]) -> pd.Series:
    valores = pd.to_numeric(serie, errors="coerce")
    return valores.where(valores.between(*rango))


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza determinista del archivo crudo.

    - Usa `id_paciente_proy` como índice y elimina PII y columnas de gestión.
    - Unifica las variables binarias (incluido el target `diabetes`) a 0/1.
    - Codifica `sexo` como `es_hombre` y las ordinales a enteros.
    - Fuerza tipos numéricos y anula valores fuera de rango plausible.

    Funciona aunque falten columnas (p. ej. los targets en una petición de la API).
    """
    df = df.copy()
    if COL_ID in df.columns:
        df = df.set_index(COL_ID)
    df = df.drop(columns=COLS_PII + COLS_GESTION, errors="ignore")

    for col in COLS_BINARIAS_ORIGINALES:
        if col in df.columns:
            df[col] = normalizar_binaria(df[col])

    if "sexo" in df.columns:
        sexo = _normalizar_texto(df["sexo"])
        df["es_hombre"] = sexo.map({"hombre": 1.0, "mujer": 0.0}).astype(float)
        _avisar_desconocidos(df["sexo"], df["es_hombre"])
        df = df.drop(columns="sexo")

    for col, mapa in MAPAS_ORDINALES.items():
        if col in df.columns:
            df[col] = codificar_ordinal(df[col], mapa)

    if "imc" in df.columns:
        df["imc"] = _numerica_en_rango(df["imc"], RANGO_IMC)
    for col in ("dias_salud_mental_mala", TARGET_REGR):
        if col in df.columns:
            df[col] = _numerica_en_rango(df[col], RANGO_DIAS)

    orden = COLS_CONTINUAS + COLS_ORDINALES + COLS_BINARIAS + [TARGET_CLASIF, TARGET_REGR]
    return df[[c for c in orden if c in df.columns]]


def anadir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables derivadas sobre datos ya limpios (sin usar ningún target).

    - `imc_categoria`: categoría OMS del IMC (1-6), más interpretable para clústeres.
    - `n_factores_riesgo_cv`: nº de factores de riesgo cardiovascular presentes
      (hipertensión, colesterol alto, tabaquismo, cardiopatía, ictus).
    - `falta_habitos` / `falta_imc`: indicadores de los dos bloques de nulos
      detectados en el EDA, por si la ausencia en sí es informativa.
    """
    df = df.copy()
    df["imc_categoria"] = pd.cut(df["imc"], _CORTES_IMC, labels=False, right=False) + 1
    factores = ["presion_alta", "colesterol_alto", "fumador", "enfermedad_cardiaca", "ictus"]
    df["n_factores_riesgo_cv"] = df[factores].sum(axis=1, min_count=1)
    df["falta_habitos"] = df["fumador"].isna().astype(float)
    df["falta_imc"] = df["imc"].isna().astype(float)
    return df


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza + variables derivadas. Es el primer paso de todos los preprocesadores."""
    return anadir_features(limpiar(df))


# ---------------------------------------------------------------------------
# Variables por línea de trabajo
# ---------------------------------------------------------------------------


def columnas_features(linea: str) -> dict[str, list[str]]:
    """Columnas de entrada por tipo para cada línea de trabajo.

    - Clasificación: todo salvo `diabetes` (los días de mala salud física son una
      respuesta más de la encuesta y, por tanto, una entrada legítima).
    - Regresión: todo salvo `dias_salud_fisica_mala` (`diabetes` sí entra).
    - Clustering: subconjunto interpretable `COLS_CLUSTERING`.
    """
    if linea == "clustering":
        return {
            "continuas": [c for c in COLS_CLUSTERING if c in COLS_CONTINUAS + [TARGET_REGR]],
            "ordinales": [c for c in COLS_CLUSTERING if c in COLS_ORDINALES],
            "binarias": [c for c in COLS_CLUSTERING if c in COLS_BINARIAS + [TARGET_CLASIF]],
        }
    if linea == "clasificacion":
        return {
            "continuas": COLS_CONTINUAS + [TARGET_REGR],
            "ordinales": COLS_ORDINALES + COLS_FEATURES_NUEVAS_ORD,
            "binarias": COLS_BINARIAS + COLS_FEATURES_NUEVAS_BIN,
        }
    if linea == "regresion":
        return {
            "continuas": COLS_CONTINUAS,
            "ordinales": COLS_ORDINALES + COLS_FEATURES_NUEVAS_ORD,
            "binarias": COLS_BINARIAS + COLS_FEATURES_NUEVAS_BIN + [TARGET_CLASIF],
        }
    raise ValueError(f"Línea desconocida: {linea!r}. Opciones: {LINEAS}")


def construir_preprocesador(linea: str) -> Pipeline:
    """Pipeline crudo → matriz lista para modelar.

    1. `preparar`: limpieza determinista + variables derivadas.
    2. `ColumnTransformer`:
       - continuas y ordinales: imputación por mediana + estandarización;
       - binarias: imputación por la moda (se quedan en 0/1). En clustering también
         se estandarizan para que todas las variables pesen igual en las distancias.
    """
    cols = columnas_features(linea)
    numerico = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    binario = SimpleImputer(strategy="most_frequent")
    if linea == "clustering":
        binario = make_pipeline(binario, StandardScaler())

    columnas = ColumnTransformer(
        [
            ("num", numerico, cols["continuas"] + cols["ordinales"]),
            ("bin", binario, cols["binarias"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    return Pipeline(
        [
            ("preparar", FunctionTransformer(preparar)),
            ("columnas", columnas),
        ]
    )


# ---------------------------------------------------------------------------
# Particiones
# ---------------------------------------------------------------------------


def tramos_regresion(y: pd.Series) -> pd.Series:
    """Tramos del target de regresión para estratificar: 0, 1-29 y 30 días."""
    return pd.cut(y, [-0.5, 0.5, 29.5, 30.5], labels=["0", "1-29", "30"])


def separar_target(df_crudo: pd.DataFrame, linea: str) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve (X crudo, y limpio) indexados por `id_paciente_proy`.

    X se mantiene crudo (sin el target) porque el preprocesador empieza limpiando.
    """
    target = {"clasificacion": TARGET_CLASIF, "regresion": TARGET_REGR}[linea]
    y = limpiar(df_crudo[[COL_ID, target]])[target].astype(int)
    X = df_crudo.drop(columns=target).set_index(COL_ID, drop=False)
    X.index.name = None
    return X, y


def dividir(
    df_crudo: pd.DataFrame, linea: str, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Partición train/test estratificada (por clase o por tramos del target)."""
    X, y = separar_target(df_crudo, linea)
    estratos = y if linea == "clasificacion" else tramos_regresion(y)
    return train_test_split(
        X, y, test_size=test_size, stratify=estratos, random_state=SEMILLA
    )
