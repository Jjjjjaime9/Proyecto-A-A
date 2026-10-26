# Proyecto de Aprendizaje Automático: encuesta de salud y diabetes

Proyecto de la asignatura de Aprendizaje Automático (UIE). A partir de la encuesta anual de salud de un consorcio de atención primaria (`proy_diabetes_dev.csv`) se abordan tres líneas de trabajo:

| Línea | Tarea | Objetivo |
|---|---|---|
| Clusterización | Perfiles de paciente interpretables | — |
| Clasificación | Cribado de diabetes | `diabetes` (sí/no) |
| Regresión | Carga de enfermedad | `dias_salud_fisica_mala` (0–30) |

## Estado: punto de control 1 (S09)

- [x] Análisis exploratorio completo → `notebooks/01_eda.ipynb`
- [x] Código de preprocesado → `src/preprocesamiento.py`
- [x] Versión preprocesada de los datos → `data/processed/` y `models/preprocesadores/`
- [ ] Modelado (punto de control 2)
- [ ] API común (punto de control 2)

## Estructura

```
Proyecto/
├── proy_diabetes_dev.csv           datos crudos (no se modifican)
├── requirements.txt
├── src/
│   ├── preprocesamiento.py         limpieza, variables derivadas, pipelines y particiones
│   └── eda_utils.py                estilo gráfico, estadísticos y gráficos del EDA
├── notebooks/
│   ├── 01_eda.ipynb                EDA (resumen ejecutivo, 10 secciones, 20 figuras)
│   └── 02_preprocesado.ipynb       ejecución del preprocesado y comprobaciones
├── data/processed/
│   ├── diabetes_limpio.parquet/.csv          limpio, sin imputar (análisis y perfilado)
│   ├── clasificacion/{X,y}_{train,test}.parquet
│   ├── regresion/{X,y}_{train,test}.parquet
│   └── clustering/X_clustering.parquet
├── models/preprocesadores/{clasificacion,regresion,clustering}.joblib
└── reports/figures/                figuras del EDA (PNG, 150 dpi) para informe y slides
```

## Cómo reproducir

```bash
pip install -r requirements.txt
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocesado.ipynb
```

El orden importa: el notebook 02 genera todos los ficheros de `data/processed/` y `models/preprocesadores/`. La partición usa una semilla fija (`SEMILLA = 42`), así que el resultado es reproducible.

## Uso del preprocesador

Los preprocesadores son `Pipeline` de sklearn que reciben **datos crudos** (tal y como vienen en el CSV o en una petición de la API) y devuelven la matriz lista para el modelo:

```python
import joblib
from src import preprocesamiento as pp

pre = joblib.load("models/preprocesadores/clasificacion.joblib")
X = pre.transform(pp.cargar_datos().drop(columns="diabetes"))
```

Necesitan que `src` sea importable, porque el primer paso del pipeline llama a `pp.preparar`.

## Registro de decisiones

| # | Hallazgo (EDA) | Decisión |
|---|---|---|
| 1 | 7 columnas PII ofuscadas (`dni`, `nombre`, …) | Eliminar: sin valor predictivo y sujetas a protección de datos |
| 2 | `usuario_alta_datos` independiente de ambos objetivos (χ² p = 0,48; V ≈ 0) | Eliminar: columna de gestión sin señal ni riesgo de fuga |
| 3 | `id_paciente_proy` | Índice, nunca feature |
| 4 | 4 binarias con hasta 8 codificaciones (`si/SI/True/1.0`…), ~25 % de registros | Unificar a 0/1; un valor irreconocible pasa a nulo (la API nunca falla) |
| 5 | Ordinales con gradientes monótonos con los objetivos | Codificación ordinal (enteros) |
| 6 | 8 % de filas con nulos: bloque de hábitos (376) + patrón anidado de abandono (267 ⊂ 397 ⊂ 657); compatibles con MCAR | Imputar mediana/moda dentro del pipeline (ajustado solo en train) + indicadores `falta_habitos`, `falta_imc` |
| 7 | IMC asimétrico con outliers clínicamente reales (obesidad III) | Conservar; añadir `imc_categoria` (OMS) |
| 8 | Factores de riesgo CV muy asociados a la diabetes | Variable `n_factores_riesgo_cv` (V = 0,38 con diabetes) |
| 9 | Sin multicolinealidad entre los predictores originales (VIF ≤ 1,7) | Se mantienen todos; en modelos lineales, usar variable derivada **o** sus componentes |
| 10 | `diabetes` remuestreada al 33 % | Partición estratificada; probabilidades no poblacionales (limitación) |
| 11 | `dias_salud_fisica_mala`: 60 % ceros, 9 % en 30, sobredispersión ×18, *heaping* | Estratificar por tramos 0 / 1–29 / 30; métrica MAE; valorar modelos *hurdle*/Tweedie |
| 12 | Clustering: continuo dominado por "carga de enfermedad" y "edad cardiometabólica vs malestar" | 12 variables de salud y hábitos estandarizadas; sociodemográficas solo para describir |

## Limitaciones conocidas

- La prevalencia de diabetes (33 %) es artificial por el remuestreo: no admite lecturas epidemiológicas y las probabilidades del clasificador no son poblacionales.
- Todos los datos son autorreportados (sesgo de recuerdo, deseabilidad social, redondeo de días).
- Diseño transversal: las asociaciones no son causales (p. ej. alcohol elevado ↔ menos diabetes).
- Pendiente del contrato de la API: confirmar qué columnas llegan en cada petición y ajustar `pp.columnas_features` si hace falta.
