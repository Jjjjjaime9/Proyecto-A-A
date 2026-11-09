# Proyecto de Aprendizaje Automático: encuesta de salud y diabetes

Proyecto de la asignatura de Aprendizaje Automático (UIE). A partir de la encuesta anual de salud de un consorcio de atención primaria (`proy_diabetes_dev.csv`) se abordan tres líneas de trabajo:

| Línea | Tarea | Objetivo | Estado |
|---|---|---|---|
| Clusterización | Perfiles de paciente interpretables | — | Preprocesado |
| Clasificación | Cribado de diabetes | `diabetes` (sí/no) | **Modelada y servida en la API** |
| Regresión | Carga de enfermedad | `dias_salud_fisica_mala` (0–30) | Preprocesado |

## Estado

**Punto de control 1 (S09)**
- [x] Análisis exploratorio completo → `notebooks/01_eda.ipynb`
- [x] Código de preprocesado → `src/preprocesamiento.py`
- [x] Versión preprocesada de los datos → `data/processed/` y `models/preprocesadores/`

**Punto de control 2 (S12)**
- [x] Revisión pedida en el PC1: objetivos cruzados como predictores → descartados con evidencia (decisión #13)
- [x] Clasificación modelada → `notebooks/03_modelado_clasificacion.ipynb` y `src/modelado.py`
- [x] Artefacto final reproducible → `python src/entrenar.py clasificacion` → `artifacts/`
- [x] API común respondiendo para `clasificacion` → `src/servicio.py`, con 16 pruebas en `tests/`
- [x] ZIP de entrega con `data/processed/` y artefactos → `python src/empaquetar.py`
- [ ] Regresión y clusterización (siguiente fase)

## Resultados: clasificación (`diabetes`)

Modelo: **ensamble por voto suave** de una regresión logística con *splines* y un *gradient boosting* poco profundo. Umbral de decisión: **0,35**.

| | CV en train¹ | Test (n = 2 550) · IC 95 % | Dummy |
|---|---|---|---|
| ROC-AUC | 0,828 | **0,826** [0,809–0,842] | 0,500 |
| PR-AUC | 0,680 | 0,683 [0,646–0,718] | 0,333 |
| Brier | 0,157 | 0,158 [0,150–0,166] | 0,222 |
| Sensibilidad (umbral 0,35) | 0,773 | **0,759** [0,727–0,789] | 0 |
| Especificidad (umbral 0,35) | 0,729 | 0,726 [0,705–0,746] | 1 |

¹ ROC-AUC, PR-AUC y Brier: media de la CV repetida 5 × 5. Sensibilidad y especificidad: predicciones fuera de pliegue de la CV de 5 pliegues, con las que se eligió el umbral.

Detecta 644 de los 848 diabéticos de test. Uno de cada 1,7 pacientes citados es diabético, frente a uno de cada 3 si se citara al azar. Las variables más influyentes son salud percibida, IMC, edad, hipertensión y colesterol. Detalle completo y figuras 21–28 en el notebook 03.

## Estructura

```
Proyecto/
├── proy_diabetes_dev.csv           datos crudos (no se modifican)
├── requirements.txt
├── src/
│   ├── preprocesamiento.py         limpieza, variables derivadas, pipelines y particiones
│   ├── modelado.py                 catálogo de modelos, evaluación (CV, IC bootstrap) y CONFIG_FINAL
│   ├── entrenar.py                 reentrena la configuración final y guarda el artefacto de la API
│   ├── servicio.py                 API común (plantilla del profesor, sin modificar)
│   ├── empaquetar.py               genera el ZIP de entrega y comprueba que no falte nada
│   └── eda_utils.py                estilo gráfico, estadísticos y gráficos
├── notebooks/
│   ├── 01_eda.ipynb                EDA (resumen ejecutivo, 10 secciones, figuras 01–20)
│   ├── 02_preprocesado.ipynb       ejecución del preprocesado y comprobaciones
│   └── 03_modelado_clasificacion.ipynb   selección, ajuste, umbral, test e interpretación (figuras 21–28)
├── data/processed/                 datos limpios y particiones transformadas (generado por el notebook 02)
├── models/preprocesadores/         preprocesadores ajustados solo con train (notebook 02)
├── artifacts/
│   ├── proy_modelo_clasificacion.joblib   modelo que carga la API (datos crudos → predicción)
│   └── proy_modelo_clasificacion.json     metadatos: versión, columnas, configuración y métricas
├── reports/
│   ├── figures/                    figuras para informe y slides (PNG, 150 dpi)
│   └── metricas_clasificacion.json métricas de CV y test (las genera el notebook 03)
├── tests/test_servicio.py          pruebas del contrato de la API
└── ejemplos/peticion_clasificacion.json   petición de demostración (5 pacientes reales de test)
```

## Cómo reproducir

```bash
pip install -r requirements.txt

# 1. Análisis, preprocesado y modelado (en este orden)
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocesado.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_modelado_clasificacion.ipynb

# 2. Artefacto final que sirve la API (reentrena con los 12 750 pacientes)
python src/entrenar.py clasificacion

# 3. Pruebas del servicio
python -m pytest tests -q

# 4. ZIP de entrega (incluye data/processed, models y artifacts)
python src/empaquetar.py --nombre PC2
```

Todo usa una semilla fija (`SEMILLA = 42`), así que el resultado es reproducible. El notebook 03 termina con 10 comprobaciones (`assert`) sobre el modelo que se va a servir. Además verifica que `modelado.CONFIG_FINAL` coincide con lo que ha decidido, para que notebook y artefacto no se desincronicen.

## API

El servicio implementa el contrato común con la plantilla del profesor, sin cambios. Carga los artefactos de `artifacts/` y sirve las líneas cuyo artefacto existe; ahora mismo, `clasificacion`.

```bash
uvicorn src.servicio:app --host 0.0.0.0 --port 8000
```

| Endpoint | Respuesta |
|---|---|
| `GET /health` | `{"estado": "ok", "lineas": ["clasificacion"], "modelo_version": "1.0.0"}` |
| `POST /predict` | `{"linea", "modelo_version", "predicciones": [{"clase": 0/1, "probabilidad": p}, …]}`, una por instancia |
| `GET /docs` | Documentación interactiva (Swagger) para probarla desde el navegador |

Demostración con la petición de ejemplo:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data-binary @ejemplos/peticion_clasificacion.json
```

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/predict -Method Post -ContentType "application/json" -InFile ejemplos/peticion_clasificacion.json
```

**Qué acepta.** Instancias **crudas**, tal y como vienen en el CSV:
- Las columnas sobrantes se ignoran: PII, id, `usuario_alta_datos` y los objetivos.
- Valores en cualquiera de las codificaciones vistas (`si/SI/True/1.0`…) o no vistas (`Sí`, `YES`).
- Nulos y valores imposibles, que se tratan como ausentes y se imputan. **Nunca se descarta una instancia.**
- Solo devuelve 422 si una de las 18 columnas que usa el modelo no aparece en *ninguna* instancia, o si la línea no está servida.

**Detalle técnico.** Los artefactos referencian `preprocesamiento` y `modelado` como módulos de primer nivel. Es lo mismo que resuelve el servicio al añadir `src/` a `sys.path`, así que el artefacto carga desde cualquier directorio de trabajo; hay una prueba que lo verifica.

## Registro de decisiones

| # | Hallazgo | Decisión |
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
| 13 | **Revisión del PC1.** El objetivo de una línea como predictor de la otra no aporta nada: ΔROC-AUC = 0,0000 con `dias_salud_fisica_mala` en clasificación, ΔMAE = −0,002 días con `diabetes` en regresión (CV pareada) | **Descartados en ambas líneas.** En cribado el diagnóstico es justo lo desconocido, y exigirlos en la API arriesgaba un 422. Clustering los conserva (describen el perfil) |
| 14 | El modelo solo necesita 18 columnas crudas | Se ajusta con `pp.columnas_crudas`: `feature_names_in_` pide solo eso y el servicio descarta el resto |
| 15 | Ranking de pacientes para citar → la métrica relevante es de orden; la API devuelve probabilidad | Selección por **ROC-AUC**; calidad de la probabilidad con Brier/log-loss; exactitud nunca (el 67 % es trivial) |
| 16 | 7 familias en CV: todas las razonables entre 0,816 y 0,825, diferencias < 1 de entre pliegues | KNN y RF descartados; se ajustan logística + splines (rejilla) y boosting (búsqueda aleatoria) |
| 17 | En CV repetida 5 × 5, el ensamble supera a la logística simple en 24/25 particiones (+0,004 AUC). Regla fijada de antemano: ≥ 80 % | **Ensamble por voto suave** (logística + splines y boosting) |
| 18 | Probabilidades ya calibradas; la isotónica no mejora el Brier | No se recalibra |
| 19 | Con umbral 0,5 solo se detecta el 58 % de los diabéticos; máximo F1 y Youden coinciden en 0,35 | **Umbral 0,35** (sensibilidad 0,77), fijado dentro del pipeline con `FixedThresholdClassifier` |
| 20 | Configuración cerrada con CV y validada una vez en test | El artefacto final se reentrena con los 12 750 pacientes (train + test) |
| 21 | En el PC1 el ZIP llegó sin `data/processed/` ni artefactos | `src/empaquetar.py` los incluye explícitamente y falla si falta algún bloque |

## Limitaciones conocidas

- La prevalencia de diabetes (33 %) es artificial por el remuestreo: no admite lecturas epidemiológicas y las probabilidades del clasificador no son poblacionales. Para el uso real habría que corregir el *prior* y reajustar el umbral.
- Todos los datos son autorreportados (sesgo de recuerdo, deseabilidad social, redondeo de días). Los cuatro tipos de modelo se quedan en ROC-AUC ≈ 0,82–0,83: el techo parece estar en la información disponible, no en el algoritmo.
- Diseño transversal: las asociaciones no son causales (p. ej. la mala salud percibida es en parte consecuencia de la diabetes).
- El servicio sigue la plantilla del profesor (`servicio.py`). Falta confirmar con el enunciado ampliado (contrato publicado en S07) qué columnas incluye cada petición y la métrica de la fase competitiva. Si la métrica dependiera de la `clase`, el umbral se ajusta en `modelado.CONFIG_FINAL` sin tocar el modelo.
