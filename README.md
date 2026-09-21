# SpotPriceIQ — Predicción del precio spot eléctrico español

Proyecto de ciencia de datos y MLOps para **predecir el precio marginal horario del mercado eléctrico español (OMIE)** a partir de datos de mercado, meteorología y operación de la red. El objetivo es un pipeline reproducible y automatizado, de extremo a extremo, desde la ingesta de datos públicos hasta el servicio de predicciones.

## Fuentes de datos

Todas públicas y gratuitas:

- **OMIE** — precio marginal horario del mercado diario (el *target*).
- **AEMET** — variables meteorológicas (temperatura, viento, precipitación, insolación) de 16 estaciones seleccionadas por su cercanía a parques eólicos y solares.
- **ESIOS / REE** — variables de operación del sistema: demanda, generación por tecnología (eólica, solar FV, termosolar, ciclo combinado, nuclear, hidráulica) e intercambios internacionales, en versión *prevista* (conocida por adelantado) y *real*.

## Arquitectura de datos (medallion)

| Capa | Carpeta | Contenido |
|------|---------|-----------|
| Bronze (*data lake*) | `data/raw/` | Datos tal cual llegan de las APIs (JSON, ficheros OMIE). |
| Silver | `data/interim/` | Integrado y estructurado, pendiente de limpieza. |
| Gold (*warehouse*) | `data/processed/` | Tabla limpia, tipada y **validada**, lista para modelar. |

**Versionado**: el *código* se versiona en git; el *dataset de modelado* (`tabla_features.parquet`) se versiona con **DVC** para reproducir con qué datos exactos se entrenó cada modelo (ver [Versionado de datos](#versionado-de-datos-dvc)); los *datos crudos* no se versionan, se regeneran ejecutando el pipeline.

## Pipeline

```
1. Ingesta        →  OMIE + AEMET + ESIOS               →  data/raw/
2. Integración    →  aplanar, datetime_utc, unir 3 fuentes →  data/interim/
   + ETL limpieza →  centinelas, tipos, validación Pandera →  data/processed/
3. Modelado       →  feature engineering + LSTM y XGBoost (por separado)
4. MLOps          →  MLflow, FastAPI, Streamlit, Prefect, Docker, CI/CD
```

### Decisiones clave del ETL (capa `processed`)

- **Centinelas AEMET:** `Ip` (precipitación inapreciable) → `0`; `Acum` (acumulado con fecha ambigua) → `NaN`.
- **Tipos:** columnas climáticas de AEMET convertidas de texto (coma decimal) a `float`.
- **Ceros y negativos del precio:** se conservan. Se verificó por código que son precio marginal cero por exceso de renovable, no huecos enmascarados (890 negativos, concentración en horas solares, correlación con generación FV).
- **`precio_portugal`:** se conserva en el warehouse; se excluye como *feature* en modelado por riesgo de *leakage* (mercado MIBEL acoplado).
- **NaN de features:** se dejan sin imputar en esta capa; la imputación (determinista) es responsabilidad de la fase de *feature engineering*.
- **Validación:** contrato de datos con **Pandera** (`src/validation/control_datos.py`) como puerta de calidad previa al guardado.

## Estructura del proyecto

```
prediccion-electrica/
├── data/                   raw / interim / processed (crudos no versionados; tabla_features via DVC)
├── notebooks/              exploración e iteración (01–09)
├── src/                    código fuente (producción)
│   ├── ingesta/            clientes OMIE / ESIOS / AEMET
│   ├── procesamiento/      concatenacion.py · etl.py · features.py
│   ├── validation/         control_datos.py (contrato Pandera + contrato de cobertura)
│   ├── features/           feature_sets.py (selección de sets para modelado)
│   ├── evaluacion/         metricas.py · backtesting.py
│   ├── api/                FastAPI (main.py, schemas.py)
│   └── orquestacion/       Prefect: pipeline_diario.py · reentrenamiento.py · programar.py
├── tests/                  tests (pytest)
├── reports/                salidas de evaluación
├── docs/                   diario de aprendizaje (diario.md)
└── mlflow.db               tracking + registry de MLflow
```

## Stack

Python · pandas · Pandera · LSTM + XGBoost · MLflow · FastAPI · Prefect · Streamlit · Docker · GitHub Actions

## Servicios locales

Cómo levantar cada servicio y dónde queda su interfaz. Se usa `python -m ...` porque el App Control de Windows bloquea los *shims* `.exe`.

| Servicio | Comando | URL |
|----------|---------|-----|
| Prefect (servidor + UI) | `python -m prefect server start` | http://127.0.0.1:4200 |
| MLflow (tracking + registry) | `python -m mlflow ui` | http://127.0.0.1:5000 |
| FastAPI (API de predicción) | `python -m uvicorn src.api.main:app --reload` | http://127.0.0.1:8000/docs |
| Streamlit (dashboard, *pendiente*) | `python -m streamlit run app/dashboard.py` | http://localhost:8501 |

Son los puertos por defecto de cada herramienta; quedarán declarados en `docker-compose.yml` al contenerizar (Fase 11), que pasará a ser la fuente de verdad de puertos y URLs.

## Versionado de datos (DVC)

El código va en git, pero los `.parquet` grandes no. Para versionarlos sin hinchar el repo se usa **DVC**: git guarda un *pointer* ligero (`.dvc`, con el md5 del fichero) y DVC guarda el dato real en un *remote*.

Se versiona **`data/processed/tabla_features.parquet`** (la tabla que alimenta el entrenamiento). Versionarla ata cada modelo al dataset exacto con el que se entrenó — clave porque el pipeline **no es reproducible en el tiempo**: OMIE, ESIOS y AEMET cambian y añaden histórico, así que "regenerar desde código" no devuelve el dataset de hace meses; DVC sí.

Remote local (`dvc-storage`, fuera del repo). Flujo:

| Acción | Comando |
|--------|---------|
| Versionar / actualizar el dataset | `python -m dvc add data/processed/tabla_features.parquet` |
| Guardar el pointer en git | `git add ...tabla_features.parquet.dvc && git commit -m "..."` |
| Subir el dato al remote | `python -m dvc push` |
| Recuperar el dato (otra máquina / estado pasado) | `python -m dvc pull` |

Para volver a un estado pasado, `git checkout <commit>` + `python -m dvc checkout` recuperan código y dato sincronizados.

## Estado actual

- ✅ Ingesta de las tres fuentes (OMIE, ESIOS, AEMET), modularizada en `src/ingesta/`.
- ✅ Integración en tabla maestra horaria en UTC (DST resuelto) — `src/procesamiento/concatenacion.py`.
- ✅ ETL de limpieza + validación Pandera → `tabla_maestra_procesada.parquet`.
- ✅ Feature engineering (imputación en 3 capas, calendario, lags, marca de entrenables) → `tabla_features.parquet`.
- ✅ Modelado: XGBoost, LSTM, ensemble e intervalos de predicción; MLflow (tracking + registry).
- ✅ Serving con FastAPI; contenerización (Docker) y CI (GitHub Actions).
- ✅ Orquestación con Prefect: pipeline diario (ingesta → cobertura → concat → ETL → features) y reentrenamiento semanal, con límite de concurrencia = 1.
- ✅ Contrato de cobertura de fuentes crudas (suela de filas + frescura) como puerta del pipeline.
- ✅ Versionado del dataset de modelado con DVC.
- ⬜ Monitorización de drift (Evidently), dashboard (Streamlit), docs (MkDocs) y pulido del despliegue.
