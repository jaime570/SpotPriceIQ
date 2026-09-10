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

Los datos **no se versionan**: se regeneran ejecutando el pipeline. Lo que se versiona es el código que los produce.

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
├── data/                      raw / interim / processed  (no versionado)
├── notebooks/                 exploración e iteración
│   ├── 01_ingesta_aemet.ipynb
│   ├── 01_ingesta_esios.ipynb
│   ├── 01_ingesta_omie.ipynb
│   ├── 02_concatenacion.ipynb
│   └── 03_etl_limpieza.ipynb
├── src/                       código fuente (producción)
│   ├── ingestion/             (futuro) ingesta migrada
│   ├── etl/                   (futuro) limpieza migrada
│   ├── validation/
│   │   └── control_datos.py   esquema Pandera del warehouse
│   ├── features/              (futuro) feature engineering
│   ├── models/                (futuro) entrenamiento LSTM / XGBoost
│   └── pipelines/             (futuro) orquestación Prefect
├── tests/                     (futuro) tests unitarios
├── api/                       (futuro) FastAPI
├── app/                       (futuro) dashboard Streamlit
├── docker/                    (futuro) Dockerfile
└── .github/workflows/         (futuro) CI/CD
```

## Stack

Python · pandas · Pandera · LSTM + XGBoost · MLflow · FastAPI · Prefect · Streamlit · Docker · GitHub Actions

## Estado actual

- ✅ Ingesta de las tres fuentes.
- ✅ Integración en tabla maestra horaria en UTC (resolución de DST incluida).
- ✅ ETL de limpieza + validación Pandera → `data/processed/tabla_maestra_procesada.parquet`.
- ⬜ Feature engineering y entrenamiento de los dos modelos.
- ⬜ Automatización semanal (re-ingesta incremental con *upsert* + Prefect) y despliegue.
