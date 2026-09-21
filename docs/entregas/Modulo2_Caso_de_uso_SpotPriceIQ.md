# Módulo 2 — Entorno de trabajo y caso de uso de ciencia de datos

**Proyecto:** SpotPriceIQ — Predicción del precio spot eléctrico español (OMIE)
**Equipo:** Jaime Llorca · Juan Prieto · José Menjívar
**Fecha:** septiembre de 2026

---

## Parte 1 — GitHub, estructura de carpetas, entorno de desarrollo y control de versiones

Esta primera parte documenta el esqueleto reproducible del proyecto: cómo está organizado el repositorio, cómo se gestiona el entorno de ejecución y qué se versiona y cómo. El principio rector es que **el proyecto se regenera ejecutando código**: no se versionan los datos ni los modelos pesados, sino el código que los produce y la configuración que lo hace reproducible.

### 1.1. Repositorio y control de versiones

El código vive en un repositorio Git (`prediccion-electrica`) alojado en GitHub. La estrategia de versionado combina tres herramientas, cada una para lo que hace bien:

| Qué se versiona | Herramienta | Dónde vive |
|---|---|---|
| Código fuente, notebooks, configuración, documentación | **Git** | GitHub |
| Datasets y modelos (binarios grandes) | **DVC** (Data Version Control) | `.dvc/cache` + puntero `.dvc` en Git |
| Experimentos y modelos entrenados | **MLflow** (tracking + registry) | `mlflow.db` (SQLite) |

Git es ineficiente con binarios grandes, así que los datasets no entran en el repositorio: DVC guarda el fichero fuera de Git y deja únicamente un puntero de texto (`.dvc`) con el hash MD5. Así el repositorio se mantiene ligero y el dataset sigue siendo reproducible bit a bit (por ejemplo, `data/processed/tabla_features.parquet`, versionada con DVC). Esta es la respuesta al punto opcional del módulo sobre control de versiones de la documentación: la documentación (`docs/diario.md`, README, este documento) se versiona en Git como texto plano, mientras que el dato pesado se delega a DVC.

Ficheros de configuración de repositorio ya presentes: `.gitignore` (excluye `.env` con los tokens, `mlflow.db`, `mlruns/`, datos y cachés), `.dvcignore`, `.dockerignore`.

### 1.2. Estructura de carpetas

El proyecto sigue una **arquitectura medallion** para los datos (bronze → silver → gold) y una separación clara entre exploración (`notebooks/`) y código de producción (`src/`):

```
prediccion-electrica/
├── data/
│   ├── raw/            Bronze: datos crudos de las APIs (no versionado)
│   ├── interim/        Silver: integrado y estructurado
│   └── processed/      Gold: tabla limpia, tipada y validada (Pandera)
├── notebooks/          Exploración e iteración (01_ingesta … 09_interpretabilidad)
├── src/                Código de producción
│   ├── ingesta/        Clientes OMIE, ESIOS/REE, AEMET
│   ├── procesamiento/  Integración y concatenación de fuentes
│   ├── validation/     Contrato de datos Pandera (control_datos.py)
│   ├── features/       Feature engineering (feature_sets.py)
│   ├── evaluacion/     Métricas y backtesting walk-forward
│   ├── api/            Servicio FastAPI (main.py, schemas.py)
│   └── orquestacion/   Flows Prefect (ingesta diaria, reentrenamiento)
├── tests/              Tests unitarios (pytest)
├── docs/               Documentación y diario de aprendizaje
├── model_champion/     Modelo campeón exportado (formato MLflow)
├── .github/workflows/  CI/CD (ci.yml)
├── docker-compose.yml  API + servidor MLflow
├── dockerfile
├── requirements.txt
└── mlflow.db           Backend de tracking + registry (SQLite)
```

### 1.3. Entorno de desarrollo

El entorno se fija mediante `requirements.txt` con versiones ancladas (*pinned*), lo que garantiza que cualquiera que clone el repositorio reconstruya el mismo entorno. Dependencias principales:

```
fastapi==0.136.3      uvicorn==0.49.0     mlflow==3.14.0
xgboost==3.2.0        scikit-learn==1.8.0  pandas==2.3.3      pyarrow==22.0.0
```

Los secretos (tokens de las APIs de ESIOS y AEMET) se gestionan fuera del código, en un fichero `.env` excluido de Git. La contenerización con **Docker** (`dockerfile` + `docker-compose.yml`) empaqueta el servicio de API y el servidor de MLflow en imágenes reproducibles, y **GitHub Actions** (`.github/workflows/ci.yml`) ejecuta lint y tests en cada push como red de seguridad de integración continua.

**Nota de entorno real (documentada como aprendizaje):** en la máquina de desarrollo, la directiva "Control de aplicaciones" de Windows bloquea los lanzadores `.exe` que crea pip (`mlflow.exe`, `dvc.exe`, `uvicorn.exe`). La regla adoptada es invocar siempre por módulo (`python -m mlflow …`, `python -m uvicorn …`), que entra por el intérprete de confianza `python.exe`. Es el tipo de fricción real de un entorno corporativo que conviene dejar registrada.

---

## Parte 2 — Definición del caso de uso de ciencia de datos del TFM

### 2.1. Contexto y problema de negocio

El mercado eléctrico mayorista español funciona como una **subasta diaria** (mercado *day-ahead* gestionado por OMIE). Cada día, antes del mediodía, los agentes (comercializadoras, generadores, grandes consumidores, *traders*) deben presentar sus ofertas de compra y venta para las 24 horas del día siguiente **sin conocer todavía el precio que resultará**. Ese precio marginal horario se fija por casación entre oferta y demanda y determina el coste (o el ingreso) de la energía para el día siguiente.

Quien participa en ese mercado necesita una **anticipación del precio del día siguiente** para tomar decisiones: cuánta energía comprar y a qué horas, cómo cubrir su cartera, cuándo desplazar consumo o producción. Un error de previsión no es un problema abstracto: se traduce directamente en dinero. A escala de una gran eléctrica, **cada €/MWh de error de previsión evitado equivale a millones de euros al año** (véase el caso de negocio, §2.5).

**El problema, formulado como caso de uso de ciencia de datos:** predecir el precio marginal horario del mercado diario español para el día D+1, usando exclusivamente la información disponible en el momento de la subasta (mediodía de D-1), de forma que la predicción sea desplegable en producción sin fuga de información temporal (*leakage*).

### 2.2. Tipo de problema y target

Es un problema de **regresión sobre series temporales multivariante**, con estructura de *forecasting* a horizonte D+1 y resolución horaria (24 valores por día).

- **Variable objetivo (target):** `precio_espana`, precio marginal horario del mercado diario de OMIE, en €/MWh.
- **Horizonte:** día siguiente completo (24 horas), previsto en el momento de la subasta.
- **Granularidad:** horaria.
- **Particularidad de dominio:** el target admite valores **cero y negativos** (890 horas negativas en el histórico), que no son errores sino precio marginal cero por exceso de generación renovable. Se conservan y se validan por código.

### 2.3. Datos y fuentes

Todas las fuentes son **públicas y gratuitas**, lo que hace el proyecto reproducible por cualquiera:

| Fuente | Contenido | Papel |
|---|---|---|
| **OMIE** | Precio marginal horario del mercado diario | Target |
| **ESIOS / REE** | Demanda y generación por tecnología (eólica, solar FV, termosolar, ciclo combinado, nuclear, hidráulica), intercambios | Fundamentales (previstos y reales) |
| **AEMET** | Meteorología (temperatura, viento, precipitación, insolación) de 16 estaciones cercanas a parques eólicos y solares | Meteorología |

**Decisión de dominio central (anti-leakage):** el modelo usa las versiones **previstas D-1** de demanda y generación (`_prevista`, `_d1`), no las reales *ex-post*. Esto no es solo higiene metodológica: se verificó empíricamente que un modelo con los valores **reales** rinde **peor** (MAE walk-forward 16,73) que el modelo con las **previsiones** (15,70). La razón es de dominio: el precio se casa en subasta *ex-ante* con las previsiones disponibles entonces; los valores reales arrastran desviaciones posteriores a la subasta que no formaron el precio y actúan como ruido. **Entender cómo se forma el precio importa más que acumular datos.**

El volumen es modesto (~30.500 filas horarias, ~3 años de histórico; tabla de features de ~3,6 MB), lo que es relevante para el dimensionamiento de infraestructura (Módulo 9).

### 2.4. Enfoque de modelado y evaluación

El proyecto sigue deliberadamente una **escalera de complejidad**, no un salto directo al modelo sofisticado:

1. **Baselines** (naive D-1, naive D-7, media móvil). El *naive D-1* (copiar el precio de ayer) es un baseline fuerte por la autocorrelación diaria del precio: **MAE walk-forward 18,33 €/MWh**. Es el número a batir.
2. **XGBoost predictivo** (modelo campeón actual): **MAE walk-forward 15,70 €/MWh** (~14 % mejor que el baseline), con hallazgo clave de que el error se concentra en los cambios de régimen de mercado.
3. **LSTM** (secuencias horarias, lookback 168 h): empate estadístico en media (16,59) pero mayor varianza → XGBoost gana como modelo único por estabilidad y determinismo.
4. **Forecast probabilístico** (cuantiles P10/P50/P90 + calibración conformal) y **clasificadores de eventos extremos** (P(precio<0), P(precio>150)).

**Evaluación honesta con backtesting walk-forward** (nunca validación cruzada aleatoria, que metería futuro en el entrenamiento). La regla de oro es innegociable: *cada predicción usa solo información publicada en ese momento*. Métricas: **MAE** y **RMSE** en €/MWh (el MAPE se descarta porque el 18 % de las horas tienen precio ≤ 1 €/MWh y lo hacen infinito), más una **métrica económica** que traduce el error a €/año.

### 2.5. Caso de negocio (el argumento de valor)

El puente entre el error del modelo y el negocio es directo:

```
coste del error (€/año) = MAE (€/MWh) × volumen (MWh/año)
```

Con `volumen = MW × 8.760 h`, el coste anual del error de predicción a distintas escalas:

| Modelo | MAE | 1 MW | 100 MW | 1 GW |
|---|---|---|---|---|
| naive D-1 (baseline) | 18,33 | 160.571 € | 16,06 M€ | 160,6 M€ |
| XGBoost (campeón) | 15,70 | 137.532 € | 13,75 M€ | 137,5 M€ |
| **Mejora del modelo** | **2,63** | **23.039 €** | **2,3 M€** | **23,0 M€/año** |

A escala de una gran eléctrica (~1 GW de cartera), la mejora de 2,63 €/MWh del modelo sobre el baseline representa del orden de **23 M€/año**. Esta es la tesis del proyecto: *reducir error de previsión = dinero*, cuantificado y honesto (aproximación lineal y simétrica; el coste asimétrico se aborda vía cuantiles/pinball loss).

### 2.6. Alcance, encaje empresarial y limitaciones

El proyecto está pensado como un **sistema end-to-end de MLOps de producción**, no como un notebook de curso: ingesta automatizada de fuentes reales, validación con contratos de datos, backtesting riguroso, tracking de experimentos, servicio como API, orquestación, monitorización de *drift* y contenerización. Este alcance es exactamente lo que valora una empresa del sector energético (Repsol, Iberdrola y comercializadoras/*traders*).

**Limitación conocida y documentada:** el precio del gas (MIBGAS/TTF) todavía no está en la ingesta. Como marca el techo del precio spot, su ausencia hace que el modelo dependa de la autorregresión para inferir el nivel de régimen. Está documentada como limitación de la v1 y planificada como trabajo futuro (variable + indicador líder de crisis).
