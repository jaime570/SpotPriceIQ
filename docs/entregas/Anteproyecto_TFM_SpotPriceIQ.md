# Anteproyecto de Trabajo Fin de Máster

**Título:** SpotPriceIQ — Sistema end-to-end de predicción probabilística del precio spot del mercado eléctrico español (OMIE) con MLOps de producción

**Universidad:** IMF Smart Education — Máster en Data Science
**Autores:** Jaime Llorca Martínez · Juan Prieto · José Menjívar
**Tutor:** Juan Manuel Moreno

---

## 1. Resumen

Este Trabajo Fin de Máster propone el diseño, la construcción y la puesta en producción de un sistema completo de ciencia de datos para **predecir el precio marginal horario del mercado diario eléctrico español (OMIE)** para el día siguiente. El proyecto no se limita a entrenar un modelo predictivo: aborda el ciclo de vida completo de un sistema de *machine learning* de producción (MLOps) —ingesta automatizada de datos públicos, validación con contratos de datos, *feature engineering* sin fuga de información, evaluación rigurosa con *backtesting*, seguimiento de experimentos, servicio como API, orquestación, monitorización de deriva y contenerización—, replicando las prácticas que una empresa del sector energético espera de un sistema real.

Se dispone ya de resultados preliminares que validan la viabilidad: un modelo XGBoost que mejora en ~14 % el mejor baseline (MAE 15,70 frente a 18,33 €/MWh en *walk-forward*), un servicio de predicción operativo y una estimación de infraestructura en la nube de ~73 USD/mes. El TFM completará el sistema con las capas de orquestación, monitorización adaptativa al régimen de mercado y una capa de decisión económica.

## 2. Introducción y motivación

El mercado eléctrico mayorista español funciona como una subasta diaria: cada día, antes del mediodía, los agentes (comercializadoras, generadores, grandes consumidores y *traders*) presentan sus ofertas de compra y venta para las 24 horas del día siguiente **sin conocer el precio que resultará de la casación**. Anticipar ese precio es una necesidad operativa directa, y su relevancia económica es enorme: cada €/MWh de error de previsión evitado equivale, a escala de una gran cartera (~1 GW), a **millones de euros al año**.

La motivación del proyecto es doble. En lo técnico, el precio eléctrico es un problema de *forecasting* especialmente rico: series temporales multivariante, fuerte estacionalidad, no-estacionariedad marcada (la transición del régimen de crisis del gas de 2022-2023 al régimen renovable de 2025-2026), valores negativos y eventos extremos. En lo profesional, es un caso de uso de alto valor para el sector energético español (Repsol, Iberdrola, comercializadoras y mesas de *trading*), lo que convierte el proyecto en una demostración de competencias directamente empleable.

## 3. Antecedentes y estado del arte

La predicción de precios eléctricos (*Electricity Price Forecasting*, EPF) es un campo consolidado. Las familias de métodos habituales abarcan modelos estadísticos clásicos (ARIMA, modelos de regresión con variables exógenas), modelos de *machine learning* tabular (gradient boosting, *random forests*) y modelos de aprendizaje profundo para secuencias (LSTM, redes convolucionales temporales, *transformers*). La literatura reciente subraya tres tendencias relevantes para este trabajo: la superioridad frecuente de los modelos de *gradient boosting* sobre el *deep learning* en horizontes horarios con buen *feature engineering*; la importancia del *forecasting* probabilístico (intervalos y cuantiles) frente a la predicción puntual, especialmente para decisiones de riesgo; y la necesidad de una evaluación temporal correcta (*backtesting walk-forward*) que evite la fuga de información.

El diferencial de esta propuesta frente a un trabajo puramente académico es el **enfoque MLOps de producción**: no solo qué modelo predice mejor, sino cómo se ingiere el dato de forma fiable, cómo se garantiza su calidad, cómo se versiona el experimento, cómo se sirve el modelo y cómo se detecta y corrige su degradación en el tiempo.

## 4. Objetivos

**Objetivo general:** construir un sistema end-to-end, reproducible y desplegable, que prediga el precio horario del mercado diario español para D+1 usando únicamente información disponible en el momento de la subasta, y que traduzca esa predicción en una métrica de valor económico.

**Objetivos específicos:**

1. Automatizar la ingesta de las fuentes públicas del mercado (OMIE, ESIOS/REE, AEMET) con clientes robustos frente a límites de tasa y formatos reales.
2. Garantizar la calidad del dato mediante contratos de datos (Pandera), gestión de huecos horarios y del cambio de hora (días de 23 h y 25 h), y tests automáticos.
3. Diseñar un conjunto de variables predictivas sin fuga de información temporal, basado en las previsiones publicadas a D-1 y en variables de calendario.
4. Establecer un marco de evaluación honesto con *baselines*, *backtesting walk-forward* y una métrica económica en €/año.
5. Entrenar y comparar modelos (XGBoost, LSTM) y producir predicción probabilística por cuantiles con calibración.
6. Implantar prácticas MLOps: seguimiento de experimentos y registro de modelos (MLflow), versionado de datos (DVC), servicio como API (FastAPI), orquestación (Prefect), monitorización de deriva (Evidently) y contenerización (Docker, CI/CD).
7. Comunicar los resultados mediante un cuadro de mando y una documentación orientada a negocio.

## 5. Metodología y arquitectura del sistema

El proyecto sigue una metodología de **aprender construyendo** y de **corte vertical fino**: primero un pipeline mínimo que funcione de punta a punta y, sobre él, profundización por retorno. La arquitectura de datos sigue un patrón *medallion* (bronze → silver → gold), y la del sistema separa exploración (notebooks) de código de producción (paquete `src/`). El flujo, de dato crudo a decisión:

```
[Fuentes]  →  [Ingesta]  →  [Validación]  →  [Features]  →  [Baselines + Eval]
 OMIE/ESIOS     Parquet      contratos        lags/cal        naive, WF, métrica €
 AEMET                       Pandera, DST
                                                 │
                                                 ▼
     [Modelado]  →  [Tracking + Versionado]  →  [Serving]  →  [Orquestación]
     XGB / LSTM        MLflow + DVC              FastAPI       Prefect
     + intervalos
                                                 │
                                                 ▼
          [Monitorización]  →  [Dashboard]  →  [Infra + CI/CD]  →  [Docs]
           Evidently / drift     Streamlit      Docker + GHA       MkDocs
```

**Principio metodológico central (anti-leakage):** cada predicción usa exclusivamente la información publicada en el momento de la subasta. El modelo se apoya en las **previsiones D-1** de demanda y generación, no en los valores reales *ex-post*. Esta decisión, además de metodológicamente correcta, se ha validado empíricamente (ver §7).

## 6. Datos y fuentes

Todas las fuentes son públicas y gratuitas, lo que hace el trabajo íntegramente reproducible:

| Fuente | Contenido | Papel |
|---|---|---|
| OMIE | Precio marginal horario del mercado diario | Variable objetivo |
| ESIOS / REE | Demanda y generación por tecnología (eólica, solar, ciclo combinado, nuclear, hidráulica), intercambios | Fundamentales (previstos) |
| AEMET | Meteorología de estaciones cercanas a parques eólicos y solares | Variables meteorológicas |

El histórico abarca ~3 años (~30.500 registros horarios). Se conservan y validan por código los precios cero y negativos, que no son errores sino precio marginal nulo por exceso de renovable. Como limitación conocida, el precio del gas (MIBGAS/TTF) queda documentado como extensión futura.

## 7. Resultados preliminares (validación de viabilidad)

El estado actual del prototipo ya demuestra la viabilidad del sistema:

- **Baseline a batir:** *naive* D-1, MAE *walk-forward* de 18,33 €/MWh.
- **Modelo campeón (XGBoost predictivo):** MAE *walk-forward* de **15,70 €/MWh**, ~14 % mejor que el baseline, con la mejora concentrada —de forma coherente con el dominio— en los meses de mayor volatilidad.
- **Hallazgo de dominio:** el modelo con previsiones (ex-ante) supera al modelo con valores reales (ex-post), porque el precio se casa en subasta con las previsiones disponibles entonces. Entender la formación del precio pesa más que acumular datos.
- **Predicción probabilística:** cuantiles P10/P50/P90 con recalibración conformal; se identifica la no-estacionariedad como reto central (la frecuencia de precios negativos pasa del 1,5 % histórico al 11,3 % en 2026).
- **Interpretabilidad (SHAP):** el modelo razona conforme al *merit order* del mercado (la demanda sube el precio; las renovables lo bajan).
- **Caso de negocio:** la mejora de 2,63 €/MWh sobre el baseline equivale a ~23 M€/año a escala de 1 GW.

## 8. Plan de trabajo y cronograma

El trabajo se organiza en fases incrementales; cada fase produce código documentado y una entrada en el diario de aprendizaje. Las fases marcadas ✓ están validadas en el prototipo; el resto constituye el trabajo planificado del TFM.

| Fase | Contenido | Estado |
|---|---|---|
| 0 | Fundaciones: estructura, entorno, control de versiones (Git/DVC) | ✓ |
| 1 | Ingesta OMIE / ESIOS / AEMET | ✓ |
| 2 | Validación y calidad de datos (Pandera, DST, huecos) | ✓ |
| 3 | *Feature engineering* sin *leakage* | ✓ |
| 4 | *Baselines* y marco de evaluación *walk-forward* + métrica € | ✓ |
| 5 | Modelado (XGBoost, LSTM) e intervalos probabilísticos | ✓ |
| 6 | Interpretabilidad (SHAP) | ✓ |
| 7 | *Tracking* y versionado (MLflow + DVC) y serving (FastAPI) | ✓ |
| 8 | Orquestación (Prefect): ingesta diaria y reentrenamiento | En curso |
| 9 | Monitorización de deriva (Evidently) y adaptación al régimen | Planificado |
| 10 | Cuadro de mando (Streamlit) | Planificado |
| 11 | Infraestructura y CI/CD (Docker, GitHub Actions) | Planificado |
| 12 | Capa de decisión económica (señal y evaluación de valor) | Planificado |
| 13 | Documentación y *storytelling* (MkDocs, caso de negocio) | Planificado |

Distribución temporal orientativa: fundaciones y datos sólidos en el primer tercio; modelado, MLOps y serving en el segundo; monitorización, decisión económica, documentación y memoria en el último.

## 9. Recursos necesarios

**Datos:** APIs públicas de OMIE, ESIOS (token gratuito) y AEMET (token gratuito). **Almacenamiento y base de datos:** el histórico se persiste en ficheros **Parquet** particionados (capa *gold* del *data lake*), versionados con **DVC**; el seguimiento de experimentos y el registro de modelos de **MLflow** se apoyan en una **base de datos relacional** —**SQLite** (`mlflow.db`) en desarrollo y **PostgreSQL** en el despliegue en la nube— que almacena parámetros, métricas y metadatos de cada ejecución y versión de modelo. **Software:** íntegramente de código abierto (Python, pandas, XGBoost, scikit-learn, MLflow, FastAPI, Prefect, Evidently, Streamlit, Docker). **Cómputo:** el entrenamiento de los modelos tabulares es viable en CPU; el reentrenamiento de la LSTM se ejecuta puntualmente en GPU (Google Colab). **Infraestructura de despliegue (opcional):** una estimación en la nube (AWS, región de Irlanda, bajo demanda) del sistema completo en producción arroja **~73 USD/mes**, reducible a ~37 USD/mes en una configuración ajustada. El coste no es un obstáculo: el volumen de datos es pequeño y el valor del sistema reside en su fiabilidad, no en el músculo de cómputo.

## 10. Resultados esperados y evaluación

Se espera entregar un sistema funcional, reproducible y documentado que: (1) prediga el precio D+1 con un error inferior al baseline de forma robusta en *backtesting walk-forward*; (2) acompañe cada predicción de un intervalo de incertidumbre calibrado; (3) se ejecute de forma automatizada y se supervise a sí mismo detectando deriva; y (4) traduzca su rendimiento a un caso de negocio cuantificado. La evaluación se realizará con MAE y RMSE (€/MWh), cobertura de los intervalos, métricas de deriva en producción y la métrica económica de coste de error evitado.

## 11. Viabilidad, riesgos y limitaciones

La viabilidad está respaldada por el prototipo ya funcional. Los riesgos principales son la **no-estacionariedad del mercado** (mitigada con monitorización de deriva y reentrenamiento adaptativo), la **dependencia de APIs externas** (mitigada con clientes robustos y almacenamiento local del histórico) y la **ausencia del precio del gas** en la versión actual (documentada como limitación y planificada como extensión). Ninguno compromete los objetivos del TFM.

## 12. Aplicabilidad profesional

El proyecto reproduce, sobre un caso real del sector energético español, el ciclo completo que una empresa espera de un sistema de *machine learning* en producción. Su enfoque —datos reales, validación rigurosa, MLOps y traducción a valor de negocio— lo convierte en una demostración de competencias directamente alineada con perfiles de *data scientist* y *ML engineer* en el sector energético.

## 13. Referencias

- OMIE — Operador del Mercado Ibérico de Energía. Documentación del mercado diario.
- Red Eléctrica de España — ESIOS, sistema de información del operador del sistema.
- AEMET — Agencia Estatal de Meteorología, servicio de datos abiertos (OpenData).
- Weron, R. (2014). *Electricity price forecasting: A review of the state-of-the-art*. International Journal of Forecasting.
- Lago, J., De Ridder, F., De Schutter, B. (2018/2021). *Forecasting spot electricity prices: Deep learning approaches and empirical comparison*. Applied Energy.
- Lundberg, S. & Lee, S. (2017). *A unified approach to interpreting model predictions (SHAP)*. NeurIPS.
- Documentación de MLflow, DVC, Prefect, Evidently y FastAPI.
