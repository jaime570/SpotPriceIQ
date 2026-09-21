# Módulo 9 — Problema de negocio, recursos cloud y proceso MLflow

**Proyecto:** SpotPriceIQ — Predicción del precio spot eléctrico español (OMIE)
**Equipo:** Jaime Llorca · Juan Prieto · José Menjívar
**Fecha:** septiembre de 2026

---

## Parte 1 — Definición del problema de negocio con visión de proceso (process mining)

*Process mining* mira un problema no como "un modelo que predice un número", sino como un **proceso** con actividades, actores, tiempos y un flujo que se puede observar, medir y mejorar. Aplicado a este proyecto, la pregunta no es solo "¿qué precio tendrá la luz mañana?", sino "¿cómo es hoy el proceso de decidir la oferta al mercado diario, dónde están sus cuellos de botella, y cómo lo transforma un sistema de predicción?".

### 1.1. El proceso de negocio: la puja al mercado diario

El caso de uso vive dentro de un proceso operativo diario y **cíclico** de una comercializadora o mesa de *trading* de energía. Cada día D-1, antes del cierre de la subasta de OMIE (~12:00), el equipo debe presentar sus ofertas para las 24 horas del día D+1. El proceso tiene un reloj estricto: la subasta cierra a una hora fija y no se puede llegar tarde.

**Proceso AS-IS (situación actual, sin sistema ML) — actividades del *event log*:**

| # | Actividad | Actor | Coste / cuello de botella |
|---|---|---|---|
| 1 | Recopilar datos de mercado, demanda, meteo | Analista | Manual, disperso en varias fuentes y hojas de cálculo |
| 2 | Estimar la previsión de precio D+1 | Analista sénior | Depende del criterio experto; poco trazable, difícil de auditar |
| 3 | Revisar y ajustar la previsión | Responsable de mesa | Subjetivo; el conocimiento vive "en la cabeza" de una persona |
| 4 | Construir la curva de oferta | *Trader* | Presión de tiempo antes del cierre |
| 5 | Enviar la oferta a OMIE | *Trader* | Ventana horaria rígida |
| 6 | (Día D) Observar el precio real y el resultado | Analista | La comparación previsto vs real rara vez se registra sistemáticamente |

**Cuellos de botella detectados con la mirada de proceso:**

- **Actividad 2 (estimación) es el cuello crítico:** lenta, dependiente de una persona, poco reproducible y sin registro de por qué se decidió cada número. Si esa persona falta, el proceso se degrada.
- **Falta de trazabilidad:** no hay un *event log* que permita analizar después *por qué* la previsión falló un día concreto.
- **El bucle de aprendizaje (actividad 6) está roto:** el previsto vs real no se captura de forma estructurada, así que el proceso no aprende de sus propios errores.
- **Reprocesos y variabilidad:** días de alta volatilidad (invierno, crisis de gas) disparan revisiones manuales y estrés temporal.

### 1.2. Proceso TO-BE (con el sistema SpotPriceIQ)

El sistema ML no elimina al humano: **automatiza y hace trazable la actividad 2** (la estimación), libera al analista para la supervisión y cierra el bucle de aprendizaje. El proceso rediseñado, con sus tiempos, es el que orquesta Prefect:

```
[13:00 D-1]  Ingesta automática (OMIE + ESIOS + AEMET)         → Prefect flow
     ↓
             Validación de contrato de datos (Pandera)         → gate de calidad
     ↓
             Construcción de features (sin leakage)            → feature_sets
     ↓
             Predicción D+1 vía API (modelo @champion)         → FastAPI /predict
     ↓
             Predicción + intervalos P10/P50/P90 al dashboard  → Streamlit
     ↓
[humano]     El trader supervisa, ajusta si procede, y puja
     ↓
[Día D]      Se registra precio real → previsto vs real        → monitorización
     ↓
             Detección de drift; si el error se degrada        → trigger de reentrenamiento
     ↓
             (vuelta al inicio: el proceso aprende)
```

**Mejoras de proceso, medibles:**

| Dimensión | AS-IS | TO-BE |
|---|---|---|
| Tiempo de la estimación (act. 2) | Horas de trabajo experto | Segundos (llamada a la API) |
| Trazabilidad | Nula (criterio en la cabeza) | Total (MLflow: qué modelo, qué datos, qué error esperado) |
| Reproducibilidad | Baja | Bit a bit (DVC + MLflow) |
| Bucle de aprendizaje | Roto | Cerrado (monitorización → reentrenamiento) |
| Dependencia de personas | Alta (riesgo de *bus factor*) | Baja (proceso codificado y versionado) |
| Explicabilidad de un fallo | "No sabemos por qué" | SHAP + *event log* de predicciones |

### 1.3. KPIs de proceso y de negocio

- **KPIs de proceso:** *lead time* del ciclo diario (de ingesta a predicción), tasa de ejecuciones sin intervención manual, % de días con el bucle previsto-vs-real cerrado, tiempo medio de detección de degradación del modelo.
- **KPIs de modelo:** MAE y RMSE (€/MWh) en producción, cobertura de los intervalos, frecuencia de *drift* detectado.
- **KPI de negocio (el que importa a dirección):** € de coste de error evitado al año (2,63 €/MWh × volumen → ~23 M€/año a escala de 1 GW; véase el caso de negocio del Módulo 2).

La visión de proceso conecta las tres capas: un proceso más rápido y trazable (proceso) produce mejores previsiones (modelo) que ahorran dinero (negocio).

---

## Parte 2 — Estimación de recursos cloud

Esta sección dimensiona qué recursos de nube necesitaría el sistema en producción y cuantifica su coste con precios representativos (AWS, región de Europa, bajo demanda, septiembre de 2026). **Las cifras deben confirmarse en la [AWS Pricing Calculator](https://calculator.aws/)**; el objetivo del ejercicio es el razonamiento de dimensionamiento, no el céntimo exacto.

### 2.1. Perfil de carga (el punto de partida honesto)

Antes de elegir servicios, hay que entender la carga real, y aquí el dato es revelador: **es un workload pequeño**. El histórico completo son ~30.500 filas horarias y la tabla de features ocupa ~3,6 MB. El entrenamiento de XGBoost dura minutos en una CPU normal. La inferencia es una predicción de 24 valores una vez al día. No es *big data*: es un problema de **automatización fiable y siempre disponible**, no de escala de cómputo. Esto dicta la arquitectura: conviene lo **serverless y con escalado a cero**, no un clúster encendido las 24 horas.

### 2.2. Mapeo arquitectura → servicios cloud

| Componente del proyecto | Servicio AWS | Dimensionamiento |
|---|---|---|
| Ingesta diaria (Prefect flow, ~13h) | **AWS Fargate** (tarea programada por EventBridge) o **Lambda** | 0,5 vCPU / 1 GB, ~5 min/día |
| Entrenamiento periódico (XGBoost) | **Fargate** tarea puntual (ej. semanal) | 1 vCPU / 4 GB, ~15 min/semana |
| Reentrenamiento LSTM (futuro, GPU) | **EC2 g4dn.xlarge** puntual / SageMaker | Bajo demanda, apagado tras uso |
| Serving (API FastAPI) | **Fargate** servicio always-on | 0,25 vCPU / 0,5 GB, 1 tarea |
| MLflow (tracking + registry) | **Fargate** + **RDS PostgreSQL** | db.t4g.micro + backend | 
| Almacén de artefactos (modelos, datos) | **S3** | ~1–5 GB |
| Base de datos MLflow / metadatos | **RDS PostgreSQL** db.t4g.micro | 20 GB |
| Dashboard (Streamlit) | **Fargate** servicio | 0,25 vCPU / 0,5 GB |
| Entrada de tráfico | **Application Load Balancer** | 1 ALB compartido |
| Secretos (tokens ESIOS/AEMET) | **Secrets Manager** | 2 secretos |
| Programación de flows | **EventBridge Scheduler** | gratis a este volumen |

### 2.3. Estimación de coste mensual (AWS Pricing Calculator)

Estimación montada en la **[AWS Pricing Calculator](https://calculator.aws/#/estimate?id=9b088a8e3914288b90a624625674d0213bcc2f2a)**, región **Europe (Ireland) / eu-west-1**, modelo **bajo demanda (On-Demand)**. Coste inicial 0 USD; total 12 meses ≈ 882 USD.

| Servicio | Configuración | Coste/mes (USD) |
|---|---|---|
| Fargate — API + dashboard (2 tareas always-on) | 0,25 vCPU + 0,5 GB × 730 h | 18,02 |
| Fargate — servidor MLflow (always-on) | 0,5 vCPU + 1 GB × 730 h | 18,02 |
| Fargate — ingesta diaria | 0,5 vCPU + 1 GB × 5 min/día | 0,07 |
| Fargate — entrenamiento (semanal) | 1 vCPU + 4 GB × 15 min × 4/mes | 0,06 |
| RDS PostgreSQL | db.t4g.micro + 20 GB gp3, Single-AZ | 16,85 |
| Amazon S3 | 5 GB Standard | 0,12 |
| Elastic Load Balancing (ALB) | 1 Application Load Balancer | 19,57 |
| AWS Secrets Manager | 2 secretos (tokens ESIOS/AEMET) | 0,82 |
| EventBridge Scheduler | disparo de flows | 0,00 (Free Tier) |
| **Total (arquitectura always-on)** | | **73,53 USD/mes (≈ 68 €)** |

Nota de dimensionamiento: en el RDS se desactivaron *RDS Proxy*, *Database Insights* y *Extended Support* por no ser necesarios a esta escala; con ellos activados el servicio se disparaba a ~40 USD/mes. Es un buen ejemplo de que el coste en cloud lo determina la configuración fina, no solo el tamaño de la instancia.

### 2.4. Palancas de optimización

El coste está dominado por lo **siempre encendido** (API, dashboard, MLflow, ALB, RDS), no por el cómputo de ML. De ahí las palancas:

- **Escalado a cero / serverless:** si el dashboard y MLflow no necesitan estar 24/7, moverlos a servicios que se apagan solos (o levantarlos bajo demanda) recorta la mayor parte del gasto.
- **Fargate Spot** para las tareas puntuales de entrenamiento (~70 % más barato); son reintentables, así que la interrupción no es crítica.
- **Sustituir RDS por el fichero SQLite** que ya usa el proyecto (`mlflow.db`) si no hay concurrencia real: elimina ~15 €/mes. RDS solo se justifica con varios usuarios concurrentes.
- **Prescindir del ALB** exponiendo el servicio directamente o vía API Gateway con Lambda para cargas tan bajas.
- **Free Tier / créditos** para la fase de portfolio/TFM.

**Conclusión de dimensionamiento:** el coste se concentra en cuatro partidas always-on (los dos servicios Fargate always-on, el ALB y el RDS suman ~72 de los 73,53 USD). Una versión ajustada que **prescinda del ALB** (~19,57 USD) y **sustituya RDS por el SQLite** que ya usa el proyecto (~16,85 USD) baja el total al entorno de **~37 USD/mes**, e incluso a casi cero en fase de aprendizaje con Free Tier. El mensaje de ingeniería es que el sistema **no necesita** infraestructura pesada: el valor está en la fiabilidad del proceso, no en el músculo de cómputo.

---

## Parte 3 — Proceso MLflow sobre el proyecto del Módulo 2

MLflow ya está montado e integrado en el proyecto (implementado en el Bloque 7 del proyecto). Esta parte documenta el proceso: qué resuelve, cómo está estructurado y cómo se conecta con el servicio.

### 3.1. Qué resuelve MLflow y sus dos mitades

MLflow convierte "entrené un modelo y salió un número" en un registro **trazable, reproducible y comparable**. Tiene dos mitades:

- **Tracking** — un cuaderno de laboratorio automático. Cada ejecución (`start_run`) es una foto de un experimento con tres cosas: *params* (la receta / hiperparámetros), *metrics* (el resultado) y *artifacts* (el modelo entrenado).
- **Model Registry** — un catálogo de los modelos "buenos". Toma un modelo de una ejecución y le da un nombre estable (`spotprice-xgboost`), una versión (v1, v2…) y un alias móvil (`@champion`). El servicio no abre un `.pkl`: pide `spotprice-xgboost@champion`, desacoplando el servicio del entrenamiento.

### 3.2. Decisiones de infraestructura

- **Backend SQLite directo, sin servidor HTTP para loguear.** El código escribe directo a `sqlite:///mlflow.db` en la raíz del proyecto (ruta absoluta, para no depender del directorio de trabajo de los notebooks). El registry exige un backend de base de datos, por eso SQLite y no el *file-store* por defecto. Para visualizar, `python -m mlflow ui`. En Docker (contenerización) se monta ya un servidor MLflow real (`docker-compose.yml`).
- **Invocación por módulo** (`python -m mlflow …`) por la restricción de "Control de aplicaciones" de Windows.
- **Desactivar la inferencia de entorno en `log_model`** pasando `pip_requirements=["xgboost","scikit-learn"]` explícito: la inferencia por defecto lanza un subproceso pip que en este entorno se cuelga. Medido: con inferencia ~3,9 s; sin ella ~0,5 s.

### 3.3. Estructura de ejecuciones (runs)

Tres ejecuciones bien diferenciadas por el tag `validacion`, para no confundir nunca el número optimista con el honesto:

| Run (tag `validacion`) | Qué es | MAE | Uso |
|---|---|---|---|
| `holdout_6m` | Entrenamiento único sobre los últimos 6 meses | 13,6 | Número **optimista** |
| `walk_forward_expanding` | 30 reentrenamientos mes a mes (curva por `step`) + resumen `MAE_wf_mean`/`MAE_wf_std` | 15,70 ± 5,29 | Número **honesto** |
| `produccion_full_data` | Modelo entrenado con TODO el histórico | — (referencia `MAE_wf_ref`=15,70) | El que va a **producción** |

Decisión clave: se **valida** con walk-forward pero se **sirve** un modelo entrenado con todos los datos (al predecir mañana usarías todo el histórico hasta hoy, no una versión recortada). Ese modelo de producción no tiene MAE propio (evaluarlo sobre sus datos de entrenamiento sería *leakage*), así que se le adjunta como ficha técnica el MAE del walk-forward (`MAE_wf_ref` = 15,70), que además sirve de línea base para la monitorización de *drift*.

### 3.4. Registry: versión vs alias

La versión es una foto inmutable que sube sola (v1, v2…). El alias es una etiqueta móvil pegada a UNA versión. Registrado **`spotprice-xgboost` v1 con alias `@champion`**. Cuando se reentrene (monitorización/drift), se registrará la v2 y se moverá el alias; el servicio sirve la nueva versión **sin tocar código**.

### 3.5. Integración con el servicio (el pago del registry)

La API FastAPI carga el modelo una sola vez al arrancar (en el handler `lifespan`) con:

```python
mlflow.pyfunc.load_model("models:/spotprice-xgboost@champion")
```

Pide el modelo por alias, sin saber que es la v1 ni dónde está el fichero. `pyfunc` es la interfaz genérica (el mismo `.predict()` sirve para XGBoost o para el futuro LSTM), lo que es clave para el *gating* por régimen previsto. El endpoint `/model-info` expone nombre, versión resuelta desde `@champion`, `run_id` y `MAE_wf_ref`: la API dice **qué** sirve y **cómo de bueno** se espera que sea.

### 3.6. Complemento con DVC

MLflow y DVC se complementan: MLflow dice "usé el dataset X"; **DVC** garantiza que ese X es reproducible bit a bit. Versionado `data/processed/tabla_features.parquet` con DVC. Juntos cierran la reproducibilidad de extremo a extremo: mismo código (Git) + mismo dato (DVC) + mismo experimento (MLflow).
