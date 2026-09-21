# Diario de aprendizaje — SpotPriceIQ

Registro de decisiones técnicas y de dominio, fase a fase.

---

## 2026-08-30 — Cierre de `feature_sets.py`: lags de precio (arranque Fase 4)

**Qué se hizo**
Se incorporaron los tres lags de precio (`precio_lag_24`, `precio_lag_168`,
`precio_media_24`) a los dos feature sets (`predictivo` y `explicativo`) en
`src/features/feature_sets.py`, con una regla por prefijo:

```python
lags = df.columns[df.columns.str.startswith(("precio_lag", "precio_media"))].tolist()
```

**Por qué van en ambos sets (decisión de dominio)**
Los lags no son leakage: en D-1, tras la subasta de las ~13h, ya se conocen el
precio de ayer (`lag_24`) y el de hace una semana (`lag_168`). Es información
pasada, disponible en el momento de predecir D+1. Por eso son válidos incluso
para el modelo predictivo, no solo para el explicativo ex-post.

**Por qué `startswith` y no `contains`**
`contains("media")` habría capturado por error las 15 columnas `velmedia_*`
(viento AEMET); `contains("dia")` habría capturado `velmedia`, `medina_de_pomar`
y `demanda_prevista_diaria`. El anclaje al inicio (`startswith`) con la tupla
`("precio_lag", "precio_media")` aísla exactamente los 3 lags y no toca
`precio_espana` (target) ni `precio_portugal` (leakage MIBEL). Verificado.

**Aclaración de naming: `dia` vs `semana`**
No hay solapamiento (sospecha descartada con datos):
- `dia` = día del MES (1-31).
- `semana` = día de la SEMANA (0-6). Codificada como cíclica con periodo 7.
El naming es confuso pero son variables distintas; ambas pueden coexistir.

**Calendario: decisión crudas vs cíclicas (resuelta)**
Se añade el calendario a ambos sets usando SOLO la codificación cíclica
(`hora_sin/cos`, `semana_sin/cos`, `mes_sin/cos`) + `festivo`. Se descartan las
crudas (`hora`, `dia`, `semana`, `mes`) y también `dia` (día del mes, poca señal
para el precio spot y sin versión cíclica).

Regla por sufijo, coherente con el estilo del módulo:
```python
calendario = df.columns[df.columns.str.endswith(("_sin", "_cos"))].tolist()
if "festivo" in df.columns:
    calendario.append("festivo")
```

Motivo (set único compartido XGB + LSTM): las cíclicas son el denominador común
que funciona para ambas familias. La hora 23 y la 0 quedan adyacentes en el
círculo (sin+cos), evitando el salto artificial 23→0 que confunde a LSTM y
modelos lineales. Meter además las crudas duplicaría la señal: en XGBoost
diluye las importancias entre 3 columnas por variable; en lineales introduce
multicolinealidad. Se prioriza la fuente única de verdad sobre un set
diferenciado por modelo (más óptimo en teoría, peor de mantener).

Resultado: predictivo = 102 features, explicativo = 104.


---

## 2026-08-30 — Fase 4: split temporal + baselines evaluados

**Split temporal (holdout de 6 meses)**
Corte cronológico, nunca aleatorio (en series, el shuffle mete futuro en train).
- Fecha de corte: 2025-12-29 22:00 UTC (ultima_fecha − `pd.DateOffset(months=6)`).
- Train: 26.183 filas (2022-12-31 → 2025-12-29). Test: 4.369 filas (~ene–jun 2026), 14,3% del total.
- Test sin NaN en `precio_lag_24/168` ni en el target: las filas de arranque con lags vacíos caen todas en train.
- El test se trata como caja cerrada: no se mira ni se usa para decidir/tunear nada hasta la evaluación final. La misma partición sirve para baselines y modelos (comparación justa).

**Baselines evaluados sobre el test** (función `evaluar` → MAE, RMSE, MAPE, sMAPE):

| Baseline | MAE | RMSE | sMAPE | MAPE |
|---|---|---|---|---|
| naive_d1 (ayer) | **15,5** | 23,69 | 0,775 | inf |
| naive_d7 (hace 7d) | 25,8 | 37,74 | 0,979 | inf |
| naive_media (media 24) | 30,55 | 39,11 | 0,943 | inf |

**Interpretación / decisiones de dominio**
- `naive_d1` es el mejor con diferencia → fuerte autocorrelación diaria del precio spot. Es el **baseline a batir**: el objetivo del modelado es MAE(test) < 15,5 €/MWh.
- La media móvil es la peor en MAE: suaviza y se come los picos de un mercado volátil.
- **MAPE = inf**: 798/4.369 horas (18%) con precio ≤ 1 €/MWh, incluido −9,82 (precios negativos por exceso de renovables). El MAPE se descarta como métrica principal; se reportan **MAE y RMSE**. Caso borde real del mercado eléctrico español.
- RMSE ≫ MAE (ratio ~1,5) → errores grandes concentrados en los picos de precio.

**Métricas de referencia de la fase:** MAE (comunicación a negocio, €/MWh) y RMSE (sensibilidad a picos).


---

## 2026-08-30 — Fase 4.3: métrica económica (€/año)

**Qué se hizo**
Traducción del MAE a coste anual del error: `coste (€/año) = MAE (€/MWh) × volumen (MWh/año)`, con `volumen = MW × 8760 h`. Tres escenarios de volumen: 1 MW, 100 MW y 1 GW (escala de gran eléctrica).

**Coste anual del error de predicción (€/año):**

| Baseline | 1 MW | 100 MW | 1 GW |
|---|---|---|---|
| naive_d1 | 135.762 € | 13,58 M€ | 135,76 M€ |
| naive_d7 | 226.020 € | 22,60 M€ | 226,02 M€ |
| naive_media | 267.625 € | 26,76 M€ | 267,62 M€ |

**Mensaje de negocio**
Solo por elegir el naive_d1 frente a la media móvil, a escala de 1 GW se ahorran ~132 M€/año (267,62 − 135,76). Cada €/MWh de error reducido equivale a millones/año a escala utility. Es la demostración de la tesis "reducir error = dinero" antes incluso de entrenar un modelo.

**Supuestos y limitaciones (honestidad)**
- Aproximación **lineal y simétrica**: se asume que 1 € de error de más cuesta igual que 1 € de menos.
- Lo calculado es el **coste del error**, no el "ahorro del modelo": ese se obtendrá en la Fase 5 como `(15,5 − MAE_modelo) × volumen`, cuando el modelo baje del baseline.
- **Factor de captura** (qué fracción del error se materializa en coste) y **coste asimétrico** pospuestos a la Fase 5; la asimetría se abordará correctamente vía **pinball / quantile loss** (intervalos, Fase 5.3), no con costes inventados.


---

## 2026-08-30 — Fase 4.4: backtesting walk-forward (baseline naive D-1)

**Diseño**
Walk-forward mensual sobre todo el histórico. Cada fold = un mes de test; se reservan los primeros ~12 meses como arranque (`meses[12:]`). Los meses se etiquetan con `datetime_utc.dt.to_period("M")` (año+mes en un valor ordenable) y se itera por MES (no por fila). El train de cada fold no se usa en el baseline (el naive no entrena); la maquinaria queda montada para reutilizarla con el modelo en la Fase 5.

**Manejo de NaN**
`precio_lag_24` tiene 96 NaN: arranque de la serie + huecos internos ligados a DST/días incompletos (may-2023, oct/nov-2025). El target no tiene NaN. Las horas sin lag disponible se **excluyen** de la métrica con `dropna` dentro de cada fold (no se imputan, para no falsear la evaluación). `entrenable` no cubre estos NaN (no mira los lags).

**Resultado**
- MAE walk-forward = **18,33 ± 4,69** €/MWh (31 folds).
- El holdout único daba **15,5** → era **optimista**. La media de los 6 meses del holdout (ene–jun 2026) es ≈15,4, arrastrada por un feb-2026 anómalo (MAE 7,78, el mejor de los 31 meses) y varios meses de primavera. El walk-forward destapa el error real, más alto.

**Hallazgo de dominio: el error del naive es ESTACIONAL**
- Peores meses: otoño-invierno (oct–mar), MAE 22–25. Precio volátil (calefacción, gas, olas de frío, poca solar) → copiar "el precio de ayer" falla.
- Mejores meses: primavera (mar–may), MAE 8–15. Precio estable (mucha solar predecible) → el naive acierta.
- La desviación de 4,69 NO es ruido: es estacionalidad estructural. La fiabilidad de la predicción naive es peor justo en los meses críticos (invierno).

**Implicación para la Fase 5**
El margen de mejora está en el invierno (donde el naive es malo, 22–25); en primavera el naive ya es difícil de batir (7–8). El modelo debe evaluarse también con walk-forward y juzgarse sobre todo por su mejora en los meses invernales.

**Pendiente (Fase 5)**
Comparación expanding vs rolling: sin sentido con baselines (el naive ignora el train, daría idéntico); se hará con el modelo, con hipótesis previa (los datos pre-crisis-de-gas podrían degradar la predicción actual → rolling podría batir a expanding).


---

## 2026-09-02 — Fase 5.1: XGBoost predictivo (base, sin tuning)

**Setup**
- `XGBRegressor` con hiperparámetros de arranque (n_estimators=400, lr=0.05, max_depth=6, subsample/colsample=0.8), **sin tuning**. Objetivo: número base contra el baseline antes de optimizar (el tuning es 5.5, con margen esperado pequeño).
- X = set **predictivo** (102 features, sin `_real` ni `precio_portugal`), y = `precio_espana`. Mismo split que Fase 4. `entrenable` regenerado (ahora cubre los lags) → X sin NaN, sin `dropna` suelto.
- `evaluar` y `walk_forward` extraídos a `src/evaluacion/` (`metricas.py`, `backtesting.py`) para no duplicar la maquinaria entre baseline y modelo. `walk_forward` es genérico: recibe una función `predecir(tr, te)`, así el mismo bucle sirve para naive y para XGB.

**Resultado**
- Holdout: MAE **13,61** / RMSE 18,16 (vs naive holdout 15,5 / 23,69).
- Walk-forward (31 folds): MAE **15,70 ± 5,29** vs naive **18,33 ± 4,69**. El holdout era optimista (igual que pasó con el naive). Número honesto: **15,7 ≈ 14% mejor** que el baseline.

**Hallazgo clave: la media esconde dos regímenes (análisis mes a mes)**
- XGB gana en **24 de 30 meses**. Donde más: otoño/invierno de alta volatilidad (oct-2024 +11,9; oct-2025 +11,5; nov-2025 +10,4) — justo donde el naive era peor. Confirma la tesis: el modelo aporta donde más se necesita.
- XGB **pierde, y catastróficamente, en primavera-2024**: mar-2024 −20,5; abr-2024 −23,4 (los 4 peores meses son feb–may 2024).
- Corte por **año** (más nítido que por estación): 2024 = **−0,44** (empata con el naive), 2025 = **+5,48**, 2026 = **+2,70**. Por estación: invierno +3,19 vs resto +1,93 (ambos positivos, pero contaminados por el desastre de 2024).

**Interpretación (hipótesis fuerte, a verificar)**
En los primeros folds el train está dominado por 2022–2023 (crisis del gas, precios altos). En primavera-2024 el precio se desploma (renovable, demanda baja, negativos) y el modelo, anclado al régimen alto, **sobrepredice**; el naive se adapta al instante. Es un fallo de **cambio de régimen + train inmaduro, NO estacional**: en 2025–2026 XGB maneja bien la primavera. Explica la mayor desviación (5,29): el error no es ruido, se concentra en el arranque.

**Implicación → motiva el experimento expanding vs rolling (punto 4, Fase 5)**
Si el problema es que el expanding arrastra la crisis del gas, una ventana **rolling** (que olvida lo viejo) debería recortar justo los meses de 2024. Hipótesis y evidencia ahora alineadas.

**Pendiente**
- Verificar la hipótesis del régimen (niveles de precio real vs predicho en primavera-2024).
- Experimento rolling vs expanding.
- Modelo **explicativo** (mismo notebook 06, solo cambia X).
- Tuning con Optuna/`TimeSeriesSplit` (5.5).


---

## 2026-09-02 — Fase 5.1 (cont.): experimento expanding vs rolling

**Motivación e hipótesis**
El XGBoost expanding fallaba en primavera-2024 (mar −20, abr −23 vs naive). Hipótesis (arrastrada de la Fase 4): el train arrastra el régimen de la crisis del gas 2022-2023 → una ventana **rolling** que olvide lo viejo debería adaptarse mejor y arreglar 2024. Se añadió el parámetro `ventana` a `walk_forward` (None = expanding; N = rolling de N meses). Ventana = 12 (un ciclo estacional completo; ventanas más cortas perderían la estacionalidad).

**Resultado — MAE walk-forward por año**

| Año | naive | expanding | rolling-12 |
|---|---|---|---|
| 2024 | 18,04 | 18,48 | **19,61** |
| 2025 | 19,91 | **14,43** | 15,67 |
| 2026 | 15,39 | **12,69** | 15,41 |

Global: expanding 15,70 ± 5,29 · rolling-12 17,19 ± 6,14 · naive 18,33 ± 4,69.

**Veredicto: hipótesis RECHAZADA.**
- El rolling **no arregla 2024**: es peor que expanding (19,61 vs 18,48) e incluso peor que el naive (18,04).
- El rolling **pierde contra expanding en los tres años**. Olvidar histórico no ayudó en ningún punto.
- Reinterpretación del fallo de 2024: no era "arrastra basura vieja que confunde", era "régimen de precios bajos casi sin precedentes + con MENOS datos (rolling) va aún peor". El problema no se cura acortando la ventana.

**Decisión de diseño (cerrada con evidencia):** modelo de producción con **ventana expanding**. El valor de más datos de entrenamiento pesa más que el supuesto lastre del régimen viejo.

**Vías reales para el fallo de régimen (no la ventana):** más/mejores features, más histórico, o modelos de intervalos/cuantiles (Fase 5.3) que señalen la incertidumbre en meses raros.

**Cierre con `ventana=24`.** rolling-24: MAE global 15,60 ± 5,31 (≈ expanding 15,70, dentro del ruido). Por año: 2024 idéntico a expanding (aún no hay 24 meses de histórico previo → mismo train); 2025 14,12 vs 14,43; 2026 12,80 vs 12,69. Todas las diferencias minúsculas. El caso decisivo es **2026**, el único año donde rolling-24 y expanding difieren de verdad (rolling-24 ya suelta la crisis del gas): si el histórico viejo estorbara, rolling-24 ganaría ahí — y no lo hace (es un pelo peor). Hipótesis **rechazada sin ambigüedad**: el histórico viejo no degrada la predicción. Decisión final: **expanding** (rolling-24 solo lo iguala añadiendo un hiperparámetro sin beneficio → navaja de Occam).


---

## 2026-09-02 — Fase 5.1 (cont.): modelo explicativo (ex-post) — el "techo" que no lo es

**Setup**
Mismo XGBoost, misma maquinaria; solo cambia X → set **explicativo** (104 features, con las `_real`). Mismo split, mismo target, `y` reutilizado. Sorpresa menor: **0 NaN en las `_real`** (el dato real de ESIOS cubre todo el rango `entrenable`), aunque `entrenable` no las comprobaba.

**Resultado (robusto: igual en holdout y walk-forward)**
| Método | MAE walk-forward | (holdout) |
|---|---|---|
| naive D-1 | 18,33 ± 4,69 | 15,5 |
| **XGB predictivo** (previsiones) | **15,70 ± 5,29** | 13,61 |
| XGB explicativo (reales) | 16,73 ± 5,75 | 14,45 |

El explicativo **bate al naive pero PIERDE contra el predictivo** (~1 punto peor), en las dos evaluaciones.

**Conclusión — el "techo teórico" no existe**
En el mercado diario, la demanda/generación **reales no explican el precio mejor que las previsiones**. Razón de dominio: el precio se casa en subasta **ex-ante**, con la información disponible entonces (las **previsiones**). Los valores `_real` arrastran las **desviaciones post-subasta** (errores de previsión, indisponibilidades imprevistas) que **no formaron el precio** → son ruido irrelevante. El modelo con menos información pero **correcta** (previsiones) gana al que tiene más información pero **posterior** (reales).

**Implicación de negocio (corrige la nota previa del roadmap)**
La palanca NO es "conocer la realidad", es **tener buenas previsiones a tiempo de subasta**. El hueco predictivo↔explicativo NO mide "el valor de previsiones perfectas" (esa lectura era falsa). Lo que demuestra es que entender *cómo se forma el precio* (ex-ante) importa más que acumular datos ex-post.

**No se hace rolling del explicativo:** la cuestión de ventana ya se cerró con el predictivo; el explicativo es diagnóstico y no aportaría nada.


---

## 2026-09-02 — Fase 5.3: intervalos de predicción (cuantiles + conformal)

**Setup**
Cuantiles P05/P10/P50/P90/P95 con XGBoost (`objective="reg:quantileerror"`, `quantile_alpha=q`, un modelo por cuantil). Sobre el modelo predictivo (el campeón). Maquinaria reutilizable en `src/evaluacion/`: `pinball_loss` (métricas) y `walk_forward_cuantiles` (backtesting de cobertura).

**Cruce de cuantiles**
Con 5 modelos independientes, el 34% de las horas tenían cruce (p.ej. P90 > P95). Solución: **rearrangement** (ordenar cada fila con `np.sort`, axis=1). Método con fundamento (Chernozhukov), no un parche.

**Calibración cruda: intervalos SOBRECONFIADOS**
- Holdout: cob_80 = 0,51 (obj 0,80), cob_90 = 0,68 (obj 0,90) → bandas demasiado estrechas.
- Walk-forward (30 meses) confirma que es **sistemático**, no del holdout: cob_80 media ≈ 0,51, **ni un mes llega a 0,80**. Los peores son los de cambio de régimen (mar-2024: 0,087).
- Diagnóstico: fallo **asimétrico** — 36% de reales por debajo de P10 vs 14% por encima de P90; y **mediana sesgada alta** (60% de reales por debajo de P50) → el modelo **sobrepredice** en el régimen bajo de 2026.
- Lección: **pinball bajo (~3-5) pese a mala cobertura** → el pinball no basta para juzgar intervalos; hay que medir cobertura.

**Recalibración con conformal (CQR)**
- Simétrico (holdout): cob_80 0,51→0,72; cob_90 0,68→0,86.
- Asimétrico: ensanche **abajo 6,25 / arriba 0,74** (confirma que el fallo es 100% cola baja) → cob_80 0,74; cob_90 0,88. Mejora clara pero no llega al objetivo.

**Conclusión honesta**
El residuo NO es de método (simétrico ≈ asimétrico en cobertura), es **no-estacionariedad + sesgo de centro**: (1) el test 2026 tiene la cola baja más extrema que el set de calibración → el ensanche aprendido se queda corto; (2) conformal corrige **dispersión, no sesgo de centro**, y el modelo sobrepredice en 2026 — ensanchar la banda hacia abajo es tapar un centro mal colocado con una tirita. La **cura real es adaptación al régimen**: recalibración/retraining sobre datos recientes y conformal adaptativo (online) → conecta con el Bloque 10 (monitorización + trigger de retraining).

**Decisión:** intervalos vía cuantiles + monotonización + CQR asimétrico como base. El gap residual se aborda en producción con recalibración adaptativa, no ampliando más la banda.

**Pendiente:** P(precio<0) y P(precio>150) a partir de los cuantiles (extremos); limpiar las líneas de `importlib.reload` del notebook al cerrarlo; (opcional/rigor) CQR dentro del walk-forward.


---

## 2026-09-02 — Fase 5.3 (cont.): extremos — P(precio<0) y P(precio>150)

**Frecuencias base (el cambio de régimen, cuantificado)**
| Evento | train | test (2026) |
|---|---|---|
| precio < 0 | 1,5% | **11,3%** (×7,5) |
| precio > 150 | 3,0% | **1,1%** |

El mundo pasó de "precios altos, spikes frecuentes, negativos raros" (crisis del gas 2022-23) a "precios bajos, spikes raros, negativos frecuentes" (2026 renovable). Esto **explica retroactivamente** el fallo de la cola baja de los intervalos: el modelo entrenó con apenas 1,5% de negativos → nunca aprendió a esperar precios tan bajos → suelo demasiado alto en 2026.

**Método:** clasificador dedicado por evento (XGBClassifier, binary:logistic) en vez de derivar de los 5 cuantiles (grueso y heredaría la mala calibración). Se separa **discriminación** (¿qué horas?) de **calibración** (¿probabilidad absoluta correcta?).

**Clasificador P(precio<0) — resultados (holdout)**
- Discriminación **excelente**: ROC-AUC **0,935**, PR-AUC 0,666 (base rate 0,113). El modelo sabe QUÉ horas son de riesgo (drivers: renovable alta, demanda baja).
- Calibración **rota**: prob media 0,013 vs frecuencia real 0,113 (subestima ×9). Causa: prevalencia de entrenamiento (1,5%) ≠ realidad 2026 (11,3%). Brier 0,098 (engañoso solo: ≈ el de predecir siempre la base rate; esconde la buena discriminación → nunca juzgar con una métrica sola).

**Recalibración (isotonic, split temporal dentro del test = simula producción)**
- Recalibrado con 1er tramo del test, evaluado en el 2º:
  - prob media 0,024 → **0,222** (real 0,122): arregla la subestimación gorda, pero **se pasa**.
  - AUC 0,957 → 0,955 (intacto: recalibrar no toca el ranking).
  - Brier 0,093 → **0,081** (mejora ~13%).
- El "pasarse" = no-estacionariedad **dentro** de 2026 (el 1er tramo tenía más negativos que el 2º). Lección: recalibrar funciona, pero debe hacerse sobre ventana **reciente y móvil**, actualizada en continuo → Bloque 10.

**Distinción clave (senior):** discriminación (difícil, resuelta: AUC 0,94) vs calibración (fácil de arreglar con datos recientes). Un modelo puede saber QUÉ horas sin acertar el CUÁNTO absoluto.

**Pendiente:** clasificador de spikes (>150) — desbalance brutal en test (1,1%), saldrá más justo; limpiar `importlib.reload` al cerrar el 07.


---

# 📌 Resumen de la sesión (2026-09-02) — Fase 5: modelado XGBoost, intervalos y extremos

Sesión larga y muy productiva. Cerrada la Fase 5.1 (XGBoost) y la Fase 5.3 (probabilístico).
Hilo conductor de todos los hallazgos: **la no-estacionariedad del mercado**.

**Infraestructura y correcciones**
- `entrenable` **regenerado**: antes se calculaba antes de los lags y no cubría sus NaN; movido
  detrás de los lags → ahora filtrar por `entrenable` da X **sin NaN** (30240 filas, era 30552).
- Refactor a `src/evaluacion/`: `evaluar`, `pinball_loss` (metricas.py) y `walk_forward`,
  `walk_forward_cuantiles` (backtesting.py) — maquinaria única para baseline y modelos, patrón
  "función que se pasa como argumento". Bug `mes`/`meses` corregido en el 05.

**Descubrimiento 1 — XGBoost predictivo bate al naive, pero por regiones**
MAE walk-forward 15,70 vs 18,33 (~14%). No es uniforme: **gana en invierno** (donde el naive
era peor) y **se hunde en primavera-2024** (−20/−23), primer aviso de cambio de régimen.

**Descubrimiento 2 — Expanding gana; hipótesis rolling RECHAZADA**
Se pensaba que arrastrar la crisis del gas degradaba el modelo → rolling ganaría. Falso:
rolling-12 peor en los 3 años, rolling-24 empata. El histórico viejo no estorba. Decisión: expanding.

**Descubrimiento 3 — El explicativo NO es un techo**
Usar demanda/generación REALES rinde PEOR (16,73) que usar las PREVISIONES (15,70). Porque el
precio se casa **ex-ante** en subasta con las previsiones; los reales son posteriores. Entender
la formación del precio > acumular datos ex-post.

**Descubrimiento 4 — Los intervalos crudos mienten (sobreconfiados)**
Cuantiles con 34% de cruces (→ monotonización). Cobertura 0,51 vs 0,80 objetivo, sistemática.
Fallo asimétrico: la **cola baja** falla (el modelo no predice precios lo bastante bajos) y la
mediana está sesgada alta → sobrepredice en 2026. CQR (conformal) mejora pero no llega: el
residuo es no-estacionariedad + sesgo de centro (conformal corrige dispersión, no sesgo).

**Descubrimiento 5 — El cambio de régimen, cuantificado**
Precio negativo: **1,5% (histórico) → 11,3% (2026)**, ×7,5. Spikes >150: 3,0% → 1,1%. El mundo
pasó de "caro, spikes frecuentes, negativos raros" a "barato, spikes raros, negativos frecuentes".
Esto explica retroactivamente el fallo de la cola baja de los intervalos.

**Descubrimiento 6 — Discriminación vs calibración**
Clasificador P(precio<0): ROC-AUC **0,935** (sabe QUÉ horas, difícil) pero mal calibrado
(prob media 0,013 vs 0,113, por el desfase de prevalencia, fácil de arreglar). Recalibración
isotonic: no toca el ranking (AUC intacto), baja el Brier; pero debe ser **adaptativa** (ventana
reciente y móvil) porque el régimen deriva incluso dentro de 2026.

**Conclusión transversal**
La no-estacionariedad aparece en TODO: en el punto (primavera-2024), en los intervalos (cola
baja) y en los extremos (×7 negativos). Ningún truco estático lo arregla del todo — la respuesta
es el **Bloque 10** (monitorización + retraining/recalibración adaptativa). Este arco honesto
(plantear hipótesis, probarlas, aceptar que fallan, entender por qué) es el mayor valor de portfolio
de la sesión.

**Próximo paso:** Fase 5.2 — LSTM (notebook 08), a comparar contra XGBoost (15,70) en el punto.


---

## Fase 5.2 — LSTM (notebook 08)

### Tramo A — Preparación de datos y elección de exógenas

Decisión de diseño: **LSTM "pura"** — el canal protagonista es la secuencia histórica de
`precio_espana` (autorregresiva, lookback 168 h para que la red pueda aprender la estacionalidad
semanal por sí misma), acompañada de pocas exógenas previstas. Filosofía deliberada: darle a la
red la oportunidad legítima de brillar aprendiendo el patrón temporal, en vez de regalárselo como
features de calendario/lags (eso es lo que ya tiene XGBoost). Si aun así pierde contra XGBoost,
es un hallazgo honesto y esperable en tabular horario con buen feature engineering.

**Descubrimiento 7 — El "×10" de las previstas ESIOS no es un bug, es resolución sub-horaria**
Al inspeccionar las previstas de demanda/eólica saltaron valores físicamente imposibles
(demanda_prevista ~290.000 "MW", cuando la demanda peninsular real es ~24 GW). No es un fallo del
pipeline: ESIOS devuelve el valor horario como **agregado de submuestras** según la resolución
nativa del indicador. Confirmado por el ratio entre versiones: demanda_prevista/demanda_prevista_diaria
= 3,00 (5-min ×12 vs 15-min ×4) y eolica_prevista/eolica_prevista_d1 = 4,10 (15-min ×4 vs horaria ×1).
La versión _d1 ya viene en MW horarios limpios (~6,9 GW media eólica, físicamente correcto).
Implicación: un factor multiplicativo *constante* por columna lo absorbe el escalado de la LSTM
(y es invisible a los splits de XGBoost) → no invalida nada de lo hecho.

**Descubrimiento 8 — Genérica vs D+1: no son la misma serie, y el std del ratio lo delata**
Correlaciones genérica–D+1: demanda r=0,981, eólica r=0,978. Altas pero NO 1,0 → son previsiones
distintas, no la misma reescalada. La clave está en la **dispersión del ratio**: en demanda
std=0,10 sobre 3,0 (~3%, factor casi constante, apenas divergen); en eólica std=0,62 sobre 4,0
(~15%, el factor baila). Un cambio de unidades no puede hacer que el factor varíe hora a hora →
esa varianza es la firma de una serie que se **refresca intradía** con nowcast. El viento es lo
que más mejora intradía, por eso la genérica se aleja tanto de la D+1.

**Decisión — Se eligen las previsiones D+1/diarias (horizonte de subasta)**
El spot de OMIE se casa a mediodía de D-1: lo único legítimo es lo conocido en ese momento (D+1).
Las genéricas, al refrescarse intradía, arriesgan incorporar información post-subasta → leakage,
mayor en eólica (donde el std alto avisa). Principio de precaución: con r~0,98 elegir la D+1 no
sacrifica poder predictivo (coste ~0), y evita un riesgo real (>0). Selección final de exógenas:
`demanda_prevista_diaria`, `eolica_prevista_d1`, `solar_fv_prevista` (sin D+1 disponible; solar es
muy predecible por el ciclo solar), `solar_termica_prevista` (543 = PBF, ya es día-antes) +
`festivo` (calendario ex-ante que el sin/cos no captura) + calendario cíclico. Canal autorregresivo:
`precio_espana`. Excluidos: todos los `_real` (ex-post), `precio_portugal` (MIBEL co-determinado),
lags/rollings de precio (redundantes: la ventana de 168 h ya los contiene), bloque meteo AEMET
(redundante con las previstas de cantidad, que son el driver ya "digerido"; parsimonia).

**Manejo DST**: único timestamp duplicado en toda la serie = 2023-10-29 01:00 UTC (cambio de hora de
otoño, día de 25 h). Las dos pasadas por la hora local ambigua difieren ~2-3% (irrelevante).
Dedup keep="first" (la pasada CEST, cronológicamente anterior). eolica_prevista_d1 no tiene
duplicados (otro punto a su favor).


### Tramo B — Modelado LSTM y veredicto (holdout + walk-forward)

Diseno: LSTM pura, lookback 168h. Canal autorregresivo = precio con **lag-24**, que
respeta el conjunto de informacion de la subasta dia-antes de OMIE (el precio mas fresco
usable es el del dia anterior, no el de la hora anterior; coherente con que XGBoost use
lag_24/lag_168). Exogenas D+1 (demanda_prevista_diaria, eolica_prevista_d1, solar) +
festivo + calendario ciclico. Modelo simple (1 capa LSTM de 32 + Dense(1)), loss=MAE,
early stopping. Escalado por fold (fit solo en train). Corrido en GPU (Colab).

**Descubrimiento 9 - El holdout unico enganya; el walk-forward manda**
En el holdout de 2026 (6 meses) el ensemble LSTM parecia batir a XGBoost (MAE 12,12 vs
13,55, mismas horas). Pero el holdout solo medía el regimen reciente y favorable. El
walk-forward completo (30 meses, misma maquinaria que XGBoost, inicio=12, expanding) lo
desmiente: **LSTM ensemble 16,59 ± 7,96 vs XGBoost 15,70 ± 5,29**. Leccion de metodo: un
unico split puede mentir; el walk-forward es el juez.

**Descubrimiento 10 - Varianza por semilla y el remedio del ensemble**
Un LSTM suelto tiene ~7% de varianza de MAE solo por la semilla de inicializacion
(13,51 ± 0,91 sobre 4 semillas en el holdout; XGBoost es determinista). Promediar 4
semillas (ensemble por averaging) baja el error por debajo de la mejor semilla individual
y estabiliza. Se adopta el ensemble como artefacto reproducible. La varianza en si es un
resultado: la inestabilidad es una desventaja practica frente al determinismo de XGBoost.

**Descubrimiento 11 - La no-estacionariedad, otra vez, y mas fuerte en la LSTM**
El ±7,96 del walk-forward es un modelo de dos mitades: catastrofico en primavera-2024
(MAE 39/43/28 - el cambio de regimen post-crisis del gas, con solo 12-15 meses de
histórico) y sobresaliente en 2025-2026 (9-13, por debajo de la media de XGBoost en varios
meses). La LSTM, anclada al histórico reciente, revienta en la transicion aun mas que
XGBoost, pero DOMINA el regimen vigente. Es el mismo hilo de no-estacionariedad que
atraviesa todo el proyecto, ahora en la LSTM.

**Veredicto Fase 5.2**
En media, empate estadistico (16,59 vs 15,70; la diferencia 0,89 es menor que el error
estandar ~1-1,5). En estabilidad, gana XGBoost (2/3 de la varianza) -> mejor modelo UNICO
en produccion. Pero el ensemble LSTM es superior en el regimen reciente (2025-26) y queda
como candidato fuerte para (a) el ensemble XGB+LSTM (Fase 5.4) y (b) el retraining
adaptativo / ventana reciente (Bloque 10). XGBoost sigue siendo el punto de referencia (15,70).


### Bloque 6 — Interpretabilidad (SHAP sobre XGBoost)

Técnica: TreeSHAP (exacto para árboles) sobre el XGBoost predictivo, explicando el holdout de 2026.

**Descubrimiento 12 — El modelo razona como el merit order (global + local)**
Ranking global: dominan los lags de precio (lag_24 el mayor, lag_168, media_24 → persistencia +
estacionalidad diaria/semanal), y tras ellos los fundamentales previstos (solar_fv #1 de los
fundamentales, eólica, demanda). El bloque meteo de AEMET queda al fondo → **confirma empíricamente
la decisión de excluirlo** (redundante frente a las previsiones de cantidad que lo subsumen).
Beeswarm: signos correctos — demanda ↑ sube precio, renovables ↑ lo bajan. Waterfalls locales: una
hora negativa se construye con renovables restando ~-28 € (mecanismo de precios negativos explícito);
una hora cara, con lag_24 y demanda sumando y escasez de renovables (solar=0, viento bajo) sumando.
El modelo entiende escasez/abundancia, no memoriza.

**Descubrimiento 13 — Sesgo de régimen (-3,58) y colas, otra vez la no-estacionariedad**
Holdout 2026: MAE 13,55 y bias (residuo medio) = **-3,58** → sobre-predice de media. Por tramos:
sobre-predice negativos (resid -5,03) y normales (-6,61), infra-predice los picos (+8,54). Es
"encogimiento hacia el régimen viejo": el valor base SHAP (72,5 €) está anclado al régimen caro de
entrenamiento y no alcanza los extremos del 2026. Error por hora: peor en las rampas (mañana 7-9h;
tarde 18-19h ~16,5), calmo de noche y a mediodía. Triangula desde interpretabilidad el mismo hilo que
el punto (primavera-2024) y los intervalos (cola baja) → respuesta = recalibración/retraining adaptativo
(Bloque 10). El sesgo -3,58 es corregible barato (restar residuo reciente) = versión mínima de esa recalibración.

Nota de rigor: SHAP muestra que XGBoost se apoya en demanda_prevista y eolica_prevista GENÉRICAS
(las de posible leakage sutil intradía). Candidatas a limpieza futura para dejarlo a solo-D+1 como la LSTM.

Documento explicativo del bloque: docs/interpretabilidad.md


---

## 2026-09-08 — Bloque 7: MLflow + DVC (tracking, registry y versionado de datos)

Objetivo: dejar de trabajar de memoria. Convertir "entrené un modelo y salió un número" en
un registro trazable, reproducible y comparable, y versionar el dato que lo alimenta.

**Concepto — las dos mitades de MLflow**
- *Tracking*: cuaderno de laboratorio automático. Cada `start_run` es una foto de un experimento
  con tres cosas: params (la receta), metrics (el resultado) y artifacts (el modelo).
- *Model Registry*: catálogo de modelos "buenos". Coge un modelo de un run y le da nombre estable
  (`spotprice-xgboost`), versión (v1, v2…) y alias (`@champion`). La API de la Fase 7 no abrirá un
  `.pkl`: pedirá `spotprice-xgboost@champion`, desacoplando servicio de entrenamiento.

**Decisión de infraestructura — SQLite directo, sin servidor HTTP**
Se descartó el `mlflow server` para loguear. El notebook escribe directo al fichero
`sqlite:///mlflow.db` de la RAÍZ (ruta absoluta, para que no dependa del cwd de los notebooks).
El registry exige backend de BD, por eso SQLite y no el file-store por defecto. Para visualizar,
`python -m mlflow ui` cuando haga falta. En Docker (Fase 12) ya se montará un servidor de verdad.

**Problema real 1 — App Control de Windows bloquea los `.exe` de pip**
La directiva "Control de aplicaciones" bloquea los lanzadores `mlflow.exe` / `dvc.exe` que crea pip.
Regla adoptada: invocar siempre por módulo, `python -m mlflow ...` / `python -m dvc ...`, que entra
por `python.exe` (de confianza). Corolario grave: `mlflow.xgboost.log_model` por defecto infiere el
entorno lanzando un subproceso `pip` → ese subproceso lo bloquea App Control y la celda se CUELGA sin
error (21 min hasta interrumpir). Solución: pasar `pip_requirements=["xgboost","scikit-learn"]`
explícito para desactivar la inferencia. Medido: log_model con inferencia ~3,9 s; sin ella ~0,5 s.

**Problema real 2 — el tracking_uri se enrutó a la BD equivocada**
Síntoma: runs que "funcionaban" pero no aparecían en la UI. Causa: `get_tracking_uri()` apuntaba a
`sqlite:///notebooks/mlflow.db` (ruta relativa resuelta desde `notebooks/`), una BD distinta a la del
servidor. Lección: el destino de MLflow es estado de proceso; fíjalo explícito y con ruta absoluta.

**Estructura de runs — tres, bien diferenciados por el tag `validacion`**
- `holdout_6m`: entrenamiento único sobre los últimos 6 meses. MAE 13,6 → número OPTIMISTA.
- `walk_forward_expanding`: 30 reentrenamientos mes a mes. MAE por fold logueado con `step=i`
  (MLflow lo pinta como curva → se ve el pico de 2024) + resumen `MAE_wf_mean` 15,70 / `MAE_wf_std` 5,29.
  Es el número HONESTO. El backtesting (`walk_forward`) no toca MLflow; el logging envuelve su DataFrame.
- `produccion_full_data`: modelo entrenado con TODOS los datos, el que va a producción.
El tag `validacion` es un post-it de texto (mutable, a diferencia de los params, inmutables) que permite
filtrar y no confundir nunca el número optimista con el honesto.

**Decisión clave — qué modelo se registra**
Se valida con walk-forward, pero se SIRVE un modelo entrenado con todo el histórico (en el momento de
predecir mañana usarías todos los datos hasta hoy, no una versión recortada). Ese modelo no tiene MAE
propio (evaluar sobre datos de entrenamiento sería leakage), así que se le adjunta como referencia el
MAE del walk-forward (`MAE_wf_ref` = 15,70): es la mejor estimación de su error real, sirve de ficha
técnica en el registry y de línea base para la monitorización de drift (Fase 10).

**Registry — versión vs alias**
Versión = foto inmutable que sube sola (v1, v2…). Alias = etiqueta móvil que pega a UNA versión. La API
pedirá `@champion` sin saber el número; al reentrenar (Fase 10) se registra v2 y se mueve el alias, y la
API sirve la nueva sin tocar código. Registrado `spotprice-xgboost` v1 con alias `@champion`.

**DVC — Git para datos**
Git es malo con binarios grandes. DVC guarda el fichero fuera de Git (en `.dvc/cache`) y deja en Git solo
un puntero de texto `.dvc` con el hash. Versionado `data/processed/tabla_features.parquet`
(md5 802a9623…, 3,6 MB). Complementa a MLflow: MLflow dice "usé el dataset X", DVC garantiza que X es
reproducible bit a bit. A 3,6 MB Git aún lo aguantaría; se monta por la práctica y de cara a los modelos
pesados (LSTM reentrenado) que vendrán.

**Hallazgos de higiene del repo**
- El "Git ✓" del roadmap (Fase 0.2) no estaba hecho: `git init` real ejecutado hoy (primer commit del repo,
  solo los ficheros de DVC — nada de `git add .`).
- El `.env` con tokens ESIOS/AEMET ya estaba en `.gitignore` (bien). Añadidos `mlflow.db`, `mlruns/` y huérfanos.
- PENDIENTE antes de subir a GitHub (Fase 11): limpiar outputs de notebooks gigantes con `nbstripout`
  (`01_ingesta_esios` ~157 MB, `01_ingesta_aemet` ~87 MB) — GitHub rechaza ficheros >100 MB.
- Choque de dependencias resuelto: MLflow 3.14 exige `cryptography<49`, DVC (asyncssh) exige `>=48.0.1`
  → única versión-puente `cryptography==48.0.1`. Argumento a favor de aislar DVC en su propio entorno.

**Pendiente para Fase 10 (no se hizo hoy, por decisión)**
Registrar el LSTM (`spotprice-lstm`) como custom pyfunc (preprocesado + scaler + ventaneo 168h + 4 semillas)
para el gating XGBoost↔LSTM por régimen. Prerrequisito: exportar de Colab los pesos de las 4 semillas +
scaler + config; ahora solo hay `reports/lstm_tensores.npz` (tensores de entrada, no el modelo). El registry
ya soporta dos modelos independientes; el gating (elegir cuál según régimen) es trabajo de la Fase 10, con la
señal de régimen que dará la monitorización de drift — no se cablea a mano.


---

## 2026-09-09 — Fase 7: Serving con FastAPI

Objetivo: convertir "tengo un modelo en el registry" en "tengo un servicio que predice". Un modelo
guardado no vale nada si nadie puede llamarlo; el serving es la puerta de entrada al modelo.

**Concepto — qué es y por qué una API (no un script)**
FastAPI envuelve el modelo en un servicio web. Da tres cosas gratis: validación de entrada/salida con
Pydantic (rechaza basura antes de llegar al modelo), documentación interactiva automática en `/docs`
(Swagger), y es async. Por qué una API y no llamar al modelo en un script: es la **puerta común y
reutilizable** para muchos consumidores (el proceso diario de Prefect, el dashboard de Streamlit, quizá
un tercero), versionada y testeada, sin que cada uno reinvente "cómo llamar al modelo". Es la forma que
espera el sector.

**Estructura**
`src/api/main.py` (app + endpoints) y `src/api/schemas.py` (contratos Pydantic). Se arranca con
`python -m uvicorn src.api.main:app --reload` (el `python -m` por el App Control de Windows, que
bloquea `uvicorn.exe`).

**Carga del modelo en el `lifespan` (el pago del registry)**
El modelo se carga UNA vez al arrancar (no en cada petición — sería lento), en el handler `lifespan`
de FastAPI. Se hace con `mlflow.pyfunc.load_model("models:/spotprice-xgboost@champion")`: pide el modelo
por alias, sin saber que es la v1 ni dónde está el fichero. El día que se promueva una v2 (Fase 10), esta
misma línea sirve la nueva sin tocar código. `pyfunc` es la interfaz genérica (mismo `.predict()` para
XGBoost o el LSTM futuro) → clave para el gating.

**Endpoints**
- `/health` (GET) → ¿vivo? (trivial pero esencial para Docker/monitorización).
- `/model-info` (GET) → nombre, versión (resuelta desde `@champion`), run_id y `MAE_wf_ref` (15,70) →
  la API dice qué sirve Y cómo de bueno se espera que sea.
- `/predict` (POST) → predice.

**Decisión de diseño — qué recibe `/predict` (opción A vs B)**
- Opción A (la de ahora): recibe el vector de las 102 features predictivas ya construido. La API valida
  el conjunto y predice. Es el MOTOR: limpio, aislado, testeable.
- Opción B (futura): recibe una fecha D+1 y la API arma/busca las features sola. Es la PUERTA CÓMODA.
Se hace A primero para no mezclar "aprender a servir" con "orquestar features". En producción nadie mete
102 features a mano: las arma el sistema (Prefect) y llama a la API; el dashboard la consume.

**Contrato Pydantic**
Request = `features: dict[str, float]` (un solo campo mapa nombre→valor; declarar 102 campos sería
inmantenible). Pydantic garantiza que son floats; la comprobación de que estén las 102 exactas la hace el
endpoint (si faltan → HTTP 422 con mensaje). El vector se arma como DataFrame de 1 fila EN EL ORDEN de
`get_feature_sets(...)["predictivo"]` (el modelo es sensible al orden de columnas). Anti-leakage: las
esperadas salen del set "predictivo", nunca del explicativo.

**Descubrimiento 14 — la API confirma el sesgo de la interpretabilidad**
Con una fila real (precio real 123,795 €/MWh), la API predijo 111,75 → ~12 corto. No es un fallo: esa
hora es un precio ALTO y el modelo infra-predice los picos, exactamente el sesgo -3,58 ("encogimiento
hacia el régimen viejo") hallado con SHAP en el Bloque 6. Coherencia total entre lo interpretado y lo
servido.

**Tests (pytest + TestClient)**
`tests/test_api.py`: `TestClient` levanta la app en memoria sin uvicorn. 4 tests: /health OK, /model-info
correcto, /predict con payload válido (200) y /predict incompleto rechazado (422) — camino feliz + portero.
Gotcha nº1 de FastAPI: hay que usar `with TestClient(app) as c:` para que se dispare el `lifespan` (si no,
el modelo no se carga y /predict peta con KeyError). Los 4 en verde.


## 2026-09-15 — Fase 8: Orquestación con Prefect

**Qué se hizo**
Flow diario de ingesta (`src/orquestacion/pipeline_diario.py`) que encadena las
tres fuentes como subflows: `ingesta_omie()`, `ingesta_esios()`, `ingesta_aemet()`.
Programado para ~13h (tras la subasta de OMIE). Más un flow de reentrenamiento
periódico. `python -m prefect ...` por el App Control de Windows (bloquea los shims `.exe`).

**Decisión de arquitectura — orquestación por capas (importante)**
Los módulos de dominio (`ingesta/`, `procesamiento/`) NO dependen de Prefect: son
Python puro, importables desde notebooks o tests sin arrancar un runtime. Prefect
vive solo en `src/orquestacion/`. La orquestación **se hereda por el contexto de
llamada, no por decorar cada función**: lo que se programa y ejecuta a diario es el
`@flow` de arriba; todo lo que llame desde dentro (sea `@task`, subflow o función
normal) corre dentro de esa ejecución, con sus logs y su parada ante excepción.

**Cuándo un `@task` y cuándo no**
`@task` es la unidad de reintento/caché/observabilidad. Tiene sentido en la
**ingesta** (una API que falla por un pico de red y se recupera). NO tiene sentido
en transformaciones en memoria (no hay nada que reintentar en aislado, y pasar
DataFrames entre tasks obliga a serializar sin beneficio). Por eso las funciones de
procesamiento se dejan puras y se envuelven —si hace falta— desde la capa de
orquestación, no al revés.

---

## 2026-09-15 — La saga del bug de ESIOS: `guardar()` machacaba el histórico

**El síntoma**
La tabla ESIOS cruda había quedado en **97 filas**. La validación de CONTENIDO
(pandera) la daba por buena: 97 filas con tipos y rangos correctos pasan el
esquema. El colapso era invisible para el contrato de contenido.

**La causa raíz (doble)**
1. `guardar()` escribía con `to_csv` **sin append** → cada ejecución sobrescribía
   el fichero entero en vez de acumular.
2. La ventana de descarga estaba **fija a 4 días**. Combinada con lo anterior, cada
   corrida dejaba solo los últimos días y borraba todo el histórico previo.

**El arreglo**
- Se confirmó que la API de ESIOS **sí sirve histórico** (no era una limitación de la fuente).
- Re-pull completo 2023→hoy → recuperadas las **32.496 filas**.
- Se blindó `guardar()`: **append + dedupe** (`keep="last"`, reconvirtiendo `datetime_utc`)
  + `sort` + **ventana dinámica** (7 días atrás, 1 adelante).

**Alcance del daño (honestidad)**
El modelo ya entrenado NO se vio afectado: `tabla_features` conservaba el histórico
completo. Lo que estaba roto era la **reproducibilidad** (regenerar desde crudo daba
97 filas). Ahora arreglado. Lección: una validación de contenido no detecta un
colapso de volumen → de aquí nace el contrato de cobertura (siguiente entrada).

---

## 2026-09-15 — Fase 2: contrato de cobertura (`validar_cobertura`)

**Qué se hizo**
Nueva función en `control_datos.py` complementaria al esquema de pandera. El esquema
valida el CONTENIDO de la maestra; la cobertura valida las FUENTES CRUDAS antes de
procesar: (1) suela de filas por fuente (detecta un colapso tipo ESIOS-97) y (2)
frescura — que el dato más reciente no sea más viejo que `dias_frescura` días
(detecta una fuente que dejó de actualizarse). Recoge todos los fallos y lanza
`ValueError` si hay alguno.

**Frescura: `hoy − fecha_max`**
`hoy` = fecha de calendario del sistema; `fecha_max` = último dato del DataFrame.
La resta da los días de retraso desde la última ingesta. Umbrales: OMIE/ESIOS 2 días,
AEMET 4 (publica con ~2 días de retraso).

**Supuesto documentado (UTC/local)**
La frescura se mide en días de calendario comparando `hoy` (local) contra `fecha_max`
(derivada de `datetime_utc`, UTC). Para umbrales en días completos no cambia el
veredicto, pero queda anotado como supuesto.

**Verificado (2026-09-16)**: pasa. Filas muy holgadas; ESIOS confirma las 32.496 recuperadas.

**Limitación (honestidad)**
El suelo de filas detecta un colapso TOTAL (ESIOS-97), no una degradación PARCIAL
reciente (perder los últimos 3 días no baja del mínimo). Es el alcance buscado ahora.

---

## 2026-09-16 — Extensión del pipeline: concatenación y ETL modularizados

**Objetivo**
Encadenar `concat → etl → features` tras la ingesta, para que el flow diario llegue
hasta `tabla_features.parquet`. Se convierten los notebooks 02/03/04 en módulos de
`src/procesamiento/`.

**`concatenacion.py` cerrado**
Función orquestadora `construir_tabla_maestra()` (cargar → pivotar AEMET →
datetime OMIE/ESIOS → concatenar) + `if __name__`. Verificado con datos reales:
**30.623 filas × 96 columnas**, `datetime_utc` único y ordenado, target `precio_espana`
al 100%, ESIOS previstas al 100%, 0 duplicados. Las 30.623 filas frente a las ~32.448
horas teóricas del rango cuadran con el **hueco conocido de OMIE (~1.848 horas)**
pendiente de imputación.

**Decisión de arquitectura — `pivot` vs `pivot_table` (capa estructural vs limpieza)**
El notebook 02 usaba `.pivot()` (reorganiza, NO agrega, tolera strings). Al modularizar
se probó `.pivot_table()`, que SIEMPRE agrega (con `mean` por defecto) y por tanto exige
columnas numéricas → reventaba con las de AEMET en texto. Se decide **volver a `pivot`**
y mantener la limpieza de tipos en el ETL. Motivo: el artefacto se llama `tabla_maestra_
ESTRUCTURAL` — esa capa es *estructura* (unir y alinear fuentes), no *limpieza*. Separar
"montar la estructura" (concatenación) de "limpiar el contenido" (ETL) es la separación
de responsabilidades correcta y coherente con las capas interim→processed. `pivot` además
exige pares (fecha, estación) únicos → chequeo de integridad gratis (verificado: 0 duplicados).

**Decisión de dominio — centinelas de AEMET**
AEMET publica en formato español (coma decimal) y mete códigos de texto en columnas
numéricas. En `prec`: `'Ip'` (241 casos) = precipitación **i**na**p**reciable (<0,1 mm) y
`'Acum'` (1 caso) = acumulado de varios días. Decisión:
- `'Ip'` → **0** (sabemos que casi no llovió; es dato real, no hueco).
- `'Acum'` → **NaN** (el valor diario es desconocido; lo trata la imputación).

**Bug encontrado y corregido (mejora sobre el notebook)**
En el notebook 03 el orden era `replace({'Ip': 0})` (0 entero) ANTES de `.str.replace`.
Como `.str` sobre un entero devuelve NaN, los 241 `'Ip'` se convertían silenciosamente
en **NaN**, no en 0 — la decisión documentada no se ejecutaba. En `etl.py` se corrige el
orden: **coma→punto primero, luego centinelas, luego `astype(float)`**, así los 241 `'Ip'`
se conservan como 0. Impacto real bajo (una variable meteo, además imputada), pero es el
tipo de fallo que solo aflora verificando con datos, no a ojo.

**`etl.py` modularizado**
`ejecutar_etl()`: cargar estructural → `limpiar_aemet` → `validar_warehouse` (que
**propaga** si falla: puerta dura, no `try/except` que trague) → guardar
`tabla_maestra_procesada.parquet`. Rutas relativas con `Path(__file__).parents[2]`
(fuera el `sys.path.append("..")` del notebook). `select_dtypes('object')` aísla las
columnas de AEMET dinámicamente, sin listas fijas (tras la concatenación, las únicas
de texto son las de AEMET).

**Desajuste contrato ↔ pipeline por leakage (hallazgo importante)**
Al ejecutar el ETL, la validación falló con `COLUMN_NOT_IN_DATAFRAME` en las **8
columnas `_real`** de ESIOS. El esquema de pandera aún las exigía, pero el pipeline las
excluye (correctamente) por ser la tabla explicativa **leakage** para el target. El
contrato estaba obsoleto: nunca se había validado contra los datos sin leakage. Se
actualizó `control_datos.py`:
- `ESIOS_NO_NEGATIVAS` → renombrada a **`ESIOS_PREVISTAS`** (6 columnas). El eje que importa
  ahora es *prevista vs real-excluida*, no *signo*: la distinción negativas/no-negativas
  existía para contrastar con las `_real`, que ya no están.
- Eliminada `ESIOS_CON_NEGATIVOS` (solo contenía `_real`).

**Matiz de dominio pendiente (a futuro)**
La generación `_real` de D-1 SÍ es conocida a la hora de predecir D+1 y **no** sería
leakage como feature con lag. Reincorporarla así es una decisión aparte, no abordada hoy.

---


## 2026-09-17 — Cierre y puesta en marcha del pipeline diario (features, cableado, concurrencia)

**Feature engineering modularizado** (`src/procesamiento/features.py`, desde el notebook 04)
Imputación en tres capas por CAUSA del hueco: (1) interpolación temporal para micro-huecos ≤6h,
(2) donante entre estaciones AEMET (solo temperatura, correlación 0,9+) con offset mensual, (3)
climatología (media mensual) para viento/lluvia/sol. Después: calendario + cíclicas, lags de precio,
y la marca `entrenable` **al final** (tras los lags, para que sus NaN cuenten). Rutas relativas con
`parents[2]`. Verificado: 30.623 × 112, 30.263 filas entrenables.

**Cableado del pipeline diario** (`pipeline_diario.py`)
Cadena completa: ingesta → cobertura (puerta) → concat → etl → features. Las funciones de dominio siguen
siendo Python puro; en la capa de orquestación se envuelven en `@task` finos. Se llaman **directas** (no
`.submit()`), así Prefect las ejecuta en serie y el orden queda garantizado sin `wait_for`. La cobertura
va con `retries=0` (reintentar un chequeo no crea datos).

**Umbrales de frescura ajustados** (OMIE 3, ESIOS 3, AEMET 5 días)
Los 2/4 originales vivían justo en el filo y saltaban con un solo día sin ejecutar. Criterio nuevo:
umbral = retraso de publicación de la fuente + colchón para días sin correr (AEMET publica con ~2 días
de retraso, por eso 5). La puerta pasa a ser una alarma útil en vez de un falso positivo cada finde.

**Saga de concurrencia (hallazgo importante)**
Al levantar `serve()` por primera vez, recogió varios runs "Late" acumulados (quick-runs previos sin
ejecutor) y los lanzó **a la vez**. Como las etapas se comunican por ficheros compartidos, varios pipelines
leyendo/escribiendo los mismos `.parquet` provocaron una **condición de carrera**: un `concat` falló con
`ValueError: Index contains duplicate entries, cannot reshape`. Causa: un lector pilló `tabla_AEMET` a medio
reescribir (`to_csv` no es atómico) con duplicados transitorios. Síntoma revelador: el mismo código fallaba
en un run y no en otro → no-determinismo = carrera.

Lo bonito: la decisión de usar **`.pivot()` en vez de `.pivot_table()`** actuó de red de seguridad —falló
ruidosamente ante los duplicados en vez de promediarlos en silencio—. `pivot_table` habría producido una
tabla mal sin avisar.

**Solución:** `concurrency_limit=1` en el deployment de ingesta (en `programar.py`). Aunque se acumulen runs,
se ejecutan en serie, uno tras otro, nunca solapados → carrera eliminada de raíz.

**Nota operativa (Prefect):** el *servidor* (`prefect server start`, UI en 4200) orquesta y muestra; el
`serve()` (`programar.py`) es quien **ejecuta**. Un run cuya hora pasó sin ejecutor queda "Late" y se recoge
en cuanto `serve()` vuelve a estar vivo. En producción real, ambos procesos van siempre encendidos (Docker).

---

## 2026-09-17 — Fix del filtro `entrenable` en reentrenamiento + versionado con DVC

**Fix en `reentrenamiento.py`**
`entrenar()` hacía `modelo.fit(df[feats], df[TARGET])` sobre **todo** el df, sin filtrar por `entrenable`.
Entrenaba sobre ~360 filas que la máscara marca como NO entrenables (primera semana sin `precio_lag_168`,
cola, huecos), con NaN justo en `lag_24`/`lag_168` — las features más importantes según SHAP. XGBoost tolera
NaN, así que no daba error: un fallo **silencioso**. Corregido añadiendo `df = df[df["entrenable"]]` antes del
fit, para que producción entrene sobre la misma población (X sin NaN) que se evaluó en los notebooks. Coherencia
recuperada entre lo documentado, lo medido y lo servido.

**Versionado del dataset con DVC**
Se versiona `data/processed/tabla_features.parquet`. El porqué de fondo: el pipeline **no es reproducible en el
tiempo** —OMIE/ESIOS/AEMET cambian y añaden histórico, así que "regenerar desde código" no devuelve el dataset
de hace meses—. DVC guarda el snapshot exacto y lo ata a un commit de git, de modo que cada modelo queda ligado
al dataset con el que se entrenó (auditable y reproducible).

Mecánica: git versiona un *pointer* `.dvc` (con el md5); DVC guarda el dato real en un *remote* local
(`dvc-storage`, fuera del repo). Flujo: `dvc add` (actualiza el pointer + caché) → `git commit` del pointer
(ata código↔dato) → `dvc push` (sube el dato). Alcance: solo `tabla_features` (la que alimenta el entrenamiento);
la procesada/interim se derivan y los crudos se regeneran. Se matizó la frase del README ("los datos no se
versionan"): código en git, dataset clave en DVC, crudos regenerables.

---


### Próximos pasos (orden de cierre)

Estado: Fases 0–5 (modelado), Bloque 6 (interpretabilidad), Bloque 7 (MLflow+DVC), Fase 7 (serving) y Fase 8 (Prefect) ✅. Extensión del pipeline: concatenación y ETL modularizados y verificados; falta features + cableado completo.

1. **Docker + docker-compose + GitHub Actions (CI)** — contenerizar API + MLflow; automatizar lint+tests
   en cada push. PRERREQUISITO antes del push a GitHub: `nbstripout` a los notebooks gigantes
   (01_ingesta_esios ~157MB, aemet ~87MB) — GitHub rechaza >100MB.
2. **Prefect (orquestación)** — flow de ingesta diaria (~13h, tras subasta) + flow de reentrenamiento.
3. **Evidently (monitorización)** — data/model drift, predicho vs real diario, trigger de reentrenamiento;
   AQUÍ entra el **gating XGBoost↔LSTM por régimen** → requiere registrar el LSTM (`spotprice-lstm`) como
   custom pyfunc (prerequisito: exportar de Colab pesos de las 4 semillas + scaler + config de ventaneo).
4. **Dashboard Streamlit** — predicción D+1 con intervalos; panel de drift; predicho vs real.
5. **Docs MkDocs + README pulido** — arquitectura, decisiones, caso de negocio (€ ahorrados).
6. **Al final:** economía/trading (P&L, Sharpe) y tuning con Optuna.

Cabos sueltos: opción B del `/predict` (por fecha); limpiar demanda/eolica GENÉRICAS del XGBoost a solo-D+1;
gas MIBGAS/TTF como variable e indicador líder de crisis (limitación documentada).
