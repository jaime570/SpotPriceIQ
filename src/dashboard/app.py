"""Dashboard de SpotPriceIQ (solo lectura).

Lanzar desde la raíz del proyecto:  python -m streamlit run src/dashboard/app.py
(python -m añade la raíz al sys.path, por eso funciona `from src...`)."""

import pandas as pd
import streamlit as st

from src.dashboard.datos import (
    RUTA_EVALUACION,
    RUTA_REGISTRO,
    TABLA_FEATURES,
    cargar_datos,
    con_huecos,
    predicho_vs_real,
    ultima_prediccion,
)
from src.monitorizacion.evaluacion import DIAS_SEGUIDOS, UMBRAL_RATIO, UMBRAL_SESGO
from src.orquestacion.prediccion_manana import MARGEN_INICIO

st.set_page_config(page_title="SpotPriceIQ", layout="wide")

def _mtime(ruta):
    """Fecha de modificación del fichero (0 si no existe). Es la clave de la caché."""
    return ruta.stat().st_mtime if ruta.exists() else 0


@st.cache_data
def _cargar(mtime_registro, mtime_evaluacion, mtime_features):
    """Los argumentos no se usan dentro: solo sirven para que la caché se invalide
    cuando el pipeline reescribe algún fichero (11:15 predicción, 14:30 evaluación)."""
    return cargar_datos()


registro, evaluacion, precio_real = _cargar(
    _mtime(RUTA_REGISTRO), _mtime(RUTA_EVALUACION), _mtime(TABLA_FEATURES)
)

st.title("SpotPriceIQ · precio spot OMIE D+1")
st.caption(f"Registro: {len(registro)} horas predichas · "
           f"Evaluación: {0 if evaluacion is None else len(evaluacion)} días")

ult = ultima_prediccion(registro)
dia_objetivo = ult["fecha_objetivo"].iloc[0]
modelo = f"v{ult['modelo_version'].iloc[0]}"
momento = ult["momento_prediccion"].iloc[0].tz_convert("Europe/Madrid")
ahora = pd.Timestamp.now(tz="Europe/Madrid")

if ahora.time() < MARGEN_INICIO:
    esperado =ahora.date()
else:
    esperado= ahora.date() + pd.Timedelta(days=1) 

ultimo=dia_objetivo.date()

if ultimo >= esperado:
    st.success(f"Predicción al día · última: {ultimo:%d-%m} (hecha el {momento:%d-%m a las %H:%M})")
else:
    st.warning(f"Falta la predicción del {esperado:%d-%m}. Última disponible: {ultimo:%d-%m}")
    
st.subheader(f"Predicción para el {dia_objetivo:%d-%m-%Y}")

st.caption(f"Modelo {modelo} · predicho el {momento:%d-%m-%Y a las %H:%M}")


id_minimo = ult["prediccion"].idxmin()
id_maximo = ult["prediccion"].idxmax()
media = ult["prediccion"].mean()
hora_min = ult.loc[id_minimo, "hora_madrid"]
hora_max = ult.loc[id_maximo, "hora_madrid"]
precio_max = ult['prediccion'].max()
precio_min = ult['prediccion'].min()


c1,c2,c3 = st.columns(3)
c1.metric("Precio medio", f"{media:.2f} €/MWh")
c2.metric(f"Precio mínimo · {hora_min:%H:%M}", f"{precio_min:.2f} €/MWh")
c3.metric(f"Precio máximo · {hora_max:%H:%M}", f"{precio_max:.2f} €/MWh")

grafico = ult.rename(columns={"hora_madrid": "Hora", "prediccion": "Modelo", "naive_d1": "Naive (Precio ayer)"})

st.line_chart(
    grafico,
    x="Hora",
    y=["Modelo", "Naive (Precio ayer)"],
    color=["#4C9BE8", "#9AA0A6"],
)

#tabla de las 4 horas más baratas y 4 más caras, con hora y precio previsto
def _tabla_horas(df):
    """Hora (texto HH:MM) y precio predicho, listo para mostrar."""
    return pd.DataFrame({
        "Hora": df["hora_madrid"].dt.strftime("%H:%M"),
        "Precio previsto (€/MWh)": df["prediccion"].round(2),
    })

baratas = ult.nsmallest(4, "prediccion").sort_values("hora_madrid")
caras = ult.nlargest(4, "prediccion").sort_values("hora_madrid")

col_b, col_c = st.columns(2)
with col_b:
    st.markdown("**4 horas más baratas** · consumir / cargar")
    st.dataframe(_tabla_horas(baratas), hide_index=True, use_container_width=True)
with col_c:
    st.markdown("**4 horas más caras** · evitar / vender")
    st.dataframe(_tabla_horas(caras), hide_index=True, use_container_width=True)


#  Fiabilidad reciente de la lista: historial, NO el día mostrado (su precio real aún no existe).
if evaluacion is not None:
    recientes = evaluacion.tail(7)          # últimos 7 días EVALUADOS (no de calendario)
    n = len(recientes)
    spearman = recientes["spearman"].median()
    regret_c = recientes["regret_compra_k4"].median()
    regret_v = recientes["regret_venta_k4"].median()
    aviso = " · orientativo, pocos datos" if n < 5 else ""
    st.caption(
        f"Fiabilidad reciente (mediana de los últimos {n} días evaluados{aviso}): "
        f"orden de las horas Spearman {spearman:.2f} · comprando en sus 4 horas más baratas "
        f"se pagó {regret_c:.1f} €/MWh más que en las 4 mejores reales · vendiendo en las "
        f"4 más caras, {regret_v:.1f} €/MWh menos que en las 4 mejores."
    )


st.subheader("Predicho vs real")
pvr = predicho_vs_real(registro, precio_real)
primera_fecha = pvr["fecha_objetivo"].min().date()
ultima_fecha = pvr["fecha_objetivo"].max().date()

rango = st.date_input("Periodo", value=(primera_fecha, ultima_fecha), min_value=primera_fecha, max_value=ultima_fecha)
if len(rango) != 2:
    st.info("Elige la fecha final del periodo")
    st.stop()

inicio, fin = rango

filtro = (pvr["fecha_objetivo"].dt.date.between(inicio, fin))
grafico_pvr = con_huecos(pvr[filtro]).rename(columns={"hora_madrid": "Hora", "precio_real": "Precio OMIE", "prediccion": "Modelo", "naive_d1": "Naive (Precio ayer)"})

st.line_chart(
    grafico_pvr,
    x="Hora",
    y=["Modelo", "Naive (Precio ayer)", "Precio OMIE"],
    color=["#4C9BE8", "#9AA0A6", "#F0A04B"]
)


if evaluacion is None:
    st.info("Aún no se ha ejecutado la evaluación diaria (14:30).")
else:
    filtro_evaluacion = (evaluacion.index.date >= inicio) & (evaluacion.index.date <= fin)
    mae_dias=evaluacion[filtro_evaluacion].rename(columns={"mae_modelo": "Modelo", "mae_naive": "Naive (Precio ayer)"})
    st.markdown("**Error medio diario (MAE, €/MWh)**: más bajo es mejor")
    mae_m = mae_dias["Modelo"].mean()
    mae_n = mae_dias["Naive (Precio ayer)"].mean()
    mae_dias = mae_dias.assign(Día=mae_dias.index.strftime("%d-%m"))
    st.bar_chart(
        mae_dias,
        x="Día",
        y=["Modelo", "Naive (Precio ayer)"],
        color=["#4C9BE8", "#9AA0A6"],
        stack=False,
    )
    st.caption(f"Periodo: modelo {mae_m:.1f} · naive {mae_n:.1f} €/MWh "
               f"({(mae_m / mae_n - 1) * 100:+.0f} %)")


# ─────────────────────────── Sección 3 · Salud del modelo ───────────────────────────
st.subheader("Salud del modelo")

if evaluacion is None:
    st.info("Aún no hay evaluación (se genera a las 14:30).")
else:
    ultimo = evaluacion.iloc[-1]
    n_dias = len(evaluacion)

    # 3a · Semáforo: el caso "pocos datos" va PRIMERO (con NaN no se puede afirmar nada).
    if pd.isna(ultimo["ratio_7d"]):
        st.info(f"La alarma necesita 5 días evaluados (hay {n_dias}). "
                "Hasta entonces no se puede afirmar nada sobre la salud del modelo.")
    elif ultimo["alarma"]:
        st.error(f"ALARMA: ratio o sesgo fuera de umbral {DIAS_SEGUIDOS} días seguidos. "
                 "Revisar: ¿cambio de nivel de precios? ¿datos? El reentreno semanal puede no bastar.")
    elif ultimo["condicion_alarma"]:
        st.warning(f"Fuera de umbral hoy (ratio > {UMBRAL_RATIO} o |sesgo| > {UMBRAL_SESGO} €). "
                   f"Vigilando: la alarma salta a los {DIAS_SEGUIDOS} días seguidos.")
    else:
        st.success("Modelo dentro de umbrales.")

    # 3b · Valores actuales con su umbral al lado.
    c1, c2 = st.columns(2)
    if pd.isna(ultimo["ratio_7d"]):
        c1.metric(f"Ratio MAE 7d modelo/naive (umbral {UMBRAL_RATIO})", "—")
        c2.metric(f"Sesgo 7d (umbral ±{UMBRAL_SESGO} €/MWh)", "—")
    else:
        c1.metric(f"Ratio MAE 7d modelo/naive (umbral {UMBRAL_RATIO})", f"{ultimo['ratio_7d']:.2f}")
        c2.metric(f"Sesgo 7d (umbral ±{UMBRAL_SESGO} €/MWh)", f"{ultimo['sesgo_7d']:+.1f} €/MWh")
    st.caption("Ratio < 1: el modelo se equivoca menos que el naive. "
               "Sesgo > 0: predice por encima del precio real; < 0: por debajo.")

    # 3c · Evolución con la línea de umbral (solo si ya hay algún valor).
    # asfreq("D") añade los días sin evaluación como NaN -> la línea se corta en los huecos.
    salud = evaluacion[["ratio_7d", "sesgo_7d"]].asfreq("D")
    if salud["ratio_7d"].notna().any():
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Ratio MAE 7d** (por encima de la línea = peor que el naive × umbral)")
            st.line_chart(
                salud[["ratio_7d"]].assign(umbral=UMBRAL_RATIO)
                .rename(columns={"ratio_7d": "Ratio 7d", "umbral": "Umbral"}),
                color=["#4C9BE8", "#E5534B"],
            )
        with g2:
            st.markdown("**Sesgo 7d (€/MWh)** (fuera de la banda = error sistemático)")
            st.line_chart(
                salud[["sesgo_7d"]].assign(sup=UMBRAL_SESGO, inf=-UMBRAL_SESGO)
                .rename(columns={"sesgo_7d": "Sesgo 7d", "sup": "Umbral +", "inf": "Umbral −"}),
                color=["#4C9BE8", "#E5534B", "#E5534B"],
            )
st.subheader("Negocio: ¿acierta las decisiones?")
st.caption(
    "Regret (K = 4): sobrecoste medio, en €/MWh, de operar en las 4 horas seleccionadas por el "
    "modelo frente a las 4 horas óptimas según el precio real (compra en las más baratas, venta en "
    "las más caras). Un valor de 0 indica una selección óptima. Referencia: misma selección "
    "realizada con el precio del día anterior (naive)."
)

# Zona de empate ±5 %: diferencias pequeñas se muestran en gris, sin color de "gana/pierde",
# para no pintar en rojo un empate (p. ej. 3,12 vs 3,11). Umbral de criterio, documentado en el diario.
EMPATE_PCT = 5


def _delta_vs_naive(modelo, naive):
    """Texto y color del delta del modelo frente al naive (menos regret = mejor)."""
    if naive <= 0:
        return None, "off"
    pct = (modelo / naive - 1) * 100
    if abs(pct) < EMPATE_PCT:
        return "≈ empate", "off"
    return f"{pct:+.0f} %", "inverse"


if evaluacion is None:
    st.info("Aún no hay evaluación (se genera a las 14:30).")
else:
    filtro_neg = (evaluacion.index.date >= inicio) & (evaluacion.index.date <= fin)
    neg = evaluacion[filtro_neg]
    venta_m = neg["regret_venta_k4"].mean()
    venta_n = neg["regret_venta_naive_k4"].mean()
    compra_m = neg["regret_compra_k4"].mean()
    compra_n = neg["regret_compra_naive_k4"].mean()

    delta_venta, color_venta = _delta_vs_naive(venta_m, venta_n)
    delta_compra, color_compra = _delta_vs_naive(compra_m, compra_n)

    c1, c2 = st.columns(2)
    c1.metric(f"Regret venta · naive {venta_n:.2f} €/MWh", f"{venta_m:.2f} €/MWh",
              delta=delta_venta, delta_color=color_venta)
    c2.metric(f"Regret compra · naive {compra_n:.2f} €/MWh", f"{compra_m:.2f} €/MWh",
              delta=delta_compra, delta_color=color_compra)

    # 4c · Regret de venta por día (modelo vs naive)
    st.markdown("**Regret de venta por día (€/MWh)**: más bajo es mejor")

    regret_dias = neg.rename(columns={
        "regret_venta_k4": "Modelo",
        "regret_venta_naive_k4": "Naive (Precio ayer)",
    })
    regret_dias = regret_dias.assign(Día=regret_dias.index.strftime("%m-%d"))

    st.bar_chart(
        regret_dias,
        x="Día",
        y=["Modelo", "Naive (Precio ayer)"],
        color=["#4C9BE8", "#9AA0A6"],
        stack=False,
    )
