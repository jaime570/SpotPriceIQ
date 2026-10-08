"""Dashboard de SpotPriceIQ (solo lectura).

Lanzar desde la raíz del proyecto:  python -m streamlit run src/dashboard/app.py
(python -m añade la raíz al sys.path, por eso funciona `from src...`)."""

import streamlit as st
import pandas as pd


from src.dashboard.datos import (
    RUTA_EVALUACION, RUTA_REGISTRO, TABLA_FEATURES, cargar_datos, ultima_prediccion,predicho_vs_real,con_huecos
)

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
    
    st.bar_chart(
        mae_dias,
        y=["Modelo", "Naive (Precio ayer)"],
        color=["#4C9BE8", "#9AA0A6"],
        stack=False
    )
