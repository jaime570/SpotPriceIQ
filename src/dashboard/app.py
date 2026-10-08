"""Dashboard de SpotPriceIQ (solo lectura).

Lanzar desde la raíz del proyecto:  python -m streamlit run src/dashboard/app.py
(python -m añade la raíz al sys.path, por eso funciona `from src...`)."""

import streamlit as st
import pandas as pd


from src.dashboard.datos import (
    RUTA_EVALUACION, RUTA_REGISTRO, TABLA_FEATURES, cargar_datos, ultima_prediccion,
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

grafico = ult.rename(columns={"hora_madrid": "hora", "prediccion": "Modelo", "naive_d1": "Naive (Precio ayer)"})

st.line_chart(
    grafico,
    x="hora",
    y=["Modelo", "Naive (Precio ayer)"],
    color=["#4C9BE8", "#9AA0A6"],
)