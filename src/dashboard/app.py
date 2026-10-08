"""Dashboard de SpotPriceIQ (solo lectura).

Lanzar desde la raíz del proyecto:  python -m streamlit run src/dashboard/app.py
(python -m añade la raíz al sys.path, por eso funciona `from src...`)."""

import streamlit as st

from src.dashboard.datos import (
    RUTA_EVALUACION, RUTA_REGISTRO, TABLA_FEATURES, cargar_datos,
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