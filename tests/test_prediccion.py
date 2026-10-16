import pandas as pd
import pytest
from src.monitorizacion.prediccion import seleccionar_filas_d1


def test_seleccionar_filas_d1_falla_si_falta_una_feature():
    """Incidente del 12-oct: si ESIOS aún no ha publicado eolica_prevista_d1, las filas de mañana
    están incompletas y se lanza ValueError nombrando la columna, en vez de predecir con un hueco."""

    df = pd.DataFrame({
        "fecha": pd.to_datetime(["2026-10-17", "2026-10-17"]),
        "precio_espana": [float("nan"), float("nan")],      
        "completa_features": [True, False],
        "eolica_prevista_d1": [500.0, float("nan")],        
    })

    with pytest.raises(ValueError, match="eolica_prevista_d1"):
        seleccionar_filas_d1(df)


def test_seleccionar_filas_d1_devuelve_solo_las_de_manana():
    """Con un día pasado (con precio) y mañana completo (sin precio), devuelve solo las filas de mañana."""

    df = pd.DataFrame({
        "fecha": pd.to_datetime(["2026-10-16", "2026-10-17", "2026-10-17"]),
        "precio_espana": [120.0, float("nan"), float("nan")],   
        "completa_features": [True, True, True],
        "eolica_prevista_d1": [400.0, 500.0, 520.0],
    })

    resultado = seleccionar_filas_d1(df)

    assert len(resultado) == 2
    assert (resultado["fecha"] == "2026-10-17").all() 