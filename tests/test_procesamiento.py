import pandas as pd
import numpy as np 
from src.procesamiento.etl import limpiar_aemet
import pytest
from src.procesamiento.concatenacion import pivotar
from src.procesamiento.features import marcar_entrenable,imputar_capa_1

def test_limpiar_aemet_centinelas_y_comas():
    df = pd.DataFrame({"prec_madrid": ['12,2', 'Ip', 'Acum', '0,0']})
    out = limpiar_aemet(df)
    assert out["prec_madrid"].dtype == float
    assert out["prec_madrid"].iloc[0] == 12.2
    assert out["prec_madrid"].iloc[1] == 0.0
    assert pd.isna(out["prec_madrid"].iloc[2])


def test_pivotar_falla_con_duplicados():
    df = pd.DataFrame({"fecha": ["2023-01-01", "2023-01-01"], "nombre": ["Madrid", "Madrid"],
                       "tmed": 15.0, "tmax": 20.0, "velmedia": 5.0, "racha": 10.0,
                       "sol": 8.0, "prec": 0.0})
    with pytest.raises(ValueError):
        pivotar(df)

def test_marcar_entrenable():
    df = pd.DataFrame({
        "datetime_utc": ["2023-01-01 00:00", "2023-01-01 01:00"],
        "fecha": ["2023-01-01", "2023-01-01"],
        "precio_espana": [50.0, 55.0],
        "precio_portugal": [48.0, 52.0],
        "tmed_madrid": [10.0, np.nan],
    })
    out = marcar_entrenable(df)
    assert out["entrenable"].iloc[0] == True
    assert not out["entrenable"].iloc[1]

def test_imputar_capa1_rellena_hueco_corto():
    df = pd.DataFrame({
        "datetime_utc": pd.to_datetime(
            ["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 02:00"], utc=True),
        "precio_espana": [50.0, 55.0, 60.0],
        "demanda_prevista": [100.0, np.nan, 200.0],
    })
    out = imputar_capa_1(df)
    assert out["demanda_prevista"].iloc[1] == 150.0
