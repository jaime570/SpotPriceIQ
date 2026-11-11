import numpy as np
import pandas as pd
import pytest

from src.procesamiento.concatenacion import pivotar
from src.procesamiento.etl import limpiar_aemet
from src.procesamiento.features import añadir_gas, gas_d1_diario, imputar_capa_1, marcar_entrenable


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
        "tmed_madrid_lag3d": [10.0, np.nan],
    })
    out = marcar_entrenable(df)
    assert out["entrenable"].iloc[0]
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


# --- gas_d1 (MIBGAS) ---

def _tabla_gas(fechas, precios):
    return pd.DataFrame({"fecha_entrega": pd.to_datetime(fechas), "precio_gas": precios})


def test_gas_d1_usa_la_entrega_del_dia_anterior():
    gas = gas_d1_diario(_tabla_gas(["2026-11-01", "2026-11-02"], [80.0, 81.0]), hasta="2026-11-03")
    assert gas[pd.Timestamp("2026-11-02")] == 80.0   # día T = entrega T-1
    assert gas[pd.Timestamp("2026-11-03")] == 81.0
    assert pd.Timestamp("2026-11-01") not in gas.index   # necesitaría la entrega del 31-10


def test_gas_d1_llega_hasta_el_dia_que_se_predice_y_rellena_huecos():
    # Falta la entrega del 03-11 (hueco) y no hay nada publicado después del 04-11
    tabla = _tabla_gas(["2026-11-01", "2026-11-02", "2026-11-04"], [80.0, 81.0, 84.0])
    gas = gas_d1_diario(tabla, hasta="2026-11-07")
    assert gas.index.max() == pd.Timestamp("2026-11-07")   # el calendario no se para en el último dato
    assert gas.notna().all()
    assert gas[pd.Timestamp("2026-11-04")] == 81.0          # hueco: último precio disponible
    assert gas[pd.Timestamp("2026-11-07")] == 84.0          # días sin publicar: último precio


def test_gas_d1_ignora_entregas_posteriores_al_dia_que_se_predice():
    # Por la tarde ya puede estar publicada la entrega de mañana: no debe colarse
    gas = gas_d1_diario(_tabla_gas(["2026-11-01", "2026-11-02", "2026-11-03"], [80.0, 81.0, 99.0]),
                        hasta="2026-11-03")
    assert gas.index.max() == pd.Timestamp("2026-11-03")
    assert gas[pd.Timestamp("2026-11-03")] == 81.0


def test_añadir_gas_une_por_dia_de_madrid():
    # 29-03-2026: cambio de hora, día de 23 horas. 28-03 23:00 UTC ya es el 29 en Madrid
    utc = pd.date_range("2026-03-28 00:00", "2026-03-29 21:00", freq="h", tz="UTC")
    df = pd.DataFrame({"datetime_utc": utc})
    df["fecha"] = df["datetime_utc"].dt.tz_convert("Europe/Madrid").dt.normalize().dt.tz_localize(None)
    tabla = _tabla_gas(["2026-03-27", "2026-03-28"], [70.0, 75.0])

    out = añadir_gas(df, tabla)
    assert len(out) == len(df)
    assert out.loc[out["datetime_utc"] == pd.Timestamp("2026-03-28 22:00", tz="UTC"), "gas_d1"].item() == 70.0
    assert out.loc[out["datetime_utc"] == pd.Timestamp("2026-03-28 23:00", tz="UTC"), "gas_d1"].item() == 75.0
    assert (out.loc[out["fecha"] == pd.Timestamp("2026-03-29"), "gas_d1"] == 75.0).sum() == 23
