import pandas as pd
import pytest
from src.dashboard.datos import con_huecos, _con_hora_madrid,ultima_prediccion,cargar_datos


@pytest.fixture
def dia_completo():
    return pd.DataFrame({"hora_madrid":pd.date_range("2026-10-01", periods=24, freq="h", tz="Europe/Madrid"),
                            "prediccion": range(24)
                            })

def test_con_huecos_rellena_horas_quitadas(dia_completo):
    quitadas =dia_completo[~dia_completo["prediccion"].isin([5,12,18])]
    resultado = con_huecos(quitadas)
    assert len(resultado) == 24
    assert resultado["prediccion"].isna().sum() == 3
    assert resultado["prediccion"][resultado["prediccion"].notna()].tolist() == quitadas["prediccion"].tolist()  
    assert resultado["hora_madrid"][resultado["prediccion"].isna()].dt.hour.tolist() == [5,12,18]

def test_con_huecos_sin_huecos_no_cambia(dia_completo):
    resultado = con_huecos(dia_completo)
    pd.testing.assert_frame_equal(resultado, dia_completo)

def test_con_huecos_no_extiende_los_extremo(dia_completo):
    sin_extremos = dia_completo[~dia_completo["prediccion"].isin([0,23])]
    resultado =con_huecos(sin_extremos)
    assert len(resultado) == 22
    assert resultado["prediccion"].isna().sum() == 0
    assert resultado["hora_madrid"].dt.hour.iloc[0] == 1
    assert resultado["hora_madrid"].dt.hour.iloc[-1] == 22

def test_cambio_hora_25_oct():
    df = pd.DataFrame({"datetime_utc": pd.date_range("2026-10-24 22:00", periods=25, freq="h", tz="UTC"),
                       "prediccion":range(25)})
    resultado =_con_hora_madrid(df)
    assert len(resultado) == 25
    assert (resultado["hora_madrid"].dt.hour== 2).sum() == 2
    sin_segunda_2 = resultado[~resultado["prediccion"].isin([3])]
    rellenado = con_huecos(sin_segunda_2)
    assert len(rellenado) == 25
    assert rellenado["prediccion"].isna().sum() == 1
    assert str(rellenado["hora_madrid"][rellenado["prediccion"].isna()].iloc[0]) == "2026-10-25 02:00:00+01:00"

def test_ultima_prediccion_registro_desordenado():
    registro = pd.DataFrame({
    "datetime_utc": pd.to_datetime([
        "2026-10-13 22:00", "2026-10-13 23:00",  
        "2026-10-09 22:00", "2026-10-09 23:00",   
        "2026-10-11 22:00", "2026-10-11 23:00",  
    ], utc=True),
    "fecha_objetivo": pd.to_datetime([
        "2026-10-14", "2026-10-14",
        "2026-10-10", "2026-10-10",
        "2026-10-12", "2026-10-12",
    ]),
    "prediccion": [140.0, 135.0, 160.0, 150.0, 90.0, 85.0],
    })
    resultado=ultima_prediccion(registro)

    assert len(resultado) == 2
    assert (resultado["fecha_objetivo"]== "2026-10-14").all()
    assert "hora_madrid" in resultado.columns

def test_cargar_datos_sin_evaluacion(tmp_path):                 
    registro = pd.DataFrame({
    "datetime_utc": pd.to_datetime(["2026-10-13 22:00", "2026-10-13 23:00"], utc=True),
    "fecha_objetivo": pd.to_datetime(["2026-10-14", "2026-10-14"]),
    "prediccion": [140.0, 135.0],
    })
    registro.to_parquet(tmp_path / "predicciones.parquet")

    features = pd.DataFrame({
    "datetime_utc": pd.to_datetime(["2026-10-13 22:00", "2026-10-13 23:00"], utc=True),
    "precio_espana": [138.5, 131.2],
    "otra": [1, 2],                      # columna que cargar_datos NO debe leer
    })
    features.to_parquet(tmp_path / "features.parquet")

    registro_leido, evaluacion, precio_real= cargar_datos(ruta_registro= tmp_path / "predicciones.parquet",
                                                         ruta_evaluacion= tmp_path / "no_existe.parquet",
                                                         ruta_features= tmp_path / "features.parquet",
                                                         )
    assert evaluacion is None
    assert list(precio_real.columns) == ["datetime_utc", "precio_espana"]
    assert len(registro_leido) == 2

def test_cargar_datos_con_evaluacion(tmp_path):
    registro = pd.DataFrame({
        "datetime_utc": pd.to_datetime(["2026-10-13 22:00", "2026-10-13 23:00"], utc=True),
        "fecha_objetivo": pd.to_datetime(["2026-10-14", "2026-10-14"]),
        "prediccion": [140.0, 135.0],
    })
    registro.to_parquet(tmp_path / "predicciones.parquet")

    features = pd.DataFrame({
        "datetime_utc": pd.to_datetime(["2026-10-13 22:00", "2026-10-13 23:00"], utc=True),
        "precio_espana": [138.5, 131.2],
    })
    features.to_parquet(tmp_path / "features.parquet")

    evaluacion_guardada = pd.DataFrame({"mae_modelo": [20.0]})
    evaluacion_guardada.to_parquet(tmp_path / "evaluacion.parquet")

    _, evaluacion, _ = cargar_datos(
        ruta_registro=tmp_path / "predicciones.parquet",
        ruta_evaluacion=tmp_path / "evaluacion.parquet",
        ruta_features=tmp_path / "features.parquet",
    )
    assert evaluacion is not None
    assert len(evaluacion) == 1