import pandas as pd
import pytest

from src.monitorizacion.evaluacion import (
    _alarma_por_dia,
    calcular_metricas,
    cruzar_con_real,
    detectar_degradacion,
    metricas_dia,
    regret_dia,
)


@pytest.fixture
def siete_dias_modelo_peor():
    """7 días × 4 horas; el modelo se pasa 10 € y el naive 5 € en todas las horas
    → ratio 7d 2,0, sesgo 7d +10 y alarma solo el día 7 (días 5, 6 y 7 fuera de umbral)."""
    cruzado = pd.DataFrame({
        "fecha_objetivo": pd.date_range("2026-10-01", periods=7).repeat(4),
        "precio_real": [10.0, 20.0, 30.0, 40.0] * 7,
    })
    cruzado["prediccion"] = cruzado["precio_real"] + 10
    cruzado["naive_d1"] = cruzado["precio_real"] + 5
    return cruzado

def test_cruzar_con_real_deja_solo_dias_completos():
    """Un día con horas sin precio real se descarta: si no, su MAE se calcularía con menos horas y falsearía el ratio 7d y la alarma."""
    
    registro = pd.DataFrame({
        "datetime_utc": pd.to_datetime([
            "2026-10-13 22:00", "2026-10-13 23:00",   
            "2026-10-14 22:00", "2026-10-14 23:00",   
        ], utc=True),
        "fecha_objetivo": pd.to_datetime(["2026-10-14", "2026-10-14", "2026-10-15", "2026-10-15"]),
        "prediccion": [100.0, 90.0, 80.0, 70.0],
    })
    tabla = pd.DataFrame({
        "datetime_utc": pd.to_datetime([
            "2026-10-13 22:00", "2026-10-13 23:00",
            "2026-10-14 22:00",                      
        ], utc=True),
        "precio_espana": [105.0, 88.0, 75.0],
    })

    resultado = cruzar_con_real(registro, tabla)

    assert len(resultado) == 2
    assert (resultado["fecha_objetivo"] == "2026-10-14").all()
    assert resultado["precio_real"].isna().sum() == 0

def test_cruzar_con_real_conserva_dia_de_25_horas():
    """El 25-oct (cambio de hora) tiene 25 horas: se compara contra las horas predichas, no contra 24, y no se descarta."""
    
    horas = pd.date_range("2026-10-24 22:00", periods=25, freq="h", tz="UTC")
    registro = pd.DataFrame({
        "datetime_utc": horas,
        "fecha_objetivo": pd.Timestamp("2026-10-25"),
        "prediccion": range(25),
    })
    tabla = pd.DataFrame({
        "datetime_utc": horas,
        "precio_espana": range(25),
    })

    resultado = cruzar_con_real(registro, tabla)

    assert len(resultado) == 25

def test_metricas_dia_con_numeros_a_mano():
    """MAE, sesgo, MAE de forma y Spearman calculados a mano en un día de 4 horas: son las métricas del TFM y de la alarma."""
    
    dia = pd.DataFrame({
        "precio_real": [10.0, 20.0, 30.0, 40.0],
        "prediccion":  [12.0, 18.0, 34.0, 40.0],   
        "naive_d1":    [15.0, 25.0, 35.0, 45.0],   
    })

    resultado = metricas_dia(dia)

    assert resultado["mae_modelo"] == 2
    assert resultado["mae_naive"] == 5
    assert resultado["sesgo"] == 1
    assert resultado["mae_forma"] == 2
    assert resultado["spearman"] == 1

def test_regret_dia_modelo_y_naive():
    """Regret = lo que pierdes al elegir horas con tu predicción frente a conocer el precio real.
    Mira el ORDEN, no el valor: el modelo falla todos los precios pero elige bien (regret 0);
    el naive elige mal la hora barata y paga 50 en vez de 10 (regret de compra 40)."""
    
    dia = pd.DataFrame({
        "precio_real": [10.0, 50.0, 30.0, 80.0],   
        "prediccion":  [12.0, 45.0, 35.0, 70.0],   
        "naive_d1":    [40.0, 20.0, 30.0, 60.0],   
    })

    resultado_modelo = regret_dia(dia, k=1)
    resultado_naive = regret_dia(dia, col="naive_d1", k=1)

    assert resultado_modelo["regret_compra"] == 0
    assert resultado_naive["regret_compra"] ==40
    assert resultado_modelo["regret_venta"] == 0
    assert resultado_naive["regret_venta"] == 0

def test_alarma_solo_con_3_dias_seguidos():
    """La alarma exige 3 días evaluados seguidos fuera de umbral: 2 no bastan, un día bueno
    reinicia la racha y un ratio NaN (sin 5 días para la media 7d) no cuenta como condición."""
    
    ratio_7d = pd.Series([float("nan"), 1.5, 1.5, 1.0, 1.5, 1.5, 1.5])
    sesgo_7d = pd.Series([0.0] * 7)  

    condicion,alarma  = _alarma_por_dia(ratio_7d, sesgo_7d)

    assert condicion.tolist() == [False, True, True, False, True, True, True]
    assert alarma.tolist() == [False, False, False, False, False, False, True]

def test_alarma_por_sesgo_en_las_dos_direcciones():
    """El sesgo cuenta en valor absoluto: quedarse corto 15 € es tan grave como pasarse
    (caso 15-oct, sesgo −23). Un sesgo de +13 está dentro de umbral y no cuenta."""
    
    ratio_7d = pd.Series([1.0] * 4)   # ratio dentro de umbral: aquí solo prueba el sesgo
    sesgo_7d = pd.Series([-15.0, -15.0, -15.0, 13.0])

    condicion,alarma  = _alarma_por_dia(ratio_7d, sesgo_7d)

    assert condicion.tolist() == [True, True, True, False]
    assert alarma.tolist() == [False, False , True, False]

def test_calcular_metricas_columnas_y_alarma_sin_datos_suficientes():
    """Une todas las piezas: una fila por día, las columnas que leen el dashboard y Telegram,
    y con solo 2 días el ratio 7d es NaN y no hay alarma (min_periods=5)."""
    
    cruzado = pd.DataFrame({
        "fecha_objetivo": pd.to_datetime(["2026-10-14"] * 4 + ["2026-10-15"] * 4),
        "prediccion":  [12.0, 45.0, 35.0, 70.0] * 2,   # el mismo día que en test_regret_dia, dos veces
        "naive_d1":    [40.0, 20.0, 30.0, 60.0] * 2,
        "precio_real": [10.0, 50.0, 30.0, 80.0] * 2,
    })

    resultado = calcular_metricas(cruzado)

    assert len(resultado) == 2
    assert "mae_modelo" in resultado.columns
    assert "mae_naive" in resultado.columns
    assert "ratio_7d" in resultado.columns
    assert "sesgo_7d" in resultado.columns
    assert resultado["ratio_7d"].isna().all()
    assert (~resultado["alarma"]).all()
    assert resultado["regret_compra_naive_k1"].tolist() == [40.0, 40.0]

def test_detectar_degradacion_coincide_con_calcular_metricas(siete_dias_modelo_peor):
    """Simulación (detectar_degradacion) y producción (calcular_metricas) deben dar la misma salud
    el último día: si alguien cambia el cálculo en una ruta y no en la otra, este test lo detecta."""

    produccion = calcular_metricas(siete_dias_modelo_peor)
    ultimo = produccion.iloc[-1]
    simulacion = detectar_degradacion(siete_dias_modelo_peor)
    assert ultimo["alarma"] == simulacion["alarma"]
    assert ultimo["ratio_7d"] == simulacion["ratio_7d"]
    assert ultimo["sesgo_7d"] == simulacion["sesgo_7d"]
    assert simulacion["alarma"] is True
    assert simulacion["ratio_7d"] == 2.0
    assert simulacion["sesgo_7d"] == 10.0

def test_detectar_degradacion_respeta_umbral_pasado(siete_dias_modelo_peor):
    """Si se pasa otro umbral (como hace politica_trigger), llega hasta la regla de la alarma:
    con ratio 2,0 y umbral 2,5 no hay alarma. Detecta pasar la constante en vez del parámetro."""

    simulacion = detectar_degradacion(siete_dias_modelo_peor, umbral=2.5)

    assert simulacion["alarma"] is False
    assert simulacion["ratio_7d"] == 2.0
    

    


      