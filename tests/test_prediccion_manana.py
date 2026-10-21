import pandas as pd

from src.orquestacion.prediccion_manana import MARGEN_INICIO, antes_de


def test_antes_de_compara_en_hora_de_madrid_en_verano():
    """A las 09:55 UTC son las 11:55 en Madrid (verano, +02:00): ya pasó el margen de las 11:50.
    Si comparara en UTC (09:55 < 11:50) dejaría pasar una predicción fuera de plazo."""

    ahora = pd.Timestamp("2026-10-16 09:55", tz="UTC")

    assert antes_de(MARGEN_INICIO, ahora=ahora) is False

def test_antes_de_dentro_de_plazo_en_invierno():
    """A las 10:45 UTC son las 11:45 en Madrid (invierno, +01:00): aún se puede predecir.
    Comprueba que tras el cambio de hora del 25-oct la conversión usa +01:00 y no +02:00."""

    ahora = pd.Timestamp("2026-10-26 10:45", tz="UTC")
    
    assert antes_de(MARGEN_INICIO, ahora=ahora) is True

def test_antes_de_fuera_de_plazo_en_invierno():
    """A las 10:55 UTC son las 11:55 en Madrid (invierno, +01:00): ya pasó el margen de las 11:50."""

    ahora = pd.Timestamp("2026-10-26 10:55", tz="UTC")
    
    assert antes_de(MARGEN_INICIO, ahora=ahora) is False
