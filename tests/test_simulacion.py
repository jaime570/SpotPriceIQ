import pandas as pd

from src.monitorizacion.simulacion import politica_mensual, politica_semanal


def test_politica_semanal_solo_reentrena_los_lunes():
    """weekday() empieza en lunes = 0: la política semanal (la de producción) reentrena el lunes
    y no el martes ni el domingo. Protege contra confundir el índice del día de la semana."""

    assert politica_semanal(pd.Timestamp("2026-10-19"), None, None) is True
    assert politica_semanal(pd.Timestamp("2026-10-20"), None, None) is False
    assert politica_semanal(pd.Timestamp("2026-10-25"), None, None) is False


def test_politica_mensual_solo_reentrena_el_dia_1():
    """La política mensual reentrena el día 1 de cada mes y no el último día del mes anterior."""
    
    assert politica_mensual(pd.Timestamp("2026-11-01"), None, None)  is True
    assert politica_mensual(pd.Timestamp("2026-10-31"), None, None) is False
    
