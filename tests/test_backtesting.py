import pandas as pd

from src.evaluacion.backtesting import meses_madrid


def test_meses_madrid_corta_por_el_mes_de_madrid_no_el_utc():
    utc = pd.Series(pd.to_datetime(["2026-10-31 22:00", "2026-10-31 23:00"], utc=True))
    meses = meses_madrid(utc)
    # 22:00 UTC = 23:00 en Madrid (octubre); 23:00 UTC = 00:00 del 1 de noviembre
    assert list(meses.astype(str)) == ["2026-10", "2026-11"]
