import pandas as pd

from src.features.feature_sets import get_feature_sets


def _df_columnas(cols):
    return pd.DataFrame(columns=cols)


# Previsiones ESIOS que NO pueden entrar al modelo:
#  - genéricas (544, 541): se refrescan intradía
#  - demanda diaria (460) y solares (542, 543): se revisan tras la subasta
#    (auditoría point-in-time, notebook 11 / diario 2026-09-29)
PREVISTAS_PROHIBIDAS = [
    "demanda_prevista", "eolica_prevista",
    "demanda_prevista_diaria", "solar_fv_prevista", "solar_termica_prevista",
]


def test_predictivo_usa_solo_previstas_verificadas():
    """Solo entra la eólica D+1 (estable tras la subasta); las genéricas y las
    que se revisan después (demanda, solares) quedan fuera, igual que el leakage."""
    df = _df_columnas([
        "precio_espana", "precio_portugal",
        "eolica_prevista_d1", *PREVISTAS_PROHIBIDAS,
        "demanda_real",
    ])
    feats = get_feature_sets(df)["predictivo"]

    assert "eolica_prevista_d1" in feats
    for c in PREVISTAS_PROHIBIDAS:
        assert c not in feats, f"{c} no debe estar en el predictivo"
    assert "demanda_real" not in feats
    assert "precio_espana" not in feats and "precio_portugal" not in feats


def test_predictivo_usa_meteo_retrasada():
    """La meteo AEMET del mismo día (observada, publicada con ~2 días de retraso)
    no entra al predictivo; solo su versión retrasada (_lag3d)."""
    df = _df_columnas([
        "precio_espana",
        "tmed_madrid_aeropuerto", "tmed_madrid_aeropuerto_lag3d",
        "prec_sevilla", "prec_sevilla_lag3d",
    ])
    feats = get_feature_sets(df)["predictivo"]

    assert "tmed_madrid_aeropuerto_lag3d" in feats and "prec_sevilla_lag3d" in feats
    assert "tmed_madrid_aeropuerto" not in feats and "prec_sevilla" not in feats
