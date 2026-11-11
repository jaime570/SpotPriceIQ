import pandas as pd

from src.ingesta.mibgas import parsear_texto, unir

# Mismo formato que el CSV real: línea de emisión, cabecera con la unidad en el precio,
# ';' al final de cada línea y otros productos mezclados con GDAES_D+1
CSV = (
    "Fecha Emisión :31/12/2023 - 21:33;\n"
    'Trading day;Product;Place of delivery;Area;First Day Delivery;Last Day Delivery;'
    '"MIBGAS Daily Price [EUR/MWh]";\n'
    "31/12/2023;GDAES_D+1;PVB;ES;01/01/2024;01/01/2024;31.18;\n"
    "30/12/2023;GDAES_D+1;PVB;ES;31/12/2023;31/12/2023;30.99;\n"
    "29/12/2023;GDAES_WE;PVB;ES;30/12/2023;31/12/2023;29.50;\n"
    "28/12/2023;GDAES_D+1;PVB;ES;29/12/2023;29/12/2023;;\n"
)


def test_parsear_filtra_producto_y_ordena():
    df = parsear_texto(CSV)
    assert list(df.columns) == ["fecha_entrega", "precio_gas"]
    # Solo GDAES_D+1 con precio; el de fin de semana y el vacío fuera
    assert len(df) == 2
    assert df["fecha_entrega"].is_monotonic_increasing
    assert df["fecha_entrega"].iloc[-1] == pd.Timestamp("2024-01-01")
    assert df["precio_gas"].iloc[-1] == 31.18


def test_unir_quita_duplicados_y_gana_la_nueva():
    previa = pd.DataFrame({"fecha_entrega": pd.to_datetime(["2026-11-01", "2026-11-02"]),
                           "precio_gas": [80.0, 81.0]})
    nueva = pd.DataFrame({"fecha_entrega": pd.to_datetime(["2026-11-02", "2026-11-03"]),
                          "precio_gas": [81.5, 82.0]})
    out = unir(previa, nueva)
    assert len(out) == 3
    assert out["fecha_entrega"].is_unique
    assert out.set_index("fecha_entrega").loc["2026-11-02", "precio_gas"] == 81.5
