import pandas as pd
import pytest

from src.ingesta.mibgas import parsear_texto, unir
from src.validation import control_datos

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


# --- Cobertura: caducidad del gas (0 días OK, 1–3 aviso, > 3 bloquea) ---

@pytest.fixture
def interim_mibgas(tmp_path, monkeypatch):
    """data/interim temporal con tabla_MIBGAS hasta el 10-11-2026 y solo la regla de MIBGAS."""
    fechas = pd.date_range(end="2026-11-10", periods=1500, freq="D")
    pd.DataFrame({"fecha_entrega": fechas.strftime("%Y-%m-%d"), "precio_gas": 80.0}).to_csv(
        tmp_path / "tabla_MIBGAS", index=False)
    monkeypatch.setattr(control_datos, "COBERTURA",
                        {"tabla_MIBGAS": control_datos.COBERTURA["tabla_MIBGAS"]})
    return tmp_path


def test_cobertura_mibgas_al_dia_sin_avisos(interim_mibgas):
    _, avisos = control_datos.validar_cobertura(hoy="2026-11-10", interim=interim_mibgas)
    assert avisos == []


def test_cobertura_mibgas_antiguedad_negativa_es_ok(interim_mibgas):
    # En la evaluación de la tarde ya puede estar publicada la entrega de mañana
    # (antigüedad −1): no es un dato viejo, no debe avisar ni bloquear
    _, avisos = control_datos.validar_cobertura(hoy="2026-11-09", interim=interim_mibgas)
    assert avisos == []


@pytest.mark.parametrize("hoy", ["2026-11-11", "2026-11-13"])
def test_cobertura_mibgas_1_a_3_dias_avisa(interim_mibgas, hoy):
    _, avisos = control_datos.validar_cobertura(hoy=hoy, interim=interim_mibgas)
    assert len(avisos) == 1
    assert "último precio" in avisos[0]


def test_cobertura_mibgas_mas_de_3_dias_bloquea(interim_mibgas):
    with pytest.raises(ValueError, match="tabla_MIBGAS"):
        control_datos.validar_cobertura(hoy="2026-11-14", interim=interim_mibgas)
