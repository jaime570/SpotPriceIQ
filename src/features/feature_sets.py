"""
Selección de features por modelo — fuente única de verdad.

Se define por REGLAS (prefijos/sufijos de columna), no por listas hardcodeadas,
para que al añadir o quitar features el resto del proyecto lo herede solo.

Lo importan tanto los notebooks (features, modelo) como el pipeline de Prefect
y el serving, de modo que todos entrenan y predicen con el mismo conjunto.
"""

# Prefijos de las variables meteo de AEMET (ya imputadas), compartidas por ambos modelos.
_METEO_PREFIXES = ("tmed_", "tmax_", "velmedia_", "racha_", "prec_", "sol_")

TARGET = "precio_espana"


def get_meteo(df):
    """Columnas meteo AEMET presentes en df."""
    return df.columns[df.columns.str.startswith(_METEO_PREFIXES)].tolist()


def get_feature_sets(df):
    """
    Devuelve los sets de features por modelo a partir de las columnas de df.

    - predictivo (D+1): usa solo lo conocido por adelantado -> indicadores `_prevista`.
    - explicativo (ex-post): usa lo realmente medido -> indicadores `_real` (leakage
      para predecir, válidos solo para análisis a posteriori).

    Ambos comparten la meteo. Cuando se añadan features de calendario / lags,
    inclúyelas aquí (p. ej. sumándolas a `meteo`) para que las hereden los dos.

    Fuera de ambos: TARGET, `precio_portugal` (leakage MIBEL) y las columnas
    técnicas (datetime_utc, fecha, flags de completitud).
    """
    meteo = get_meteo(df)
    previstas = df.columns[df.columns.str.contains("_prevista")].tolist()
    reales = df.columns[df.columns.str.endswith("_real")].tolist()
    # Lags de precio: sin leakage (en D-1, tras la subasta, ya se conocen los
    # precios de ayer y de hace una semana). startswith evita capturar velmedia_*.
    lags = df.columns[df.columns.str.startswith(("precio_lag", "precio_media"))].tolist()
    # Calendario: solo codificacion ciclica (sin/cos) + festivo. Se descartan las
    # crudas (hora, dia, semana, mes) para no duplicar senal; ver diario 2026-08-30.
    calendario = df.columns[df.columns.str.endswith(("_sin", "_cos"))].tolist()
    if "festivo" in df.columns:
        calendario.append("festivo")

    return {
        "predictivo": previstas + meteo + lags + calendario,
        "explicativo": reales + meteo + lags + calendario,
    }
