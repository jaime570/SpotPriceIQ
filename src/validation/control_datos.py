

"""
Contrato de datos del warehouse (capa `processed`) del proyecto prediccion-electrica.

Valida la tabla maestra ya integrada y limpia antes de escribirla en data/processed/.
Se apoya en Pandera: se declara el esquema esperado (tipos, nulos, rangos físicos) y
se valida en modo `lazy` para recoger TODOS los incumplimientos en un solo informe.

Uso:
    from src.validation.control_datos import validar_warehouse
    df = validar_warehouse(df)   # devuelve el df si pasa; lanza SchemaErrors si no
"""

# Import clásico, compatible con todas las versiones de pandera.
# (En pandera >= 0.20 existe además el accesor `pandera.pandas`, pero esta
#  forma funciona igual y evita avisos del linter en versiones anteriores.)
import pandera as pa
from pandera import Column, Check, DataFrameSchema
import pandas as pd
from pathlib import Path
from datetime import date


# --------------------------------------------------------------------------- #
# Catálogos de columnas                                                        #
# --------------------------------------------------------------------------- #

# 16 estaciones AEMET (tienen tmed, tmax, velmedia, racha, prec)
ESTACIONES = [
    "alcazar_de_san_juan", "almudevar", "barcelona_aeropuerto", "carmona",
    "madrid_aeropuerto", "medina_de_pomar", "miranda_de_ebro",
    "navalmoral_de_la_mata", "puertollano", "san_pedro_manrique", "sarinena",
    "trujillo", "valencia_aeropuerto", "valmadrid", "zaragoza_aeropuerto",
    "ecija",
]

# Solo estas 6 estaciones miden insolación (columna sol_)
ESTACIONES_SOL = [
    "barcelona_aeropuerto", "madrid_aeropuerto", "medina_de_pomar",
    "navalmoral_de_la_mata", "valencia_aeropuerto", "zaragoza_aeropuerto",
]


# --------------------------------------------------------------------------- #
# Construcción de las columnas por bloques                                     #
# --------------------------------------------------------------------------- #

def _columnas_aemet() -> dict:
    """Genera las 86 columnas climáticas de AEMET por patrón de nombre.

    Reglas (todas nullable: los huecos se imputan en la fase de features):
      - tmed / tmax : temperatura plausible en España  [-30, 55] °C
      - velmedia    : velocidad media del viento        >= 0
      - racha       : racha máxima de viento            >= 0
      - prec        : precipitación                     >= 0
      - sol         : horas de sol                       [0, 24]  (solo 6 estaciones)
    """
    cols = {}
    for est in ESTACIONES:
        cols[f"tmed_{est}"] = Column(float, Check.in_range(-30, 55), nullable=True)
        cols[f"tmax_{est}"] = Column(float, Check.in_range(-30, 55), nullable=True)
        cols[f"velmedia_{est}"] = Column(float, Check.ge(0), nullable=True)
        cols[f"racha_{est}"] = Column(float, Check.ge(0), nullable=True)
        cols[f"prec_{est}"] = Column(float, Check.ge(0), nullable=True)
    for est in ESTACIONES_SOL:
        cols[f"sol_{est}"] = Column(float, Check.in_range(0, 24), nullable=True)
    return cols


# ESIOS del contrato: SOLO las previstas (forecasts). Las columnas `_real`
# (generación/demanda observadas) se excluyen del pipeline por ser leakage
# para el target, así que tampoco forman parte del contrato.
ESIOS_PREVISTAS = [
    "demanda_prevista", "demanda_prevista_diaria",
    "eolica_prevista", "eolica_prevista_d1",
    "solar_fv_prevista", "solar_termica_prevista",
]


def _columnas_esios() -> dict:
    """Genera las columnas de ESIOS (solo previstas, todas >= 0 y nullable)."""
    cols = {}
    for c in ESIOS_PREVISTAS:
        cols[c] = Column(float, Check.ge(0), nullable=True)
    return cols

# --------------------------------------------------------------------------- #
# Esquema completo del warehouse                                               #
# --------------------------------------------------------------------------- #

schema_warehouse = DataFrameSchema(
    {
        # --- Restricciones DURAS -------------------------------------------
        # Clave temporal: sin nulos y única (una fila por hora)
        "datetime_utc": Column("datetime64[ns, UTC]", nullable=False, unique=True),
        # Fecha local (día): sin nulos
        "fecha": Column("datetime64[ns]", nullable=False),
        # Target: sin nulos + rango físico generoso (deja margen a picos reales)
        "precio_espana": Column(float, Check.in_range(-500, 4000), nullable=False),

        # --- Rango físico (nullable) ---------------------------------------
        # Precio de Portugal (MIBEL): puede faltar en congestión / ingesta
        "precio_portugal": Column(float, Check.in_range(-500, 4000), nullable=True),

        # --- Bloques generados ---------------------------------------------
        **_columnas_esios(),
        **_columnas_aemet(),
    },
    strict=False,   # de momento no exigimos que estén SOLO estas columnas
    coerce=False,   # no convertir tipos en silencio: queremos enterarnos si difieren
)


# --------------------------------------------------------------------------- #
# API pública                                                                  #
# --------------------------------------------------------------------------- #

def validar_warehouse(df):
    """Valida la tabla del warehouse contra el contrato.

    Devuelve el DataFrame validado si cumple. Si hay incumplimientos, lanza
    `pandera.errors.SchemaErrors` con el informe completo (lazy=True recoge
    todos los fallos, no solo el primero).
    """
    return schema_warehouse.validate(df, lazy=True)


# --------------------------------------------------------------------------- #
# Contrato de COBERTURA (sobre las fuentes crudas de data/interim)             #
# --------------------------------------------------------------------------- #
# Responde a "¿hay suficientes datos y son frescos?" — complementario al schema
# de arriba, que valida el CONTENIDO de la maestra. Esto habría cazado el
# desplome de ESIOS a 97 filas (que pasaba la validación de contenido).

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"

# Suelos sacados del recuento real de cada tabla cruda (~75-80%, con margen).
# col_fecha = columna de fecha para el chequeo de frescura (None en OMIE: se
# construye desde ano/mes/dia). dias_frescura = antigüedad máxima tolerada del
# dato más reciente (AEMET publica con ~2 días de retraso, por eso es más laxo).
COBERTURA = {
    "tabla_OMIE":  {"min_filas": 40000, "col_fecha": None,           "dias_frescura": 3},
    "tabla_ESIOS": {"min_filas": 25000, "col_fecha": "datetime_utc", "dias_frescura": 3},
    "tabla_AEMET": {"min_filas": 15000, "col_fecha": "fecha",        "dias_frescura": 5},
}


def _fecha_maxima(df, col_fecha):
    """Fecha (naive, sin hora) más reciente de una tabla cruda."""
    if col_fecha is not None:
        serie = pd.to_datetime(df[col_fecha], utc=True, errors="coerce")
        return serie.max().tz_convert(None).normalize()
    # OMIE no tiene columna de fecha: se construye desde ano/mes/dia
    fechas = pd.to_datetime(df[["ano", "mes", "dia"]].rename(
        columns={"ano": "year", "mes": "month", "dia": "day"}))
    return fechas.max().normalize()


def validar_cobertura() -> dict:
    """Contrato de cobertura de las fuentes crudas (interim).

    Por cada fuente comprueba: (1) al menos `min_filas` filas — detecta un
    colapso tipo ESIOS-97; (2) que el dato más reciente no sea más viejo que
    `dias_frescura` días — detecta una fuente que dejó de actualizarse.

    Recoge TODOS los incumplimientos y, si hay alguno, lanza ValueError con el
    informe completo. Si todo pasa, devuelve {fuente: (n_filas, fecha_max)}.
    """
    hoy = pd.Timestamp(date.today())
    fallos = []
    resumen = {}
    for archivo, reglas in COBERTURA.items():
        ruta = INTERIM / archivo
        if not ruta.exists():
            fallos.append(f"{archivo}: no existe el fichero ({ruta})")
            continue

        df = pd.read_csv(ruta)
        n = len(df)
        fecha_max = _fecha_maxima(df, reglas["col_fecha"])
        resumen[archivo] = (n, fecha_max)

        if n < reglas["min_filas"]:
            fallos.append(f"{archivo}: {n} filas < mínimo {reglas['min_filas']}")

        antiguedad = (hoy - fecha_max).days
        if antiguedad > reglas["dias_frescura"]:
            fallos.append(
                f"{archivo}: dato más reciente {fecha_max.date()} "
                f"({antiguedad} días > {reglas['dias_frescura']} permitidos)")

    if fallos:
        raise ValueError("Contrato de cobertura incumplido:\n- " + "\n- ".join(fallos))
    return resumen