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

# ESIOS que NO pueden ser negativas (generación / demanda)
ESIOS_NO_NEGATIVAS = [
    "demanda_prevista", "demanda_prevista_diaria",
    "eolica_prevista", "eolica_prevista_d1",
    "solar_fv_prevista", "solar_termica_prevista",
    "demanda_real", "eolica_real",
    "solar_fv_real",
    "ciclo_combinado_real", "nuclear_real",
]

# ESIOS que SÍ pueden ser negativas -> sin cota inferior
#   intercambios_real:  importación (-) / exportación (+)
#   hidraulica_real:    negativa cuando hay bombeo (consumo)
#   solar_termica_real: negativa de noche por consumos auxiliares (mantener el fluido)
ESIOS_CON_NEGATIVOS = ["hidraulica_real", "intercambios_real", "solar_termica_real"]


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


def _columnas_esios() -> dict:
    """Genera las 14 columnas de ESIOS (generación, demanda, intercambios)."""
    cols = {}
    for c in ESIOS_NO_NEGATIVAS:
        cols[c] = Column(float, Check.ge(0), nullable=True)
    for c in ESIOS_CON_NEGATIVOS:
        cols[c] = Column(float, nullable=True)  # sin cota inferior
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
