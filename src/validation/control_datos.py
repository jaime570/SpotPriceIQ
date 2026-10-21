

"""
Contrato de datos del warehouse (capa `processed`) del proyecto prediccion-electrica.

Valida la tabla maestra ya integrada y limpia antes de escribirla en data/processed/.
Se apoya en Pandera: se declara el esquema esperado (tipos, nulos, rangos físicos) y
se valida en modo `lazy` para recoger TODOS los incumplimientos en un solo informe.

Uso:
    from src.validation.control_datos import validar_warehouse
    df = validar_warehouse(df)   # devuelve el df si pasa; lanza SchemaErrors si no
"""

from datetime import date
from pathlib import Path

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

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
def _precio_nulo_solo_en_cola(df):
    """Permite precio_espana nulo SOLO en la cola (horas futuras a predecir):
    todo precio nulo debe ser posterior al último precio conocido.
    Un nulo en mitad del histórico (hueco de OMIE) hace fallar el contrato."""
    
    ultima_fecha_precio_conocido = df.loc[df["precio_espana"].notna(), "datetime_utc"].max()
    fechas_sin_precio = df.loc[df["precio_espana"].isna(), "datetime_utc"]
    return (fechas_sin_precio > ultima_fecha_precio_conocido).all()

schema_warehouse = DataFrameSchema(
    {
        # --- Restricciones DURAS -------------------------------------------
        # Clave temporal: sin nulos y única (una fila por hora)
        "datetime_utc": Column("datetime64[ns, UTC]", nullable=False, unique=True),
        # Fecha local (día): sin nulos
        "fecha": Column("datetime64[ns]", nullable=False),
        # Target: solo nulos en cola + rango físico generoso (deja margen a picos reales)
        "precio_espana": Column(float, Check.in_range(-500, 4000), nullable=True,),

        # --- Rango físico (nullable) ---------------------------------------
        # Precio de Portugal (MIBEL): puede faltar en congestión / ingesta
        "precio_portugal": Column(float, Check.in_range(-500, 4000), nullable=True),

        # --- Bloques generados ---------------------------------------------
        **_columnas_esios(),
        **_columnas_aemet(),
    },
    strict=False,   # de momento no exigimos que estén SOLO estas columnas
    coerce=False,# no convertir tipos en silencio: queremos enterarnos si difieren
    checks=[Check(_precio_nulo_solo_en_cola,                                         
                  error="precio_espana nulo fuera de la cola (hueco en el histórico)")],
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

# bloquea = si la fuente está desactualizada, ¿se para el flujo?
# AEMET no bloquea: la meteo pesa poco en el modelo (SHAP), entra retrasada
# 3 días y, si falta, la cubren las capas de imputación. Solo se avisa.
COBERTURA = {
    "tabla_OMIE":  {"min_filas": 40000, "col_fecha": None,           "dias_frescura": 3, "bloquea": True},
    "tabla_ESIOS": {"min_filas": 25000, "col_fecha": "datetime_utc", "dias_frescura": 3, "bloquea": True},
    "tabla_AEMET": {"min_filas": 15000, "col_fecha": "fecha",        "dias_frescura": 5, "bloquea": False},
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


def validar_cobertura() -> tuple[dict, list]:
    """Contrato de cobertura de las fuentes crudas (interim).

    Por cada fuente comprueba: (1) al menos `min_filas` filas — detecta un
    colapso tipo ESIOS-97 (bloquea siempre); (2) que el dato más reciente no sea
    más viejo que `dias_frescura` días — detecta una fuente que dejó de
    actualizarse. Si la fuente tiene `bloquea=False`, (2) solo genera un aviso.

    Si hay fallos bloqueantes, lanza ValueError con el informe completo.
    Si no, devuelve ({fuente: (n_filas, fecha_max)}, [avisos]).
    """
    hoy = pd.Timestamp(date.today())
    fallos = []
    avisos = []
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
            mensaje = (f"{archivo}: dato más reciente {fecha_max.date()} "
                       f"({antiguedad} días > {reglas['dias_frescura']} permitidos)")
            if reglas["bloquea"]:
                fallos.append(mensaje)
            else:
                avisos.append(mensaje)

    if fallos:
        raise ValueError("Contrato de cobertura incumplido:\n- " + "\n- ".join(fallos))
    return resumen, avisos

