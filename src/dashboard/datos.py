from pathlib import Path

import pandas as pd

from src.monitorizacion.evaluacion import (
    RUTA_EVALUACION,
    RUTA_REGISTRO,
    TABLA_FEATURES,
    cruzar_con_real,
)


def _con_hora_madrid(df):
    """Añade `hora_madrid` (timestamp en Europe/Madrid) a partir de `datetime_utc`.

    Único sitio del dashboard donde se convierte la zona horaria. Se guarda en UTC
    (continuo, sin horas repetidas) y se muestra en hora peninsular. Al ser timestamp
    y no un entero de hora, la 02:00 repetida del 25-oct sale como dos puntos
    distintos (+02:00 y +01:00) y la 02:00 inexistente de marzo no deja hueco."""
    return df.assign(hora_madrid=df["datetime_utc"].dt.tz_convert("Europe/Madrid"))


def ultima_prediccion(registro):
    """Todas las filas del último día predicho (la fecha_objetivo más reciente del registro),
    con una columna `hora_madrid` en hora local de España.

    Usa .max() y no [-1] para no depender del orden del registro. El último día no siempre
    es mañana: si falta una predicción (hueco), devuelve el último día que sí se predijo."""
    ultimo_dia = registro[registro["fecha_objetivo"] == registro["fecha_objetivo"].max()]
    return _con_hora_madrid(ultimo_dia)


def predicho_vs_real(registro, precio_real):
    """Predicción, naive y precio real por hora, solo días completos, con `hora_madrid`.

    Reutiliza cruzar_con_real (la misma de la evaluación): no se duplica la lógica
    del cruce ni del manejo de días de 23/25 horas. Devuelve solo las columnas que
    pinta el gráfico de predicho vs real."""
    cruzado = cruzar_con_real(registro, precio_real)
    return _con_hora_madrid(cruzado)[
        ["hora_madrid", "fecha_objetivo", "prediccion", "naive_d1", "precio_real"]
    ]

def con_huecos(df):
    """Rellena con NaN las horas que faltan entre la primera y la última de `df`,
    para que los gráficos de líneas se CORTEN en los días sin predicción (p. ej.
    el 9-oct) en vez de unirlos con una recta que parecería una predicción inventada.

    El rango se genera con hora_madrid (zona horaria incluida): pd.date_range cuenta
    horas reales, así que el 25-oct salen 25 horas y en marzo 23, sin casos especiales.
    Requiere que `hora_madrid` no tenga duplicados (lo garantiza keep="first" del registro)."""
    
    df = df.set_index("hora_madrid")
    rango = pd.date_range(df.index.min(), df.index.max(), freq="h")
    return df.reindex(rango).reset_index(names="hora_madrid")


def cargar_datos(ruta_registro=RUTA_REGISTRO, ruta_evaluacion=RUTA_EVALUACION,
                 ruta_features=TABLA_FEATURES):
    """Lee las tres fuentes del dashboard (solo lectura, nunca escribe).

    Devuelve (registro, evaluacion, precio_real):
      - registro: predicciones.parquet tal cual se guardaron (point-in-time, keep="first").
      - evaluacion: evaluacion_diaria.parquet con índice de fecha, o None si aún no
        existe (antes de la primera evaluación de la tarde).
      - precio_real: solo `datetime_utc` y `precio_espana` de tabla_features; parquet
        es columnar, así que con columns= no se leen las ~100 columnas restantes.

    Las rutas son parámetros con las de producción por defecto, para poder pasar
    ficheros pequeños en los tests."""
    if Path(ruta_evaluacion).exists():
        evaluacion = pd.read_parquet(ruta_evaluacion)
    else:
        evaluacion = None
    registro = pd.read_parquet(ruta_registro)
    precio_real = pd.read_parquet(ruta_features, columns=["datetime_utc", "precio_espana"])
    return (registro, evaluacion, precio_real)


