from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from src.features.feature_sets import get_feature_sets, get_meteo

PATH = Path(__file__).resolve().parents[2]
PROCESSED = PATH / "data" / "processed"
TABLA_MAESTRA_PROCESADA = PROCESSED / "tabla_maestra_procesada.parquet"
TABLA_MIBGAS = PATH / "data" / "interim" / "tabla_MIBGAS"

def imputar_capa_1 (df_maestra_procesada, columnas=None):
    """Imputacion de microhuecos < 6 horas en capa 1 (interpolacion lineal temporal)"""

    df = df_maestra_procesada.copy()
    if columnas is None:
            columnas = df.select_dtypes(include='number').columns.drop('precio_espana')
    df = df.set_index("datetime_utc")
    df[columnas] = df[columnas].interpolate(method='time', limit=6, limit_area='inside')
    df = df.reset_index()
    return df

def imputar_donante(df, receptora, candidatas, umbral = 0.8, solape_minimo = 1000):
    """imputacion de huecos > 6 horas en capa 2 (donante con offset mensual)"""

    df = df.copy()
    candidatas = candidatas.drop(receptora)
    ranking = df[candidatas].corrwith(df[receptora]).sort_values(ascending = False)
    donante = None
    mascara_receptora = df[receptora].isna()
    n_huecos = mascara_receptora.sum()
    for candidato in ranking.index:
        solape = (df[receptora].notna() & df[candidato].notna()).sum()
        if solape < solape_minimo:
            continue
        coinciden = df.loc[mascara_receptora, candidato].isna().sum()
        cobertura = 1 - coinciden / n_huecos
        if cobertura > umbral :
            donante = candidato
            break
    if donante is None:
        return df
    else:
        meses = df['datetime_utc'].dt.month
        resta = df[receptora]-df[donante]
        offset_mensual = resta.groupby(meses).mean()
        offset_fila = meses.map(offset_mensual)
        donante_ajustado = df[donante] + offset_fila
        df[receptora] = df[receptora].fillna(donante_ajustado)
        return df

def imputar_todas_donante(df, variables, umbral=0.8, solape_minimo=1000):
    """Imputacion de huecos > 6 horas en capa 2 (donante con offset mensual) para todas las variables"""

    for var in variables:
        columnas_var = df.columns[df.columns.str.startswith(var)]
        for receptora in columnas_var:
            if df[receptora].isna().sum() > 0:
                df = imputar_donante(df, receptora, columnas_var, umbral, solape_minimo)
    return df

def imputar_climatologia (df, columnas):
    """Imputacion de huecos > 6 horas en capa 3 (media mensual)"""

    df = df.copy()
    meses = df['datetime_utc'].dt.month 
    for columna in columnas:
        media =df[columna].groupby(meses).mean()
        relleno = meses.map(media)
        df[columna] = df[columna].fillna(relleno)
    return df

def añadir_calendario(df):
    """Añade columnas de calendario al dataframe"""

    df = df.copy()

    festivos_es = holidays.Spain(years=range(2022, 2027))
    local = df['datetime_utc'].dt.tz_convert('Europe/Madrid')
    ciclos = {"hora": 24, "semana": 7, "mes": 12}

    df['hora'] = local.dt.hour
    df['dia'] = local.dt.day
    df['semana'] = local.dt.dayofweek
    df['mes'] = local.dt.month
    df['festivo'] = local.dt.date.isin(set(festivos_es)).astype(int)
    
    for col, periodo in ciclos.items():
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / periodo)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / periodo)
    return df

def añadir_lags_precio(df):
    """Añade columnas de lags y medias móviles del precio"""

    df=df.copy()
    precio = df.set_index('datetime_utc')['precio_espana']
    df['precio_lag_24']  = (df['datetime_utc'] - pd.Timedelta(hours=24)).map(precio)
    df['precio_lag_168'] = (df['datetime_utc'] - pd.Timedelta(hours=168)).map(precio)
    media_24 = precio.rolling('24h').mean()
    df['precio_media_24'] = (df['datetime_utc'] - pd.Timedelta(hours=24)).map(media_24)
    return df

def añadir_lags_meteo(df,dias=3):
    df = df.copy()
    columnas = get_meteo(df)
    diario =df.groupby("fecha")[columnas].mean().reset_index()
    diario["fecha"] = diario["fecha"] + pd.Timedelta(days=dias)
    nombres_nuevos = { c: f"{c}_lag{dias}d" for c in columnas } 
    diario = diario.rename(columns =nombres_nuevos)
    df = df.merge(diario, on="fecha", how="left")
    return df

def gas_d1_diario(tabla_gas, hasta):
    """Serie diaria de gas_d1 indexada por fecha objetivo (día de Madrid, naive).

    Point-in-time: gas_d1 del día T = precio con entrega T−1, publicado al cerrar
    la sesión de T−2, antes de la subasta de las 12:00 de T−1.

    El calendario va desde la primera fecha objetivo hasta `hasta` (el último día
    de la tabla horaria, es decir, el día que se predice), NO hasta el último dato
    publicado: así el relleno hacia delante también cubre los días sin publicar.
    Los huecos del histórico se rellenan igual, para que entrenamiento y predicción
    traten el gas de la misma forma. La caducidad (más de 3 días sin dato nuevo
    bloquea) la controla la cobertura, no esta función.
    """
    gas = tabla_gas.set_index("fecha_entrega")["precio_gas"].sort_index()
    gas.index = gas.index + pd.Timedelta(days=1)      # fecha de entrega -> fecha objetivo
    calendario = pd.date_range(gas.index.min(), pd.Timestamp(hasta), freq="D", name="fecha")
    # reindex descarta lo posterior a `hasta` (p. ej. la entrega de mañana, publicada por la tarde)
    return gas.reindex(calendario).ffill().rename("gas_d1")


def añadir_gas(df, tabla_gas):
    """Añade gas_d1 a la tabla horaria uniendo por `fecha` (día de Madrid, regla 4),
    no por la fecha UTC: las 24 horas de un día de mercado comparten el mismo gas."""
    n_antes = len(df)
    serie = gas_d1_diario(tabla_gas, hasta=df["fecha"].max())
    df = df.merge(serie.reset_index(), on="fecha", how="left")
    # how="left" con una fila por día: no se puede ganar ni perder ninguna hora
    assert len(df) == n_antes, f"añadir_gas: {n_antes} filas antes y {len(df)} después"
    return df


def marcar_entrenable(df):

    """Marca las filas entrenables y completas de features"""
    df = df.copy()
    lista_features = get_feature_sets(df)["predictivo"]
    df["completa_features"] = df[lista_features].notna().all(axis=1)
    df["entrenable"] = df["completa_features"] & df["precio_espana"].notna()
    return df

def ejecutar_features():
    """Orquesta el pipeline de features: imputa (capa1 -> donante -> climatologia),
    anade calendario y lags, marca filas entrenables y guarda tabla_features.parquet."""
    df = pd.read_parquet(TABLA_MAESTRA_PROCESADA)

    # --- Imputacion ---
    df = imputar_capa_1(df)                                     # capa 1: micro-huecos <=6h
    df = imputar_todas_donante(df, ["tmed_", "tmax_"])         # capa 2: donante (solo temperatura)
    cols_temp = df.columns[df.columns.str.startswith(("tmed_", "tmax_"))]
    df[cols_temp] = df[cols_temp].ffill().bfill()              # residual de temperatura
    cols_clima = df.columns[df.columns.str.startswith(("velmedia", "racha_", "prec_", "sol_"))]
    df = imputar_climatologia(df, cols_clima)                  # capa 3: climatologia (viento/lluvia/sol)

    # --- Feature engineering ---
    df = añadir_calendario(df)
    df = añadir_lags_precio(df)
    df= añadir_lags_meteo(df)
    df = añadir_gas(df, pd.read_csv(TABLA_MIBGAS, parse_dates=["fecha_entrega"]))
    df = marcar_entrenable(df)                                # ultimo: despues de los lags

    df.to_parquet(PROCESSED / "tabla_features.parquet", index=False)
    return df


if __name__ == "__main__":
    ejecutar_features()
