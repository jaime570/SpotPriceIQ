import pandas as pd
import numpy as np
from pathlib import Path
import holidays

PATH = Path(__file__).resolve().parents[2]
PROCESSED = PATH / "data" / "processed"
TABLA_MAESTRA_PROCESADA = PROCESSED / "tabla_maestra_procesada.parquet"

def imputar_capa_1 (df_maestra_procesada, columnas=None):
    """Imputacion de microhuecos < 6 horas en capa 1 (interpolacion lineal temporal)"""

    df = df_maestra_procesada.copy()
    if columnas is None:
            columnas = df.select_dtypes(include='number').columns.drop('precio_espana')
    df = df.set_index("datetime_utc")
    df[columnas] = df[columnas].interpolate(method='time', limit=6, limit_area='inside')
    df = df.reset_index()
    return df

def imputar_donante(df, receptora, candidatas, umbral = 0.8):
    """imputacion de huecos > 6 horas en capa 2 (donante con offset mensual)"""

    df = df.copy()
    candidatas = candidatas.drop(receptora)
    ranking = df[candidatas].corrwith(df[receptora]).sort_values(ascending = False)
    donante = None
    mascara_receptora = df[receptora].isna()
    n_huecos = mascara_receptora.sum()
    for candidato in ranking.index:
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

def imputar_todas_donante(df, variables, umbral=0.8):
    """Imputacion de huecos > 6 horas en capa 2 (donante con offset mensual) para todas las variables"""

    for var in variables:
        columnas_var = df.columns[df.columns.str.startswith(var)]
        for receptora in columnas_var:
            if df[receptora].isna().sum() > 0:
                df = imputar_donante(df, receptora, columnas_var, umbral)
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

def marcar_entrenable(df):

    """Marca las filas entrenables y completas de features"""
    df = df.copy()
    excluir = ["precio_espana", "precio_portugal", "datetime_utc", "fecha"]
    excluir += df.columns[df.columns.str.endswith("_real")].tolist()
    features_validas = df.columns.drop(excluir)
    df["completa_features"] = df[features_validas].notna().all(axis=1)
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
    cols_clima = df.columns[df.columns.str.startswith(("vel", "racha_", "prec", "sol"))]
    df = imputar_climatologia(df, cols_clima)                  # capa 3: climatologia (viento/lluvia/sol)

    # --- Feature engineering ---
    df = añadir_calendario(df)
    df = añadir_lags_precio(df)
    df = marcar_entrenable(df)                                 # ultimo: despues de los lags

    df.to_parquet(PROCESSED / "tabla_features.parquet", index=False)
    return df


if __name__ == "__main__":
    ejecutar_features()
