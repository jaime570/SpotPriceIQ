import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata

PATH = Path(__file__).resolve().parents[2]
INTERIM = PATH / "data" / "interim"
AEMET_TABLA = INTERIM / "tabla_AEMET"
OMIE_TABLA = INTERIM / "tabla_OMIE"
ESIOS_TABLA = INTERIM / "tabla_ESIOS"

def cargar_tablas():
    df_aemet = pd.read_csv(AEMET_TABLA, encoding="utf-8")
    df_omie = pd.read_csv(OMIE_TABLA, encoding="utf-8")
    df_esios= pd.read_csv(ESIOS_TABLA, encoding="utf-8")
    return (df_aemet, df_omie, df_esios)

def limpia(txt):
    txt = txt.lower().replace(',', '').replace(' ', '_')
    txt = ''.join(c for c in unicodedata.normalize('NFKD', txt)
                  if not unicodedata.combining(c))
    return txt

def pivotar(TABLA_AEMET):
    
    TABLA_AEMET['nombre'] = TABLA_AEMET['nombre'].apply(limpia)
    cols_aemet = ['tmed', 'tmax', 'velmedia', 'racha', 'sol', 'prec']
    
    # Pivotar tabla AEMET
    TABLA_AEMET['fecha'] = pd.to_datetime(TABLA_AEMET['fecha'])
    df_aemet_pivot = TABLA_AEMET.pivot(index='fecha', columns='nombre', values= cols_aemet)
    
    #aplanar nombre columnas aemet
    df_aemet_pivot.columns = [col[0] + '_' + col[1] if isinstance(col, tuple) else col for col in df_aemet_pivot.columns]   

    #borrar columnas vacias
    df_aemet_pivot = df_aemet_pivot.dropna(axis=1, how='all')
    df_aemet_pivot.reset_index(inplace=True)
    return (df_aemet_pivot)

def construir_datetime_omie(TABLA_OMIE):
    df = TABLA_OMIE.copy()
    dia = pd.to_datetime(df[['ano','mes','dia']].rename(
        columns={'ano':'year','mes':'month','dia':'day'}))

# 2. ancla UTC (medianoche local -> UTC)
    ancla = dia.dt.tz_localize('Europe/Madrid').dt.tz_convert('UTC')

# 3. paso por día: 60 min si el día tiene <=25 periodos, si no 15 min
    n = df.groupby(['ano','mes','dia'])['hora'].transform('size')
    paso = np.where(n <= 25, 60, 15)

# 4. timestamp real = ancla + (hora-1)*paso, como tiempo transcurrido
    df['datetime_utc'] = ancla + pd.to_timedelta((df['hora']-1)*paso, unit='m')
    df['datetime_utc'] = df['datetime_utc'].dt.floor('h')

    df_omie = (df.groupby('datetime_utc')[['precio_espana','precio_portugal']]
                 .mean()
                 .reset_index())
    return df_omie

def construir_datetime_esios(TABLA_ESIOS):

    df = TABLA_ESIOS.copy()
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
    return df

def concatenar(df_omie, df_esios, df_aemet):
    tabla_final = df_omie.merge(df_esios, on='datetime_utc', how='left')

# Convertir 'fecha' a medianoche en zona horaria Europe/Madrid y hacer naive
    tabla_final['fecha'] = (tabla_final['datetime_utc']
                        .dt.tz_convert('Europe/Madrid')   
                        .dt.normalize()                  
                        .dt.tz_localize(None))    

    tabla_final = tabla_final.merge(df_aemet, on='fecha', how='left')
    tabla_final.to_parquet(INTERIM / "tabla_maestra_estructural.parquet", index=False)
    return tabla_final  

def construir_tabla_maestra():
    """Construye la tabla maestra estructural a partir de las tablas crudas de interim."""

    TABLA_AEMET, TABLA_OMIE, TABLA_ESIOS = cargar_tablas()
    df_aemet_pivot = pivotar(TABLA_AEMET)
    df_omie = construir_datetime_omie(TABLA_OMIE)
    df_esios = construir_datetime_esios(TABLA_ESIOS)
    tabla_final = concatenar(df_omie, df_esios, df_aemet_pivot)
    return tabla_final

if __name__ == "__main__":
    construir_tabla_maestra()
