import pandas as pd
import numpy as np
from pathlib import Path
from src.validation.control_datos import validar_warehouse

PATH = Path(__file__).resolve().parents[2]
INTERIM = PATH / "data" / "interim"
PROCESSED = PATH / "data" / "processed"
TABLA_MAESTRA_ESTRUCTURAL = INTERIM / "tabla_maestra_estructural.parquet"

def cargar_tabla_maestra_estructural():
    df_maestra_estructural = pd.read_parquet(TABLA_MAESTRA_ESTRUCTURAL)
    return df_maestra_estructural

def limpiar_aemet(df_maestra_estructural):
    # Limpiar columnas de AEMET
    cols_aemet = df_maestra_estructural.select_dtypes(include=['object']).columns

    df_maestra_estructural[cols_aemet] = df_maestra_estructural[cols_aemet].apply(lambda x: x.str.replace(',', '.'))
    df_maestra_estructural[cols_aemet] =  df_maestra_estructural[cols_aemet].replace({'Ip': 0, 'Acum': np.nan})
    df_maestra_estructural[cols_aemet] = df_maestra_estructural[cols_aemet].astype(float)
    return df_maestra_estructural

def ejecutar_etl():
        df_maestra_estructural = cargar_tabla_maestra_estructural()
        df_maestra_estructural = limpiar_aemet(df_maestra_estructural)
        df_maestra_estructural = validar_warehouse(df_maestra_estructural)
        df_maestra_estructural.to_parquet(PROCESSED / "tabla_maestra_procesada.parquet", index=False)
        return df_maestra_estructural

if __name__ == "__main__":
    ejecutar_etl()

