import pandas as pd
from pathlib import Path
from src.features.feature_sets  import get_feature_sets
import json
import mlflow




ROOT = Path(__file__).resolve().parents[2]
TABLA_FEATURES = ROOT / "data" / "processed" / "tabla_features.parquet"
RUTA_REGISTRO = ROOT / "data" / "monitoring" / "predicciones_prueba.parquet"

def seleccionar_filas_d1(df):
    """Devuelve las filas de D+1 listas para predecir (completas y sin precio). Lanza error si hay filas futuras incompletas o de más de un día."""
    
    a_predecir = df[(df["completa_features"]) & (df["precio_espana"].isna())]
    futuras_incompletas = df[(df["precio_espana"].isna()) & (~df["completa_features"])]
    
    if not futuras_incompletas.empty:
        feats = get_feature_sets(df)["predictivo"]
        nulos = futuras_incompletas[feats].isna().sum()
        faltan = nulos[nulos > 0].index.tolist()
        raise ValueError(f"{len(futuras_incompletas)} filas futuras incompletas. Columnas vacías: {faltan[:10]}")
    if a_predecir.empty:
        print("0 filas a predecir: ¿OMIE ya ha publicado el precio de mañana?")
        return a_predecir
    if a_predecir["fecha"].nunique() > 1:
        raise ValueError(f"Filas a predecir de varios días: {a_predecir['fecha'].unique()}")
    
    return a_predecir

def predecir(a_predecir):
    """Predice las filas de D+1 con el modelo de model_champion/ (el mismo que sirve la API). Devuelve (predicciones, meta)."""
    modelo = mlflow.pyfunc.load_model(str(ROOT / "model_champion"))
    meta = json.loads((ROOT / "model_champion" / "model_meta.json").read_text())

    assert meta["features"] == get_feature_sets(a_predecir)["predictivo"], \
        "Las features del modelo no coinciden con las del código: ¿falta reentrenar o exportar?"

    X = a_predecir[meta["features"]]
    predicciones = modelo.predict(X)
    return predicciones, meta

def construir_registro(a_predecir, predicciones, meta):
    """Una fila por hora predicha: predicción, naive del mismo momento y trazabilidad del modelo."""
    registro = (a_predecir[["datetime_utc", "fecha", "precio_lag_24"]]
                .rename(columns={"fecha": "fecha_objetivo", "precio_lag_24": "naive_d1"})
                .reset_index(drop=True))
    registro["prediccion"] = predicciones
    registro["modelo_version"] = meta["version"]
    registro["run_id"] = meta["run_id"]
    registro["momento_prediccion"] = pd.Timestamp.now(tz="UTC")
    return registro
                                
def guardar_registro(registro: pd.DataFrame) -> Path:
    """Añade las predicciones al registro (data/monitoring/predicciones.parquet).
    Si una hora ya estaba predicha, se conserva la PRIMERA predicción (la hecha
    antes de la subasta): el registro nunca se sobrescribe con datos posteriores."""
    RUTA_REGISTRO.parent.mkdir(parents=True, exist_ok=True)

    if RUTA_REGISTRO.exists():
        previa = pd.read_parquet(RUTA_REGISTRO)
        tabla = pd.concat([previa, registro], ignore_index=True)
        tabla = tabla.drop_duplicates(subset=["datetime_utc"], keep="first")
    else:
        tabla = registro

    tabla = tabla.sort_values("datetime_utc").reset_index(drop=True)
    tabla.to_parquet(RUTA_REGISTRO, index=False)
    return RUTA_REGISTRO

def predecir_d1():
    """Pipeline de predicción diaria (Fase 9.2, piezas B y C):
    selecciona las horas de D+1, las predice con el @champion y las guarda
    en el registro. Debe ejecutarse ANTES de la subasta (~11:15)."""
    df = pd.read_parquet(TABLA_FEATURES)

    a_predecir = seleccionar_filas_d1(df)
    if a_predecir.empty:
        return None

    predicciones, meta = predecir(a_predecir)
    registro = construir_registro(a_predecir, predicciones, meta)
    ruta = guardar_registro(registro)

    perfil = registro.assign(
        hora=registro["datetime_utc"].dt.tz_convert("Europe/Madrid").dt.hour
    )[["hora", "prediccion"]]

    baratas = perfil.nsmallest(4, "prediccion").sort_values("hora")
    caras = perfil.nlargest(4, "prediccion").sort_values("hora")

    dia = a_predecir["fecha"].iloc[0].date()
    print(f"Predicción D+1 guardada: {dia} | {len(registro)} horas | modelo v{meta['version']} | {ruta}")
    print("  4 horas más BARATAS (comprar/cargar): "
          + ", ".join(f"{h:02d}h ({p:.1f} €)" for h, p in zip(baratas["hora"], baratas["prediccion"])))
    print("  4 horas más CARAS (vender/descargar): "
          + ", ".join(f"{h:02d}h ({p:.1f} €)" for h, p in zip(caras["hora"], caras["prediccion"])))
    return registro


if __name__ == "__main__":
    predecir_d1()
