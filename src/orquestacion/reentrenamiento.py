"""
Reentrenamiento semanal con puerta de promoción (Fase 9.3).

Decisión basada en la simulación de producción (notebook 12): reentrenar cada
semana es la política con menor error (MAE 15,90 vs 16,26 mensual). Cada lunes:

  1. VALIDAR   copia entrenada SIN los últimos 14 días, medida en esos 14 días
  2. ENTRENAR  modelo definitivo CON todos los datos
  3. REGISTRAR nueva versión en MLflow con las métricas de la validación
  4. PUERTA    si la validación gana al naive → @champion + exportar a model_champion/
               si no → queda registrada pero NO se promociona (guardarraíl ante
               reentrenos rotos: datos corruptos, ingesta fallida, bugs)
"""
import json
import shutil
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from prefect import flow, task
from xgboost import XGBRegressor

from src.features.feature_sets import TARGET, get_feature_sets

ROOT = Path(__file__).resolve().parents[2]
DB = (ROOT / "mlflow.db").as_posix()
DATA = ROOT / "data" / "processed" / "tabla_features.parquet"
MODEL_NAME = "spotprice-xgboost"
CARPETA_CHAMPION = ROOT / "model_champion"
DIAS_VALIDACION = 14


@task
def cargar_datos() -> pd.DataFrame:
    return pd.read_parquet(DATA)


@task
def entrenar(df: pd.DataFrame) -> XGBRegressor:
    feats = get_feature_sets(df)["predictivo"]
    df = df[df["entrenable"]]
    modelo = XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42, n_jobs=-1,
    )
    modelo.fit(df[feats], df[TARGET])
    return modelo


@task
def validar_candidato(df: pd.DataFrame, dias: int = DIAS_VALIDACION) -> dict:
    """Entrena una copia SIN los últimos `dias` días y la mide en ellos (holdout
    temporal: datos que la copia no ha visto). Devuelve sus MAE y los del naive."""
    feats = get_feature_sets(df)["predictivo"]
    ultimo = df.loc[df["entrenable"], "fecha"].max()
    corte = ultimo - pd.Timedelta(days=dias)

    train = df[df["fecha"] < corte]
    valid = df[(df["fecha"] >= corte) & df["entrenable"]]

    modelo_val = entrenar.fn(train)
    pred = modelo_val.predict(valid[feats])

    return {
        "mae_val": float((valid[TARGET] - pred).abs().mean()),
        "mae_naive_val": float((valid[TARGET] - valid["precio_lag_24"]).abs().mean()),
        "validacion_desde": str(corte.date()),
        "entrenado_hasta": str(ultimo.date()),
    }


@task
def registrar(modelo: XGBRegressor, validacion: dict) -> str:
    """Registra el modelo definitivo en MLflow con las métricas de la validación."""
    mlflow.set_tracking_uri(f"sqlite:///{DB}")
    mlflow.set_experiment(MODEL_NAME)
    with mlflow.start_run(run_name="reentrenamiento-semanal"):
        mlflow.set_tag("validacion", f"holdout_{DIAS_VALIDACION}d + produccion_full_data")
        mlflow.log_params(modelo.get_params())
        mlflow.log_metrics({
            "mae_val_14d": validacion["mae_val"],
            "mae_naive_val_14d": validacion["mae_naive_val"],
        })
        info = mlflow.xgboost.log_model(
            modelo, name="model",
            pip_requirements=["xgboost", "scikit-learn"],
            registered_model_name=MODEL_NAME,
        )
    return info.registered_model_version


@task
def promover_y_exportar(version: str, feats: list, validacion: dict) -> None:
    """Pone el alias @champion a `version` y la materializa en model_champion/
    (la carpeta que usan la predicción diaria y la API), con model_meta.json
    incluyendo la lista de features. El cambio de carpeta es seguro: se descarga
    a una carpeta nueva y solo al final se intercambia; la anterior queda de copia."""
    mlflow.set_tracking_uri(f"sqlite:///{DB}")
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", version)
    mv = client.get_model_version(MODEL_NAME, version)

    nueva = ROOT / "model_champion_nuevo"
    anterior = ROOT / "model_champion_anterior"
    if nueva.exists():
        shutil.rmtree(nueva)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{MODEL_NAME}/{version}", dst_path=str(nueva)
    )

    meta = {
        "model_name": MODEL_NAME,
        "alias": "champion",
        "version": int(version),
        "run_id": mv.run_id,
        **validacion,
        "features": feats,
    }
    (nueva / "model_meta.json").write_text(json.dumps(meta, indent=2))

    # Intercambio: actual → anterior (copia de seguridad), nueva → actual
    if anterior.exists():
        shutil.rmtree(anterior)
    if CARPETA_CHAMPION.exists():
        CARPETA_CHAMPION.rename(anterior)
    nueva.rename(CARPETA_CHAMPION)


@flow(name="reentrenamiento", log_prints=True)
def reentrenamiento():
    df = cargar_datos()
    feats = get_feature_sets(df)["predictivo"]

    validacion = validar_candidato(df)
    print(f"Validación ({validacion['validacion_desde']} → {validacion['entrenado_hasta']}): "
          f"MAE modelo {validacion['mae_val']:.2f} | MAE naive {validacion['mae_naive_val']:.2f}")

    modelo = entrenar(df)
    version = registrar(modelo, validacion)

    if validacion["mae_val"] < validacion["mae_naive_val"]:
        promover_y_exportar(version, feats, validacion)
        print(f"OK: v{version} promocionada a @champion y exportada a model_champion/")
    else:
        print(f"AVISO: v{version} registrada pero NO promocionada "
              f"(no gana al naive en la validación). Sigue el champion anterior.")


if __name__ == "__main__":
    reentrenamiento()