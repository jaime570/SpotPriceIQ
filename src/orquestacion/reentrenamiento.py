from pathlib import Path

import mlflow
import pandas as pd
from prefect import flow, task
from xgboost import XGBRegressor

from src.features.feature_sets import get_feature_sets, TARGET

ROOT = Path(__file__).resolve().parents[2]
DB = (ROOT / "mlflow.db").as_posix()
DATA = ROOT / "data" / "processed" / "tabla_features.parquet"
MODEL_NAME = "spotprice-xgboost"


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
def registrar(modelo: XGBRegressor) -> str:
    mlflow.set_tracking_uri(f"sqlite:///{DB}")
    mlflow.set_experiment("spotprice-xgboost")
    with mlflow.start_run(run_name="reentrenamiento"):
        mlflow.set_tag("validacion", "produccion_full_data")
        mlflow.log_params(modelo.get_params())
        info = mlflow.xgboost.log_model(
            modelo, name="model",
            pip_requirements=["xgboost", "scikit-learn"],
            registered_model_name=MODEL_NAME,
        )
    return info.registered_model_version


@flow(name="reentrenamiento", log_prints=True)
def reentrenamiento():
    df = cargar_datos()
    modelo = entrenar(df)
    version = registrar(modelo)
    print(f"Registrada nueva versión del modelo: {version}")


if __name__ == "__main__":
    reentrenamiento()