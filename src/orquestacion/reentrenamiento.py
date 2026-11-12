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

Desde 2026-11 el modelo es el ensemble 50/50 con MIBGAS (src/modelos/ensemble.py),
registrado como un único modelo pyfunc. El nombre del modelo registrado sigue
siendo `spotprice-xgboost` para que todas las versiones (y los alias @champion,
@anterior, @candidato) vivan en el mismo sitio; el tipo va en tags y en model_meta.json.

Para probar un reentreno sin tocar producción: `reentrenamiento(alias="candidato")`
registra la versión con @candidato (si gana al naive) y NO exporta a model_champion/
ni avisa del resultado por Telegram (si el flujo falla, sí: aviso_fallo).
"""
import json
import shutil
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from prefect import flow, task

from src.features.feature_sets import TARGET, get_feature_sets
from src.modelos.ensemble import RECETA_XGB, TIPO, Ensemble, registrar_en_mlflow
from src.notificaciones.telegram import avisar, aviso_fallo

ROOT = Path(__file__).resolve().parents[2]
DB = (ROOT / "mlflow.db").as_posix()
DATA = ROOT / "data" / "processed" / "tabla_features.parquet"
MODEL_NAME = "spotprice-xgboost"
CARPETA_CHAMPION = ROOT / "model_champion"
DIAS_VALIDACION = 14
ALIAS_PERMITIDOS = ("champion", "candidato")


@task
def cargar_datos() -> pd.DataFrame:
    return pd.read_parquet(DATA)


@task
def entrenar(df: pd.DataFrame) -> Ensemble:
    feats = get_feature_sets(df)["predictivo"]
    df = df[df["entrenable"]]
    return Ensemble().fit(df[feats], df[TARGET])


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
def registrar(modelo: Ensemble, validacion: dict) -> str:
    """Registra el modelo definitivo en MLflow con las métricas de la validación."""
    mlflow.set_tracking_uri(f"sqlite:///{DB}")
    mlflow.set_experiment(MODEL_NAME)
    with mlflow.start_run(run_name="reentrenamiento-semanal"):
        mlflow.set_tag("validacion", f"holdout_{DIAS_VALIDACION}d + produccion_full_data")
        mlflow.set_tag("tipo", TIPO)
        mlflow.log_params({**{f"xgb_{k}": v for k, v in RECETA_XGB.items()},
                           "recta_ordenada": modelo.recta["ordenada"],
                           "recta_pendiente": modelo.recta["pendiente"],
                           "n_features": len(modelo.features)})
        mlflow.log_metrics({
            "mae_val_14d": validacion["mae_val"],
            "mae_naive_val_14d": validacion["mae_naive_val"],
        })
        info = registrar_en_mlflow(modelo, registered_model_name=MODEL_NAME)
    return info.registered_model_version


@task
def poner_alias(version: str, alias: str) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{DB}")
    MlflowClient().set_registered_model_alias(MODEL_NAME, alias, version)


@task
def promover_y_exportar(version: str, feats: list, validacion: dict, recta: dict) -> None:
    """Pone el alias @champion a `version` y la materializa en model_champion/
    (la carpeta que usan la predicción diaria y la API), con model_meta.json
    incluyendo la lista de features. El cambio de carpeta es seguro: se descarga
    a una carpeta nueva y solo al final se intercambia; la anterior queda de copia."""
    poner_alias.fn(version, "champion")
    mlflow.set_tracking_uri(f"sqlite:///{DB}")
    mv = MlflowClient().get_model_version(MODEL_NAME, version)

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
        "tipo": TIPO,
        "receta_xgb": RECETA_XGB,
        "recta": recta,
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


@flow(name="reentrenamiento", log_prints=True, on_failure=[aviso_fallo], on_crashed=[aviso_fallo])
def reentrenamiento(alias: str = "champion"):
    if alias not in ALIAS_PERMITIDOS:
        raise ValueError(f"alias debe ser uno de {ALIAS_PERMITIDOS}, no {alias!r}")
    produccion = alias == "champion"

    df = cargar_datos()
    feats = get_feature_sets(df)["predictivo"]

    validacion = validar_candidato(df)
    print(f"Validación ({validacion['validacion_desde']} → {validacion['entrenado_hasta']}): "
          f"MAE modelo {validacion['mae_val']:.2f} | MAE naive {validacion['mae_naive_val']:.2f}")

    modelo = entrenar(df)
    version = registrar(modelo, validacion)
    gana = validacion["mae_val"] < validacion["mae_naive_val"]

    if not produccion:
        # Prueba: ni model_champion/ ni Telegram
        if gana:
            poner_alias(version, alias)
            print(f"PRUEBA: v{version} registrada con @{alias} (no se exporta a model_champion/)")
        else:
            print(f"PRUEBA: v{version} registrada sin alias (no gana al naive en la validación)")
        return version

    if gana:
        promover_y_exportar(version, feats, validacion, modelo.recta)
        print(f"OK: v{version} promocionada a @champion y exportada a model_champion/")
        avisar(f"ℹ️ Reentreno: v{version} promocionada a champion "
               f"(MAE validación {validacion['mae_val']:.1f} vs naive {validacion['mae_naive_val']:.1f})")
    else:
        print(f"AVISO: v{version} registrada pero NO promocionada "
              f"(no gana al naive en la validación). Sigue el champion anterior.")
        avisar(f"ℹ️ Reentreno: v{version} NO promocionada (no gana al naive en la validación: "
               f"{validacion['mae_val']:.1f} vs {validacion['mae_naive_val']:.1f}). Sigue el champion anterior.")
    return version


if __name__ == "__main__":
    reentrenamiento()
