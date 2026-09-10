import json
from pathlib import Path
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")   # se ejecuta desde la raíz
client = MlflowClient()
mv = client.get_model_version_by_alias("spotprice-xgboost", "champion")
run = client.get_run(mv.run_id)

# descarga (materializa) el modelo campeón a una carpeta local portable
dst = mlflow.artifacts.download_artifacts(
    artifact_uri="models:/spotprice-xgboost@champion",
    dst_path="model_champion",
)
print("Modelo materializado en:", dst)

# metadatos portables (para que /model-info no dependa del registry en el contenedor)
meta = {
    "model_name": "spotprice-xgboost",
    "alias": "champion",
    "version": mv.version,
    "run_id": mv.run_id,
    "mae_wf_ref": run.data.metrics.get("MAE_wf_ref"),
}
Path(dst, "model_meta.json").write_text(json.dumps(meta, indent=2))
print("Meta escrita:", meta)