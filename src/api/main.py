import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import PredictRequest, PredictResponse

ROOT = Path(__file__).resolve().parents[2]
ml = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carga desde la carpeta materializada (overridable por env para el contenedor).
    model_dir = os.getenv("MODEL_DIR", str(ROOT / "model_champion"))
    ml["model"] = mlflow.pyfunc.load_model(model_dir)
    ml["meta"] = json.loads((Path(model_dir) / "model_meta.json").read_text())
    ml["features"] = ml["meta"]["features"]
    yield
    ml.clear()


app = FastAPI(title="SpotPriceIQ API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    m = ml["meta"]
    return {k: v for k, v in m.items() if k != "features"} | {"n_features": len(ml["features"])}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    esperadas = ml["features"]
    faltan = [c for c in esperadas if c not in req.features]
    if faltan:
        raise HTTPException(status_code=422, detail=f"Faltan {len(faltan)} features, p.ej.: {faltan[:5]}")
    X = pd.DataFrame([[req.features[c] for c in esperadas]], columns=esperadas)
    pred = float(ml["model"].predict(X)[0])
    return PredictResponse(prediccion_eur_mwh=round(pred, 2),
                           modelo=ml["meta"]["model_name"], version=str(ml["meta"]["version"]))