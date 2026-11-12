import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, ml
from src.modelos.ensemble import RECETA_XGB, TIPO, guardar_pyfunc
from tests.conftest import FEATURES_SINTETICAS


@pytest.fixture(scope="module")
def carpeta_modelo(tmp_path_factory, ensemble_pequeno):
    """Modelo con el mismo formato que model_champion/ (pyfunc + model_meta.json),
    hecho con un ensemble pequeño y datos sintéticos: la CI no tiene MLflow ni datos,
    y así el test no depende de qué modelo haya en model_champion/."""
    carpeta = guardar_pyfunc(ensemble_pequeno, tmp_path_factory.mktemp("modelo") / "model_champion")
    meta = {"model_name": "spotprice-xgboost", "alias": "champion", "version": 0, "run_id": "test",
            "tipo": TIPO, "receta_xgb": RECETA_XGB, "recta": ensemble_pequeno.recta,
            "features": FEATURES_SINTETICAS}
    (carpeta / "model_meta.json").write_text(json.dumps(meta))
    return carpeta


@pytest.fixture
def client(carpeta_modelo, monkeypatch):
    # La API lee MODEL_DIR al arrancar (lifespan)
    monkeypatch.setenv("MODEL_DIR", str(carpeta_modelo))
    # OJO: el 'with' es lo que dispara el lifespan (carga el modelo).
    # Sin 'with', el modelo no se cargaría y /predict fallaría.
    with TestClient(app) as c:
        yield c

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    info = r.json()
    assert info["model_name"] == "spotprice-xgboost"
    assert info["tipo"] == TIPO
    assert info["receta_xgb"] == RECETA_XGB
    assert info["n_features"] == len(FEATURES_SINTETICAS)
    assert "features" not in info

def test_predict_ok(client):
    # payload válido con todas las features que pide el modelo (a 0.0 nos vale para el test)
    payload = {"features": {f: 0.0 for f in ml["features"]}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "prediccion_eur_mwh" in r.json()

def test_predict_devuelve_la_media_del_ensemble(client, ensemble_pequeno):
    fila = {"x0": 1.0, "x1": 0.0, "x2": 0.0, "gas_d1": 50.0}
    r = client.post("/predict", json={"features": fila})
    esperado = ensemble_pequeno.predict(pd.DataFrame([fila]))[0]
    assert r.json()["prediccion_eur_mwh"] == pytest.approx(round(float(esperado), 2))

def test_predict_incompleto(client):
    # mandamos solo 1 feature -> debe rechazarlo con 422
    r = client.post("/predict", json={"features": {"precio_lag_24": 45.0}})
    assert r.status_code == 422
