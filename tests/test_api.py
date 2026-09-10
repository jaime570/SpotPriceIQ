import pytest
from fastapi.testclient import TestClient
from src.api.main import app, ml

@pytest.fixture
def client():
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
    assert r.json()["model_name"] == "spotprice-xgboost"

def test_predict_ok(client):
    # construimos un payload válido con las 102 features (a 0.0 nos vale para el test)
    payload = {"features": {f: 0.0 for f in ml["features"]}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "prediccion_eur_mwh" in r.json()

def test_predict_incompleto(client):
    # mandamos solo 1 feature -> debe rechazarlo con 422
    r = client.post("/predict", json={"features": {"precio_lag_24": 45.0}})
    assert r.status_code == 422