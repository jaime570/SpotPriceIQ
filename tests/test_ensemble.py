import mlflow
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.modelos.ensemble import RECETA_XGB, Ensemble, guardar_pyfunc
from src.monitorizacion.prediccion import construir_registro, predecir_componentes
from tests.conftest import datos_sinteticos


def test_prediccion_es_la_media_de_las_componentes(ensemble_pequeno):
    X, _ = datos_sinteticos(n=50, semilla=1)
    c = ensemble_pequeno.predecir_componentes(X)
    assert list(c.columns) == ["componente_a", "componente_b"]
    np.testing.assert_allclose(ensemble_pequeno.predict(X), (c["componente_a"] + c["componente_b"]) / 2)


def test_la_recta_es_la_regresion_lineal_sobre_el_gas():
    X, y = datos_sinteticos()
    modelo = Ensemble(receta={"n_estimators": 5, "n_jobs": 1}).fit(X, y)
    ref = LinearRegression().fit(X[["gas_d1"]], y)
    assert modelo.recta["pendiente"] == pytest.approx(ref.coef_[0])
    assert modelo.recta["ordenada"] == pytest.approx(ref.intercept_)


def test_la_componente_b_extrapola_el_gas_y_la_a_no(ensemble_pequeno):
    """El motivo del híbrido: con el gas fuera del rango de entrenamiento (20–70 €),
    el árbol A se queda plano y B sigue subiendo con la pendiente de la recta."""
    X, _ = datos_sinteticos(n=1, semilla=2)
    alto, muy_alto = X.assign(gas_d1=100.0), X.assign(gas_d1=150.0)
    c_alto = ensemble_pequeno.predecir_componentes(alto)
    c_muy_alto = ensemble_pequeno.predecir_componentes(muy_alto)
    subida_b = c_muy_alto["componente_b"].iloc[0] - c_alto["componente_b"].iloc[0]
    assert subida_b == pytest.approx(50 * ensemble_pequeno.recta["pendiente"])
    assert c_muy_alto["componente_a"].iloc[0] == pytest.approx(c_alto["componente_a"].iloc[0])


def test_ordena_las_columnas_como_en_el_entrenamiento(ensemble_pequeno):
    X, _ = datos_sinteticos(n=20, semilla=3)
    np.testing.assert_allclose(ensemble_pequeno.predict(X[X.columns[::-1]]), ensemble_pequeno.predict(X))


def test_sin_gas_no_entrena():
    X, y = datos_sinteticos()
    with pytest.raises(ValueError, match="gas_d1"):
        Ensemble().fit(X.drop(columns="gas_d1"), y)


def test_receta_de_produccion_por_defecto():
    assert Ensemble().receta == RECETA_XGB


def test_guardar_y_cargar_da_las_mismas_predicciones(ensemble_pequeno, tmp_path):
    X, _ = datos_sinteticos(n=50, semilla=4)
    cargado = Ensemble.cargar(ensemble_pequeno.guardar(tmp_path / "piezas"))
    np.testing.assert_allclose(cargado.predict(X), ensemble_pequeno.predict(X))
    ficheros = sorted(p.name for p in (tmp_path / "piezas").iterdir())
    assert ficheros == ["ensemble.json", "xgb_a.ubj", "xgb_b.ubj"]


def test_pyfunc_predice_la_media_y_da_las_componentes(ensemble_pequeno, tmp_path):
    """El formato de model_champion/: se carga con mlflow.pyfunc como cualquier modelo,
    predict devuelve la media y las componentes salen de unwrap_python_model()."""
    X, _ = datos_sinteticos(n=50, semilla=5)
    modelo = mlflow.pyfunc.load_model(str(guardar_pyfunc(ensemble_pequeno, tmp_path / "pyfunc")))
    np.testing.assert_allclose(modelo.predict(X), ensemble_pequeno.predict(X))
    componentes = predecir_componentes(modelo, X)
    np.testing.assert_allclose(componentes.to_numpy(), ensemble_pequeno.predecir_componentes(X).to_numpy())


def test_registro_con_y_sin_componentes(ensemble_pequeno):
    X, _ = datos_sinteticos(n=24, semilla=6)
    a_predecir = X.assign(datetime_utc=range(24), fecha="2026-11-13", precio_lag_24=50.0)
    meta = {"version": 15, "run_id": "abc"}
    pred = ensemble_pequeno.predict(X)

    con = construir_registro(a_predecir, pred, meta, ensemble_pequeno.predecir_componentes(X))
    np.testing.assert_allclose(con["prediccion"], (con["componente_a"] + con["componente_b"]) / 2)

    # Un modelo que no es el ensemble (p. ej. XGBoost tras volver a @anterior): componentes vacías
    sin = construir_registro(a_predecir, pred, meta, componentes=None)
    assert sin[["componente_a", "componente_b"]].isna().all().all()
    assert predecir_componentes(object(), X) is None
