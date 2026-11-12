import numpy as np
import pandas as pd
import pytest

from src.modelos.ensemble import Ensemble

FEATURES_SINTETICAS = ["x0", "x1", "x2", "gas_d1"]


def datos_sinteticos(n=300, semilla=0):
    """Precio sintético que depende linealmente del gas y de x0 (sin datos reales ni MLflow)."""
    rng = np.random.default_rng(semilla)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=FEATURES_SINTETICAS[:3])
    X["gas_d1"] = rng.uniform(20, 70, size=n)
    y = pd.Series(1.8 * X["gas_d1"] + 10 * X["x0"] + rng.normal(scale=2, size=n), name="precio_espana")
    return X, y


@pytest.fixture(scope="session")
def ensemble_pequeno():
    """Ensemble con pocos árboles: rápido y suficiente para probar la mecánica."""
    X, y = datos_sinteticos()
    return Ensemble(receta={"n_estimators": 20, "n_jobs": 1}).fit(X, y)
