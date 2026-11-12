"""Modelo de producción: ensemble 50/50 de dos modelos con gas (decisión 2026-11-10).

  A = XGBoost sobre las features oficiales (97 + gas_d1).
  B = híbrido: recta precio ~ gas_d1 + XGBoost (mismas features) sobre el residuo.
  Predicción = (A + B) / 2.

Por qué B: un árbol no extrapola (regla 5) y con el gas fuera del rango de
entrenamiento A se queda corto; la recta de B sí extrapola el nivel. En otoño de
2026 A tiene sesgo negativo y B positivo, y la media compensa.

Piezas:
  - `Ensemble`: entrena, predice y guarda/carga sus piezas como ficheros
    (dos .ubj de XGBoost + un JSON con la recta, las features y la receta).
    No se serializa con pickle: los .ubj son el formato nativo de XGBoost y no
    dependen de la versión de cloudpickle.
  - `EnsemblePyfunc`: envoltorio `mlflow.pyfunc.PythonModel` para registrarlo en
    MLflow como un único modelo. `predict` devuelve la media; las componentes se
    piden con `modelo.unwrap_python_model().predecir_componentes(X)`.
    La clase se guarda por referencia (`src.modelos.ensemble`), así que para
    cargar el modelo `src/` tiene que ser importable (lo es en el repo y en Docker).
"""

import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

TIPO = "ensemble_50_50_xgb_hibrido_gas"
COLUMNA_GAS = "gas_d1"

# Receta de producción (regla 3), la misma para A y para B
RECETA_XGB = {
    "n_estimators": 400, "learning_rate": 0.05, "max_depth": 6,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "tree_method": "hist", "random_state": 42, "n_jobs": -1,
}


class Ensemble:
    """Ensemble 50/50 de A (XGBoost) y B (recta sobre gas_d1 + XGBoost del residuo).

    `receta` permite cambiar parámetros de XGBoost (p. ej. pocos árboles en los
    tests); en producción se usa RECETA_XGB tal cual.
    """

    def __init__(self, receta: dict | None = None):
        self.receta = {**RECETA_XGB, **(receta or {})}
        self.features: list[str] | None = None
        self.xgb_a: XGBRegressor | None = None
        self.xgb_b: XGBRegressor | None = None
        self.recta: dict | None = None      # {"ordenada": ..., "pendiente": ...}

    # --- Entrenamiento y predicción ---

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Ensemble":
        if COLUMNA_GAS not in X.columns:
            raise ValueError(f"El ensemble necesita la columna {COLUMNA_GAS}")
        self.features = list(X.columns)

        self.xgb_a = XGBRegressor(**self.receta).fit(X, y)

        recta = LinearRegression().fit(X[[COLUMNA_GAS]], y)
        self.recta = {"ordenada": float(recta.intercept_), "pendiente": float(recta.coef_[0])}
        residuo = np.asarray(y, dtype=float) - self._nivel_gas(X)
        self.xgb_b = XGBRegressor(**self.receta).fit(X, residuo)
        return self

    def _nivel_gas(self, X: pd.DataFrame) -> np.ndarray:
        """Parte lineal de B: ordenada + pendiente · gas_d1 (extrapola fuera de rango)."""
        gas = X[COLUMNA_GAS].to_numpy(dtype=float)
        return self.recta["ordenada"] + self.recta["pendiente"] * gas

    def predecir_componentes(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predicción de cada componente: columnas componente_a y componente_b."""
        X = X[self.features]                 # mismo orden de columnas que en el entrenamiento
        a = self.xgb_a.predict(X)
        b = self._nivel_gas(X) + self.xgb_b.predict(X)
        return pd.DataFrame({"componente_a": a, "componente_b": b}, index=X.index)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        c = self.predecir_componentes(X)
        return ((c["componente_a"] + c["componente_b"]) / 2).to_numpy()

    # --- Guardar / cargar (ficheros, sin pickle) ---

    def guardar(self, carpeta) -> Path:
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        self.xgb_a.save_model(carpeta / "xgb_a.ubj")
        self.xgb_b.save_model(carpeta / "xgb_b.ubj")
        config = {"tipo": TIPO, "features": self.features, "recta": self.recta, "receta": self.receta}
        (carpeta / "ensemble.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        return carpeta

    @classmethod
    def cargar(cls, carpeta) -> "Ensemble":
        carpeta = Path(carpeta)
        config = json.loads((carpeta / "ensemble.json").read_text(encoding="utf-8"))
        modelo = cls(receta=config["receta"])
        modelo.features = config["features"]
        modelo.recta = config["recta"]
        modelo.xgb_a = XGBRegressor()
        modelo.xgb_a.load_model(carpeta / "xgb_a.ubj")
        modelo.xgb_b = XGBRegressor()
        modelo.xgb_b.load_model(carpeta / "xgb_b.ubj")
        return modelo


class EnsemblePyfunc(mlflow.pyfunc.PythonModel):
    """Envoltorio MLflow: carga las piezas desde el artefacto `ensemble`."""

    def load_context(self, context):
        self.ensemble = Ensemble.cargar(context.artifacts["ensemble"])

    def predict(self, context, model_input, params=None):
        return self.ensemble.predict(model_input)

    def predecir_componentes(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.ensemble.predecir_componentes(X)


def _requisitos() -> list[str]:
    return [f"xgboost=={xgboost.__version__}", f"scikit-learn=={sklearn.__version__}",
            f"pandas=={pd.__version__}"]


def registrar_en_mlflow(modelo: Ensemble, registered_model_name: str):
    """Registra el ensemble como un único modelo pyfunc dentro del run activo."""
    with tempfile.TemporaryDirectory() as tmp:
        piezas = modelo.guardar(Path(tmp) / "ensemble")
        # log_model copia el artefacto, así que la carpeta temporal se puede borrar después
        return mlflow.pyfunc.log_model(
            name="model",
            python_model=EnsemblePyfunc(),
            artifacts={"ensemble": str(piezas)},
            pip_requirements=_requisitos(),
            registered_model_name=registered_model_name,
        )


def guardar_pyfunc(modelo: Ensemble, ruta) -> Path:
    """Guarda el ensemble como modelo pyfunc en una carpeta, sin MLflow tracking
    (lo usan los tests de la API: mismo formato que model_champion/)."""
    with tempfile.TemporaryDirectory() as tmp:
        piezas = modelo.guardar(Path(tmp) / "ensemble")
        mlflow.pyfunc.save_model(
            path=str(ruta),
            python_model=EnsemblePyfunc(),
            artifacts={"ensemble": str(piezas)},
            pip_requirements=_requisitos(),
        )
    return Path(ruta)
