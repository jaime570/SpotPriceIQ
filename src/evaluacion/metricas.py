"""
Métricas de evaluación — fuente única de verdad.

Extraído del notebook 05_baseline para reutilizarlo idéntico en modelado
(XGBoost, LSTM, ensemble) y evitar que la definición diverja entre notebooks.
Convención fija del proyecto: el orden de argumentos es SIEMPRE (y_real, y_pred).
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)


def evaluar(y_real, y_pred):
    """Devuelve un dict con MAE, RMSE, MAPE y sMAPE.

    Nota de dominio: con ~18% de horas a precio <= 0 (renovables), MAPE es
    inservible; se reportan MAE y RMSE. sMAPE se calcula a mano porque np.abs
    sirve tanto para Series (baselines) como para arrays numpy (modelos).
    """
    smape = (np.abs(y_pred - y_real) / ((np.abs(y_real) + np.abs(y_pred)) / 2)).mean()

    diccionario = {
        "MAE": mean_absolute_error(y_real, y_pred),
        "RMSE": root_mean_squared_error(y_real, y_pred),
        "MAPE": mean_absolute_percentage_error(y_real, y_pred),
        "sMAPE": smape,
    }
    return diccionario


def pinball_loss(y_real, y_pred, q):
    """Pinball (quantile) loss para el cuantil q en [0, 1].

    Castiga asimétricamente: quedarse corto pesa q, pasarse pesa (1-q). Al
    minimizarlo, la predicción aterriza en el cuantil q. Menor = mejor.
    Acepta Series o arrays numpy. Orden (y_real, y_pred).
    """
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    e = y_real - y_pred
    return np.mean(np.maximum(q * e, (q - 1) * e))
