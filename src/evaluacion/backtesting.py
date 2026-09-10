"""
Backtesting walk-forward — fuente única de verdad.

Generaliza el bucle de la Fase 4 para que baseline y modelos usen EXACTAMENTE
la misma maquinaria. Lo que cambia entre uno y otro (cómo se predice) se pasa
como argumento, así el bucle y el particionado viven en un solo sitio.
"""

import numpy as np
import pandas as pd
from src.evaluacion.metricas import evaluar, pinball_loss


def walk_forward(df, target, predecir, col_mes="meses", inicio=12, ventana=None):
    """Walk-forward mensual de origen móvil (punto).

    predecir(tr, te) -> predicciones para `te`.
    ventana: None = expanding (todo el pasado); N = rolling de N meses.
    Devuelve DataFrame: una fila por mes con {mes, MAE, RMSE, MAPE, sMAPE}.
    """
    meses = df[col_mes].sort_values().unique()
    filas = []
    for mes in meses[inicio:]:
        if ventana is None:
            tr = df[df[col_mes] < mes]
        else:
            tr = df[(df[col_mes] < mes) & (df[col_mes] >= mes - ventana)]
        te = df[df[col_mes] == mes]
        pred = predecir(tr, te)
        filas.append({"mes": mes, **evaluar(te[target], pred)})
    return pd.DataFrame(filas)


def walk_forward_cuantiles(df, target, cuantiles, predecir_cuantiles,
                           col_mes="meses", inicio=12, ventana=None,
                           bandas=((0.10, 0.90), (0.05, 0.95))):
    """Walk-forward mensual de CALIBRACIÓN para forecast por cuantiles.

    predecir_cuantiles(tr, te) -> DataFrame de predicciones ya MONOTONIZADO,
        con una columna por cuantil (mismos valores que `cuantiles`) e índice
        alineado con `te`.
    bandas: pares (lo, hi) de cuantiles simétricos para medir cobertura.

    Por cada mes calcula:
      - cobertura empírica de cada banda (fracción de reales dentro),
      - pinball loss medio sobre los cuantiles.
    Devuelve DataFrame: una fila por mes con {mes, cob_80, cob_90, pinball}.
    """
    meses = df[col_mes].sort_values().unique()
    filas = []
    for mes in meses[inicio:]:
        if ventana is None:
            tr = df[df[col_mes] < mes]
        else:
            tr = df[(df[col_mes] < mes) & (df[col_mes] >= mes - ventana)]
        te = df[df[col_mes] == mes]
        preds = predecir_cuantiles(tr, te)          # DataFrame monotónico
        y = te[target].to_numpy()
        fila = {"mes": mes}
        for lo, hi in bandas:
            dentro = (y >= preds[lo].to_numpy()) & (y <= preds[hi].to_numpy())
            fila[f"cob_{int(round((hi - lo) * 100))}"] = dentro.mean()
        fila["pinball"] = np.mean([pinball_loss(y, preds[q].to_numpy(), q) for q in cuantiles])
        filas.append(fila)
    return pd.DataFrame(filas)
