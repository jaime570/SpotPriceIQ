import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLA_FEATURES = ROOT / "data" / "processed" / "tabla_features.parquet"
RUTA_REGISTRO = ROOT / "data" / "monitoring" / "predicciones.parquet"
RUTA_EVALUACION = ROOT / "data" / "monitoring" / "evaluacion_diaria.parquet"

# Umbrales de la alarma de degradación (calibrados en la simulación de la 9.3).
# Únicos para todo el proyecto: los usan calcular_metricas, detectar_degradacion y simulacion.py.
UMBRAL_RATIO = 1.3     # MAE 7d modelo / MAE 7d naive
UMBRAL_SESGO = 14      # |sesgo 7d| en €/MWh
DIAS_SEGUIDOS = 3


def cruzar_con_real(registro, tabla):
    """Une las predicciones con el precio real y deja solo los días COMPLETOS
    (todas sus horas predichas tienen precio real). Funciona con días de
    23/24/25 horas porque compara contra las predichas, no contra 24."""

    real = tabla[["datetime_utc", "precio_espana"]].rename(
        columns={"precio_espana": "precio_real"}
    )
    cruzado = registro.merge(real, on="datetime_utc", how="left")

    por_dia = cruzado.groupby("fecha_objetivo")["precio_real"]
    predichas = por_dia.transform("size")   # todas las filas del día
    con_real = por_dia.transform("count")   # solo las que tienen precio real
    cruzado = cruzado[predichas == con_real].copy()

    return cruzado.reset_index(drop=True)


def metricas_dia(dia):
    """Métricas de error de un día completo (filas ya cruzadas con el real)."""

    mae_modelo = (dia["precio_real"] - dia["prediccion"]).abs().mean()
    mae_naive = (dia["precio_real"] - dia["naive_d1"]).abs().mean()
    # Sesgo: error CON signo (sin abs). Positivo = el modelo predice por encima del real.
    # Detecta la degradación de nivel (hallazgo 9.3) que el MAE de forma y Spearman no ven.
    sesgo = (dia["prediccion"] - dia["precio_real"]).mean()
    pred_centrada = dia["prediccion"] - dia["prediccion"].mean()
    real_centrada = dia["precio_real"] - dia["precio_real"].mean()
    mae_forma = (pred_centrada - real_centrada).abs().mean()
    spearman = dia["prediccion"].corr(dia["precio_real"], method="spearman")

    return pd.Series({"mae_modelo": mae_modelo, "mae_naive": mae_naive, "sesgo": sesgo,
                      "mae_forma": mae_forma, "spearman": spearman})


def regret_dia(dia, k):
    """Regret de compra y venta de un día completo para las K horas elegidas."""
    horas_baratas_modelo = dia.nsmallest(k, "prediccion")
    pagado_modelo = horas_baratas_modelo["precio_real"].mean()
    horas_baratas_reales = dia.nsmallest(k, "precio_real")
    pagado_optimo = horas_baratas_reales["precio_real"].mean()
    regret_compra = pagado_modelo - pagado_optimo

    horas_caras_modelo = dia.nlargest(k, "prediccion")
    venta_modelo = horas_caras_modelo["precio_real"].mean()
    horas_caras_reales = dia.nlargest(k, "precio_real")
    venta_optimo = horas_caras_reales["precio_real"].mean()
    regret_venta = venta_optimo - venta_modelo

    return pd.Series({"regret_compra": regret_compra, "regret_venta": regret_venta})


def _alarma_por_dia(ratio_7d, sesgo_7d, umbral=UMBRAL_RATIO, umbral_sesgo=UMBRAL_SESGO,
                    dias_seguidos=DIAS_SEGUIDOS):
    """Regla de la alarma, en UN solo sitio.
    1) Condición del día: ratio 7d > umbral, o |sesgo 7d| > umbral_sesgo.
       (NaN -> False: sin días suficientes no hay alarma.)
    2) Alarma: la condición se cumple `dias_seguidos` días evaluados seguidos
       (ventana de N filas cuya suma de True vale N)."""
    condicion = (ratio_7d > umbral) | (sesgo_7d.abs() > umbral_sesgo)
    seguidos = condicion.astype(int).rolling(dias_seguidos, min_periods=dias_seguidos).sum()
    return condicion, seguidos == dias_seguidos


def calcular_metricas(cruzado):
    """Métricas por día (error, sesgo, regret K=1/2/4), medias móviles de 7 y 30 días
    y estado de la alarma de degradación por día.
    Solo calcula: no lee ni guarda ficheros (sirve para producción y para simulación)."""
    # Solo las columnas que usan las métricas: evita el aviso de pandas
    # por pasar la columna de agrupación dentro del apply.
    columnas = ["prediccion", "naive_d1", "precio_real"]
    por_dia = cruzado.groupby("fecha_objetivo")[columnas]

    metricas = por_dia.apply(metricas_dia)

    for k in (1, 2, 4):
        regret = por_dia.apply(regret_dia, k=k).add_suffix(f"_k{k}")
        metricas = metricas.join(regret)

    # Medias móviles por días de CALENDARIO ("7D"), no por filas: si falta un día,
    # la ventana sigue siendo de 7 días reales. min_periods deja NaN hasta que haya
    # días suficientes (una "media de 7 días" con 1 día engaña).
    for col in ["mae_modelo", "mae_naive"]:
        metricas[f"{col}_7d"] = metricas[col].rolling("7D", min_periods=5).mean()
        metricas[f"{col}_30d"] = metricas[col].rolling("30D", min_periods=20).mean()

    # Salud del modelo, guardada por día (antes solo se imprimía en el log).
    metricas["sesgo_7d"] = metricas["sesgo"].rolling("7D", min_periods=5).mean()
    metricas["ratio_7d"] = metricas["mae_modelo_7d"] / metricas["mae_naive_7d"]
    metricas["condicion_alarma"], metricas["alarma"] = _alarma_por_dia(
        metricas["ratio_7d"], metricas["sesgo_7d"]
    )

    return metricas


def detectar_degradacion(cruzado, umbral=UMBRAL_RATIO, umbral_sesgo=UMBRAL_SESGO,
                         dias_seguidos=DIAS_SEGUIDOS):
    """Estado de la alarma en el ÚLTIMO día (lo usa simulacion.py).
    Misma regla que calcular_metricas, vía _alarma_por_dia."""
    por_dia = cruzado["fecha_objetivo"]
    error = cruzado["prediccion"] - cruzado["precio_real"]

    mae_dia = error.abs().groupby(por_dia).mean()
    naive_dia = (cruzado["naive_d1"] - cruzado["precio_real"]).abs().groupby(por_dia).mean()
    sesgo_dia = error.groupby(por_dia).mean()

    mae_7d = mae_dia.rolling("7D", min_periods=5).mean()
    naive_7d = naive_dia.rolling("7D", min_periods=5).mean()
    sesgo_7d = sesgo_dia.rolling("7D", min_periods=5).mean()
    ratio_7d = mae_7d / naive_7d

    _, alarma = _alarma_por_dia(ratio_7d, sesgo_7d, umbral, umbral_sesgo, dias_seguidos)

    return {
        "alarma": bool(alarma.iloc[-1]),
        "ratio_7d": float(ratio_7d.iloc[-1]),
        "sesgo_7d": float(sesgo_7d.iloc[-1]),
    }


def evaluar(ruta_registro=RUTA_REGISTRO):
    """Evalúa todos los días completos del registro y guarda la tabla diaria.
    Recalcula la tabla entera cada vez (idempotente)."""
    if not Path(ruta_registro).exists():
        print(f"No existe el registro de predicciones: {ruta_registro}")
        return None
    registro = pd.read_parquet(ruta_registro)
    tabla = pd.read_parquet(TABLA_FEATURES)

    cruzado = cruzar_con_real(registro, tabla)
    if cruzado.empty:
        print("Ningún día completo que evaluar (¿falta el precio real en tabla_features?)")
        return None

    metricas = calcular_metricas(cruzado)

    RUTA_EVALUACION.parent.mkdir(parents=True, exist_ok=True)
    metricas.to_parquet(RUTA_EVALUACION)   # sin index=False: el índice es la fecha

    print(f"Evaluación guardada: {len(metricas)} días | {RUTA_EVALUACION}")

    # El estado se lee de la tabla (última fila), no se recalcula.
    ultimo = metricas.iloc[-1]
    print(f"Degradación: ratio 7d {ultimo['ratio_7d']:.2f} | sesgo 7d {ultimo['sesgo_7d']:+.1f} €")
    if ultimo["alarma"]:
        print("ALARMA: el modelo se está degradando (ratio o sesgo fuera de umbral "
              f"{DIAS_SEGUIDOS} días seguidos). Revisar: ¿cambio de nivel de precios? ¿datos? "
              "El reentreno semanal puede no bastar.")
    return metricas


if __name__ == "__main__":
    evaluar()


    
