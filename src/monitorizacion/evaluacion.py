import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLA_FEATURES = ROOT / "data" / "processed" / "tabla_features.parquet"
RUTA_REGISTRO = ROOT / "data" / "monitoring" / "predicciones.parquet"
RUTA_EVALUACION = ROOT / "data" / "monitoring" / "evaluacion_diaria.parquet"

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
    pred_centrada = dia["prediccion"] - dia["prediccion"].mean()
    real_centrada = dia["precio_real"] - dia["precio_real"].mean()
    mae_forma = (pred_centrada - real_centrada).abs().mean() 
    spearman =  dia["prediccion"].corr(dia["precio_real"], method="spearman")
    
    return pd.Series({"mae_modelo": mae_modelo, "mae_naive": mae_naive, "mae_forma": mae_forma, "spearman": spearman})

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

def calcular_metricas(cruzado):
    """Métricas por día (error, regret K=1/2/4) y medias móviles de 7 y 30 días.
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

    return metricas


def detectar_degradacion(cruzado, umbral=1.3, umbral_sesgo=14, dias_seguidos=3):
    """Alarma de degradación del modelo (calibrada en la simulación de la 9.3):
    salta si durante `dias_seguidos` días seguidos se cumple
      - ratio MAE 7d modelo/naive > `umbral`, o
      - |sesgo 7d| > `umbral_sesgo` €/MWh (el modelo falla siempre hacia el mismo lado).
    Devuelve un dict con la decisión y los últimos valores, para poder registrarlos."""
    por_dia = cruzado["fecha_objetivo"]
    error = cruzado["prediccion"] - cruzado["precio_real"]

    mae_dia = error.abs().groupby(por_dia).mean()
    naive_dia = (cruzado["naive_d1"] - cruzado["precio_real"]).abs().groupby(por_dia).mean()
    sesgo_dia = error.groupby(por_dia).mean()

    mae_7d = mae_dia.rolling("7D", min_periods=5).mean()
    naive_7d = naive_dia.rolling("7D", min_periods=5).mean()
    sesgo_7d = sesgo_dia.rolling("7D", min_periods=5).mean()
    ratio_7d = mae_7d / naive_7d

    alarma_dia = (ratio_7d > umbral) | (sesgo_7d.abs() > umbral_sesgo)

    return {
        "alarma": bool(alarma_dia.tail(dias_seguidos).all()),
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
    
    estado = detectar_degradacion(cruzado)
    print(f"Degradación: ratio 7d {estado['ratio_7d']:.2f} | sesgo 7d {estado['sesgo_7d']:+.1f} €")
    if estado["alarma"]:
        print("ALARMA: el modelo se está degradando (ratio o sesgo fuera de umbral 3 días seguidos). "
              "Revisar: ¿cambio de nivel de precios? ¿datos? El reentreno semanal puede no bastar.")
    return metricas


if __name__ == "__main__":
    evaluar()
    


    
