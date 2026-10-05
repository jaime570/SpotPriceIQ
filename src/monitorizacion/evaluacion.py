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

def evaluar(ruta_registro=RUTA_REGISTRO):
    """Evalúa todos los días completos del registro: una fila por día con las
    métricas de error, el regret para K = 1, 2 y 4 y medias móviles de 7 y 30
    días. Recalcula la tabla entera cada vez (idempotente) y la guarda."""

    if not Path(ruta_registro).exists():
        print(f"No existe el registro de predicciones: {ruta_registro}")
        return None
    registro = pd.read_parquet(ruta_registro)
    tabla = pd.read_parquet(TABLA_FEATURES)

    cruzado = cruzar_con_real(registro, tabla)
    if cruzado.empty:
        print("Ningún día completo que evaluar (¿falta el precio real en tabla_features?)")
        return None

    columnas = ["prediccion", "naive_d1", "precio_real"]
    por_dia = cruzado.groupby("fecha_objetivo")[columnas]

    metricas = por_dia.apply(metricas_dia)

    for k in (1, 2, 4):
        regret = por_dia.apply(regret_dia, k=k).add_suffix(f"_k{k}")
        metricas = metricas.join(regret)

    for col in ["mae_modelo", "mae_naive"]:
        metricas[f"{col}_7d"] = metricas[col].rolling("7D", min_periods=5).mean()
        metricas[f"{col}_30d"] = metricas[col].rolling("30D", min_periods=20).mean()

    RUTA_EVALUACION.parent.mkdir(parents=True, exist_ok=True)
    metricas.to_parquet(RUTA_EVALUACION)   # sin index=False: el índice es la fecha

    print(f"Evaluación guardada: {len(metricas)} días | {RUTA_EVALUACION}")
    return metricas

if __name__ == "__main__":
    evaluar()
    


    
