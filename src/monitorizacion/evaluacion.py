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
    predichas = por_dia.transform("size")   
    con_real = por_dia.transform("count")   
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


def regret_dia(dia, k, col="prediccion"):
    """Regret de compra y venta de un día completo para las K horas elegidas según `col`.

    col="prediccion" (por defecto) → regret del modelo (comportamiento de siempre).
    col="naive_d1"                 → regret del naive: elegir las horas según el precio de ayer.
    Sirve de referencia: un regret del modelo solo es "bueno" si es menor que el del naive.
    El óptimo siempre se calcula con el precio real."""
    horas_baratas_elegidas = dia.nsmallest(k, col)
    pagado_elegidas = horas_baratas_elegidas["precio_real"].mean()
    horas_baratas_reales = dia.nsmallest(k, "precio_real")
    pagado_optimo = horas_baratas_reales["precio_real"].mean()
    regret_compra = pagado_elegidas - pagado_optimo

    horas_caras_elegidas = dia.nlargest(k, col)
    venta_elegidas = horas_caras_elegidas["precio_real"].mean()
    horas_caras_reales = dia.nlargest(k, "precio_real")
    venta_optimo = horas_caras_reales["precio_real"].mean()
    regret_venta = venta_optimo - venta_elegidas

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


def salud_7d(metricas, umbral=UMBRAL_RATIO, umbral_sesgo=UMBRAL_SESGO,
             dias_seguidos=DIAS_SEGUIDOS):
    """Salud del modelo por día: medias móviles 7d/30d del MAE, sesgo 7d, ratio 7d,
    condición y alarma de degradación. ÚNICO sitio donde se calculan: lo usan producción
    (calcular_metricas) y simulación (detectar_degradacion), así que no pueden divergir.
    `metricas` necesita una fila por día con índice de fecha y las columnas mae_modelo,
    mae_naive y sesgo (lo que devuelve metricas_dia aplicado por día)."""
    
    for col in ["mae_modelo", "mae_naive"]:
        metricas[f"{col}_7d"] = metricas[col].rolling("7D", min_periods=5).mean()
        metricas[f"{col}_30d"] = metricas[col].rolling("30D", min_periods=20).mean()

    metricas["sesgo_7d"] = metricas["sesgo"].rolling("7D", min_periods=5).mean()
    metricas["ratio_7d"] = metricas["mae_modelo_7d"] / metricas["mae_naive_7d"]
    metricas["condicion_alarma"], metricas["alarma"] = _alarma_por_dia(
        metricas["ratio_7d"], metricas["sesgo_7d"], umbral, umbral_sesgo, dias_seguidos
    )
    return metricas


def calcular_metricas(cruzado):
    """Métricas por día (error, sesgo, regret K=1/2/4), medias móviles de 7 y 30 días
    y estado de la alarma de degradación por día.
    Solo calcula: no lee ni guarda ficheros (sirve para producción y para simulación)."""
   
    columnas = ["prediccion", "naive_d1", "precio_real"]
    por_dia = cruzado.groupby("fecha_objetivo")[columnas]

    metricas = por_dia.apply(metricas_dia)

    for k in (1, 2, 4):
        regret = por_dia.apply(regret_dia, k=k).add_suffix(f"_k{k}")
        regret_naive = por_dia.apply(regret_dia, k=k, col="naive_d1").add_suffix(f"_naive_k{k}")
        metricas = metricas.join(regret).join(regret_naive)

    metricas = salud_7d(metricas)
    return metricas


def detectar_degradacion(cruzado, umbral=UMBRAL_RATIO, umbral_sesgo=UMBRAL_SESGO,
                         dias_seguidos=DIAS_SEGUIDOS):
    """Estado de la salud en el ÚLTIMO día (lo usa simulacion.py). Mismas piezas que
    producción (metricas_dia + salud_7d): no tiene cálculo propio, así no puede divergir."""
    
    columnas = ["prediccion", "naive_d1", "precio_real"]
    por_dia = cruzado.groupby("fecha_objetivo")[columnas]

    metricas = por_dia.apply(metricas_dia)
    metricas = salud_7d(metricas, umbral=umbral, umbral_sesgo=umbral_sesgo,
                        dias_seguidos=dias_seguidos)

    ultimo = metricas.iloc[-1]

    return {
        "alarma": bool(ultimo["alarma"]),
        "ratio_7d": float(ultimo["ratio_7d"]),
        "sesgo_7d": float(ultimo["sesgo_7d"]),
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


    
