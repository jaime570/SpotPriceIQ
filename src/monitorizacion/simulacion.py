import pandas as pd
from pathlib import Path
from src.features.feature_sets import get_feature_sets      
from src.orquestacion.reentrenamiento import entrenar        
from src.monitorizacion.evaluacion import metricas_dia, regret_dia  

ROOT = Path(__file__).resolve().parents[2]
TABLA_FEATURES = ROOT / "data" / "processed" / "tabla_features.parquet"

def predecir_dia_sim(modelo, df, dia, feats):
    """Predice las 24 horas de 'dia' con 'modelo' y devuelve sus filas del
    registro simulado (mismas columnas que el registro real + precio_real)."""
    df_dia = df[(df["fecha"] == dia) & (df["entrenable"])].copy()
    predicciones = modelo.predict(df_dia[feats])
    registro_dia = (df_dia[["datetime_utc", "fecha", "precio_lag_24", "precio_espana"]]
                    .rename(columns={"fecha": "fecha_objetivo", "precio_lag_24": "naive_d1", "precio_espana": "precio_real"})
                    .reset_index(drop=True))
    registro_dia["prediccion"] = predicciones
    return registro_dia

def politica_congelado(dia,historial,reentrenos):
    """Política 'congelado': nunca reentrena (se usa el modelo inicial toda la simulación)."""
    
    return False

def politica_mensual(dia, historial, reentrenos):
    """Devuelve True si toca reentrenar (una vez al mes, el primer día del mes)."""
    
    return dia.day == 1
def politica_semanal(dia, historial, reentrenos):
    """Devuelve True si toca reentrenar (una vez a la semana, el primer día de la semana)."""
    
    return dia.weekday() == 0

def simular(df, politica, inicio, fin):
    """Simula la producción día a día entre `inicio` y `fin`.

    Para cada día D: pregunta a `politica` si hay que reentrenar (si aún no hay
    modelo, entrena siempre), entrena solo con datos anteriores a D, predice las
    24 horas de D y guarda el resultado. Los días sin datos se saltan.

    Devuelve (registro_simulado, reentrenos): el registro con las mismas columnas
    que el real + precio_real, y la lista de fechas en que se reentrenó.
    """
    feats = get_feature_sets(df)["predictivo"]
    dias = pd.date_range(inicio, fin)
    registro_simulado = []
    reentrenos = []
    modelo = None

    for dia in dias:
        if registro_simulado:
            historial = pd.concat(registro_simulado, ignore_index=True)
        else:
            historial = pd.DataFrame()

        if modelo is None or politica(dia, historial, reentrenos):
            print(f"Simulación: reentrenando para {dia.date()}")
            modelo = entrenar.fn(df[df["fecha"] < dia])
            reentrenos.append(dia)

        registro_dia = predecir_dia_sim(modelo, df, dia, feats)

        if registro_dia.empty:
            print(f"Simulación: día {dia.date()} sin datos, se salta")
            continue

        registro_simulado.append(registro_dia)

    return pd.concat(registro_simulado, ignore_index=True), reentrenos

def politica_trigger(dia, historial, reentrenos, umbral=1.3, umbral_sesgo=14,
                     dias_seguidos=3, enfriamiento=7):
    """Reentrena si, durante `dias_seguidos` días seguidos, salta alguna alarma:
      - ratio MAE 7d modelo/naive > `umbral` (el modelo pierde claramente contra el naive), o
      - |sesgo 7d| > `umbral_sesgo` €/MWh (el modelo se equivoca siempre hacia el mismo lado:
        huella de un cambio de nivel de precios).
    Respeta `enfriamiento` días desde el último reentreno."""
    if historial.empty:
        return False
    if (dia - reentrenos[-1]).days < enfriamiento:
        return False

    por_dia = historial["fecha_objetivo"]

    # MAE diario del modelo y del naive (error sin signo)
    mae_dia = (historial["prediccion"] - historial["precio_real"]).abs().groupby(por_dia).mean()
    naive_dia = (historial["naive_d1"] - historial["precio_real"]).abs().groupby(por_dia).mean()

    # Sesgo diario (error CON signo): negativo = el modelo se queda corto
    sesgo_dia = (historial["prediccion"] - historial["precio_real"]).groupby(por_dia).mean()

    # Medias de 7 días de calendario (NaN hasta tener 5 días)
    mae_semanal = mae_dia.rolling("7D", min_periods=5).mean()
    naive_semanal = naive_dia.rolling("7D", min_periods=5).mean()
    sesgo_semanal = sesgo_dia.rolling("7D", min_periods=5).mean()

    ratio_semanal = mae_semanal / naive_semanal

    # Alarma de cada día: alguna de las dos condiciones (| = "o" para Series)
    alarma = (ratio_semanal > umbral) | (sesgo_semanal.abs() > umbral_sesgo)

    # Reentrenar solo si la alarma lleva `dias_seguidos` días seguidos encendida
    ultimos = alarma.tail(dias_seguidos)
    return ultimos.all()