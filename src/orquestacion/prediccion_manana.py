from datetime import time

import pandas as pd
from prefect import flow, get_run_logger, task

from src.monitorizacion.prediccion import predecir_d1
from src.notificaciones.telegram import avisar, aviso_fallo
from src.orquestacion.pipeline_diario import pipeline_diario

# Cierre de la subasta diaria de OMIE: 12:00 hora peninsular.
# Se compara en Europe/Madrid (no en UTC) para que el límite siga siendo
# "las 12 en España" tanto en verano (UTC+2) como en invierno (UTC+1).
CIERRE_SUBASTA = time(12, 0)
MARGEN_INICIO = time(11, 50) 


def antes_de(limite: time, ahora: pd.Timestamp | None = None) -> bool:
    """True si la hora en Madrid es anterior a limite.
    ahora es la hora actual por defecto; se puede pasar otra (con zona horaria) para los tests.
    Se compara en Europe/Madrid, no en UTC: el cierre es "las 12 en España" en verano y en invierno."""
    if ahora is None:
        ahora = pd.Timestamp.now(tz="UTC")
    return ahora.tz_convert("Europe/Madrid").time() < limite

@task
def tarea_prediccion():
    """Predice el día siguiente y guarda las predicciones en el registro."""
    return predecir_d1()


@flow(name="prediccion-manana", log_prints=True, on_failure=[aviso_fallo], on_crashed=[aviso_fallo])
def prediccion_manana():
    """Orquesta la predicción del día siguiente.
    Puerta horaria: una predicción D+1 posterior al cierre de la subasta no sirve
    para pujar y contaminaría el registro de producción (keep="first" la haría
    permanente), así que fuera de hora no se ingesta ni se predice.
    Avisos por Telegram: ⚠️ si no se predice, ✅ con las horas si se predice (heartbeat)."""
    logger = get_run_logger()

    if not antes_de(MARGEN_INICIO):
        logger.warning(f"Son más de las {MARGEN_INICIO:%H:%M} (Madrid): subasta cerrada "
                       "o a punto de cerrar. No se ingesta ni se predice; el día queda como hueco.")
        avisar(f"⚠️ Sin predicción: son más de las {MARGEN_INICIO:%H:%M}, subasta cerrada.")
        return None

    pipeline_diario()

    # Segunda comprobación: por si la ingesta se alargó y ya pasamos el cierre.
    if not antes_de(CIERRE_SUBASTA):
        logger.warning("La ingesta terminó después del cierre de la subasta: no se guarda la predicción.")
        avisar("⚠️ Sin predicción: la ingesta terminó después del cierre de la subasta.")
        return None

    registro = tarea_prediccion()
    if registro is None:
        avisar("⚠️ Sin predicción: no había horas de D+1 que predecir.")
        return None

    avisar(_resumen(registro))
    return registro

def _resumen(registro):
    """Mensaje ✅ para Telegram: día, modelo y 4 horas más baratas/caras (hora de Madrid)."""
    perfil = registro.assign(
        hora=registro["datetime_utc"].dt.tz_convert("Europe/Madrid").dt.hour
    )
    baratas = perfil.nsmallest(4, "prediccion").sort_values("hora")
    caras = perfil.nlargest(4, "prediccion").sort_values("hora")
    dia = registro["fecha_objetivo"].iloc[0]
    version = registro["modelo_version"].iloc[0]

    lista_baratas = ", ".join(f"{h:02d}h {p:.0f}€" for h, p in zip(baratas["hora"], baratas["prediccion"], strict=True))
    lista_caras = ", ".join(f"{h:02d}h {p:.0f}€" for h, p in zip(caras["hora"], caras["prediccion"], strict=True))

    return (f"✅ Predicción {dia:%d-%m} guardada (v{version})\n"
            f"Baratas: {lista_baratas}\n"
            f"Caras: {lista_caras}")

if __name__ == "__main__":
    prediccion_manana()