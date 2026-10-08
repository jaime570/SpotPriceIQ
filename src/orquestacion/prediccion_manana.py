from datetime import time

import pandas as pd
from prefect import task, flow, get_run_logger

from src.orquestacion.pipeline_diario import pipeline_diario
from src.monitorizacion.prediccion import predecir_d1

# Cierre de la subasta diaria de OMIE: 12:00 hora peninsular.
# Se compara en Europe/Madrid (no en UTC) para que el límite siga siendo
# "las 12 en España" tanto en verano (UTC+2) como en invierno (UTC+1).
CIERRE_SUBASTA = time(12, 0)
MARGEN_INICIO = time(11, 50)  # el pipeline tarda ~1 min: no arrancar demasiado justo


def antes_de(limite: time) -> bool:
    """True si la hora actual en Madrid es anterior a `limite`."""
    return pd.Timestamp.now(tz="Europe/Madrid").time() < limite


@task
def tarea_prediccion():
    """Predice el día siguiente y guarda las predicciones en el registro."""
    return predecir_d1()


@flow(name="prediccion-manana", log_prints=True)
def prediccion_manana():
    """Orquesta la predicción del día siguiente.
    Puerta horaria: una predicción D+1 posterior al cierre de la subasta no sirve
    para pujar y contaminaría el registro de producción (keep="first" la haría
    permanente), así que fuera de hora no se ingesta ni se predice."""
    logger = get_run_logger()

    if not antes_de(MARGEN_INICIO):
        logger.warning(f"Son más de las {MARGEN_INICIO:%H:%M} (Madrid): subasta cerrada "
                       "o a punto de cerrar. No se ingesta ni se predice; el día queda como hueco.")
        return None

    pipeline_diario()

    # Segunda comprobación: por si la ingesta se alargó y ya pasamos el cierre.
    if not antes_de(CIERRE_SUBASTA):
        logger.warning("La ingesta terminó después del cierre de la subasta: no se guarda la predicción.")
        return None

    return tarea_prediccion()


if __name__ == "__main__":
    prediccion_manana()