from prefect import flow, task

from src.ingesta.aemet import ingesta_aemet
from src.ingesta.esios import ingesta_esios
from src.ingesta.mibgas import ingesta_mibgas
from src.ingesta.omie import ingesta_omie
from src.notificaciones.telegram import avisar
from src.procesamiento.concatenacion import construir_tabla_maestra
from src.procesamiento.etl import ejecutar_etl
from src.procesamiento.features import ejecutar_features
from src.validation.control_datos import validar_cobertura


@task(retries=0)
def tarea_cobertura():
    """Puerta de entrada: valida cobertura de las fuentes crudas.
    Si una fuente falla de forma bloqueante, para el flow. Si está desactualizada
    pero todavía se puede usar (AEMET: se imputa; MIBGAS de 1 a 3 días: último
    precio), avisa por Telegram y sigue. El aviso ya dice qué se hace."""
    resumen, avisos = validar_cobertura()
    for aviso in avisos:
        avisar(f"⚠️ Fuente desactualizada: {aviso}")
    return resumen

@task
def tarea_concat():
    """Construye la tabla maestra estructural (concat de OMIE + ESIOS + AEMET)."""
    return construir_tabla_maestra()


@task
def tarea_etl():
    """ETL: limpia AEMET, valida el warehouse y escribe la tabla procesada."""
    return ejecutar_etl()


@task
def tarea_features():
    """Feature engineering: imputa, añade calendario y lags, marca entrenables."""
    return ejecutar_features()


@flow(name="pipeline-diario", log_prints=True)
def pipeline_diario():
    ingesta_omie()
    ingesta_esios()
    # AEMET no es crítica: si su API falla, se sigue con los datos que ya hay
    # y la imputación cubre los días que falten (degradación controlada).
    try:
        ingesta_aemet()
    except Exception as e:
        avisar(f"⚠️ AEMET no responde, se sigue sin actualizarla (se imputa): {e}")
    # MIBGAS tampoco para el pipeline si su web falla: se sigue con el último precio
    # guardado y es la cobertura la que decide (aviso de 1 a 3 días, bloqueo si más).
    try:
        ingesta_mibgas()
    except Exception as e:
        avisar(f"⚠️ MIBGAS no responde, se sigue con el último precio guardado: {e}")
    tarea_cobertura()
    tarea_concat()
    tarea_etl()
    tarea_features()


if __name__ == "__main__":
    pipeline_diario()