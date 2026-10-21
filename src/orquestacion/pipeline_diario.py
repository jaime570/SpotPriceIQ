from prefect import flow, task

from src.ingesta.aemet import ingesta_aemet
from src.ingesta.esios import ingesta_esios
from src.ingesta.omie import ingesta_omie
from src.notificaciones.telegram import avisar
from src.procesamiento.concatenacion import construir_tabla_maestra
from src.procesamiento.etl import ejecutar_etl
from src.procesamiento.features import ejecutar_features
from src.validation.control_datos import validar_cobertura


@task(retries=0)
def tarea_cobertura():
    """Puerta de entrada: valida cobertura de las fuentes crudas.
    Si una fuente crítica falla, para el flow. Si una no crítica (AEMET) está
    desactualizada, avisa por Telegram y sigue (se imputa)."""
    resumen, avisos = validar_cobertura()
    for aviso in avisos:
        avisar(f"⚠️ Fuente no crítica desactualizada, se sigue con imputación: {aviso}")
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
    ingesta_aemet()
    tarea_cobertura()
    tarea_concat()
    tarea_etl()
    tarea_features()


if __name__ == "__main__":
    pipeline_diario()