from prefect import serve

from src.orquestacion.pipeline_diario import pipeline_diario
from src.orquestacion.reentrenamiento import reentrenamiento

if __name__ == "__main__":
    ingesta = pipeline_diario.to_deployment(
        name="ingesta-diaria",
        cron="0 14 * * *",          # cada día a las 14:00
    )
    retrain = reentrenamiento.to_deployment(
        name="reentrenamiento-semanal",
        cron="0 3 * * 1",  
        concurrency_limit=1,         # lunes a las 3:00
    )
    serve(ingesta, retrain)
