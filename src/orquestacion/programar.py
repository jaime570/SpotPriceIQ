from prefect.schedules import Cron
from prefect import serve
from src.orquestacion.reentrenamiento import reentrenamiento

if __name__ == "__main__":
    retrain = reentrenamiento.to_deployment(
        name="reentrenamiento-semanal",
        schedule=Cron("0 3 * * 1", timezone="Europe/Madrid"),  # lunes a las 3:00 (hora de Madrid)
        concurrency_limit=1,
    )
    serve(retrain)
