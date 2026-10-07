# Alternativa de programación con Prefect (serve). En producción local se usa el
# Programador de tareas de Windows (script/run_*.bat); en la nube, GitHub Actions.
# Útil si se despliega con Prefect Cloud o un servidor Prefect propio.
from prefect import serve
from prefect.schedules import Cron

from src.orquestacion.reentrenamiento import reentrenamiento

if __name__ == "__main__":
    retrain = reentrenamiento.to_deployment(
        name="reentrenamiento-semanal",
        schedule=Cron("0 10 * * 1", timezone="Europe/Madrid"),  # lunes a las 10:00 (hora de Madrid)
        concurrency_limit=1,
    )
    serve(retrain)