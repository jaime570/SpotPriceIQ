from prefect import task, flow
from src.monitorizacion.evaluacion import evaluar
from src.orquestacion.pipeline_diario import pipeline_diario

@task
def tarea_evaluacion():
    """Evalúa todos los días completos del registro: una fila por día con las
    métricas de error, el regret para K = 1, 2 y 4 y medias móviles de 7 y 30
    días. Recalcula la tabla entera cada vez (idempotente) y la guarda."""
    return evaluar()
@flow(name='evaluacion-tarde', log_prints=True)
def evaluacion_tarde():
    """Orquesta la evaluación de los días completos y guarda la tabla diaria."""
    pipeline_diario()
    tarea_evaluacion()

if __name__ == "__main__": evaluacion_tarde()