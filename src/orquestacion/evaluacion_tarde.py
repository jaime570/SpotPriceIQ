from prefect import task, flow

from src.monitorizacion.evaluacion import evaluar
from src.notificaciones.telegram import avisar, aviso_fallo
from src.orquestacion.pipeline_diario import pipeline_diario


@task
def tarea_evaluacion():
    """Evalúa todos los días completos del registro: una fila por día con las
    métricas de error, el regret para K = 1, 2 y 4 y medias móviles de 7 y 30
    días. Recalcula la tabla entera cada vez (idempotente) y la guarda."""
    return evaluar()


@flow(name='evaluacion-tarde', log_prints=True, on_failure=[aviso_fallo], on_crashed=[aviso_fallo])
def evaluacion_tarde():
    """Orquesta la evaluación de los días completos y guarda la tabla diaria.
    Si la alarma de degradación está activa en el último día, avisa por Telegram."""
    pipeline_diario()
    metricas = tarea_evaluacion()

    if metricas is not None and bool(metricas["alarma"].iloc[-1]):
        ultimo = metricas.iloc[-1]
        avisar(f"🚨 Alarma de degradación: ratio 7d {ultimo['ratio_7d']:.2f} | "
               f"sesgo 7d {ultimo['sesgo_7d']:+.1f} € (fuera de umbral varios días seguidos). "
               "Revisar: ¿cambio de nivel de precios? ¿datos?")


if __name__ == "__main__":
    evaluacion_tarde()