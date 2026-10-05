from prefect import task, flow
from src.orquestacion.pipeline_diario import pipeline_diario
from src.monitorizacion.prediccion import predecir_d1 

@task 
def tarea_prediccion():
    """Predice el día siguiente y guarda las predicciones en el registro."""
    return predecir_d1()

@flow(name='prediccion-manana', log_prints=True)
def prediccion_manana():
    """Orquesta la predicción del día siguiente."""
    pipeline_diario()
    tarea_prediccion()

if __name__ == "__main__":
    prediccion_manana()
