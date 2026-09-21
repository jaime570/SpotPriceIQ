from prefect import flow, task


@task
def descargar():
    print("Descargando datos de la fuente...")
    return 100  # simula "100 registros descargados"


@task
def procesar(n_registros):
    print(f"Procesando {n_registros} registros...")
    return n_registros * 2


@flow(name="ingesta-demo")
def ingesta():
    n = descargar()          # llamada a un task
    resultado = procesar(n)  # el resultado de uno alimenta al otro
    print(f"Flow terminado. Resultado: {resultado}")


if __name__ == "__main__":
    ingesta()