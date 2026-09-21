import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from prefect import flow, task

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
AEMET_URL = "https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos"

ESTACIONES = [
    "0076", "3129", "3434X", "3463Y", "4064Y", "5304Y", "5641X", "5702X",
    "8414A", "9051", "9069C", "9287A", "9434", "9491X", "9501X", "9894Y",
]

@task(retries=5, retry_delay_seconds=10)
def descargar_estacion(indicativo: str, inicio: str, fin: str) -> list:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("AEMET_API_KEY").strip().strip("'")
    headers = {"accept": "application/json", "api_key": api_key}
    url = f"{AEMET_URL}/fechaini/{inicio}/fechafin/{fin}/estacion/{indicativo}"

    r1 = requests.get(url, headers=headers, timeout=30)
    r1.raise_for_status()
    j = r1.json()
    estado = j.get("estado")
    if estado == 404:                                   # no hay datos: válido
        return []
    if estado != 200:                                   # limitado/otro error -> reintentar
        raise RuntimeError(f"AEMET estado {estado} en {indicativo}: {j.get('descripcion')}")

    datos_url = j["datos"]
    r2 = requests.get(datos_url, headers=headers, timeout=30)
    r2.raise_for_status()
    datos = r2.json()
    if not isinstance(datos, list):                     # respuesta rara -> reintentar
        raise RuntimeError(f"AEMET no devolvió lista en {indicativo}: {str(datos)[:120]}")
    return datos             # lista de registros diarios


@task
def aplanar(resultados: list) -> pd.DataFrame:
    lista_plana = [
        registro
        for res in resultados if isinstance(res, list)
        for registro in res if isinstance(registro, dict)
    ]
    return pd.DataFrame(lista_plana)


@task
def guardar(tabla: pd.DataFrame) -> Path:
    INTERIM.mkdir(parents=True, exist_ok=True)
    destino = INTERIM / "tabla_AEMET"
    if destino.exists():
        previa = pd.read_csv(destino)
        tabla = pd.concat([previa, tabla], ignore_index=True)
        tabla = tabla.drop_duplicates(subset=["indicativo", "fecha"], keep="last")
    tabla.to_csv(destino, index=False, encoding="utf-8")
    return destino


@flow(name="ingesta-aemet", log_prints=True)
def ingesta_aemet(inicio: str | None = None, fin: str | None = None):
    if inicio is None:                              # ventana reciente por defecto
        hoy = date.today()
        inicio = (hoy - timedelta(days=9)).strftime("%Y-%m-%dT00:00:00UTC")
        fin = (hoy - timedelta(days=2)).strftime("%Y-%m-%dT23:59:59UTC")
    resultados = []
    for ind in ESTACIONES:
        resultados.append(descargar_estacion(ind, inicio, fin))
    tabla = aplanar(resultados)
    ruta = guardar(tabla)
    print(f"Guardado {tabla.shape} en {ruta}")


if __name__ == "__main__":
    ingesta_aemet()