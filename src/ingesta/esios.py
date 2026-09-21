import os
from functools import reduce
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv
from prefect import flow, task
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
ESIOS_URL = "https://api.esios.ree.es/indicators"

INDICADORES_PREDICTIVO = {
    "demanda_prevista": 544,
    "demanda_prevista_diaria": 460,
    "eolica_prevista": 541,
    "eolica_prevista_d1": 1777,
    "solar_fv_prevista": 542,
    "solar_termica_prevista": 543,
}


@task(retries=3, retry_delay_seconds=5)
def descargar_indicador(nombre: str, id_indicador: int, inicio: str, fin: str) -> dict:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("ESIOS_API_KEY").strip().strip("'")
    headers = {
        "Accept": "application/json; application/vnd.esios-api-v1+json",
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }
    params = {"start_date": inicio, "end_date": fin, "time_trunc": "hour"}
    r = requests.get(f"{ESIOS_URL}/{id_indicador}", headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


@task
def parsear_indicador(data: dict, nombre: str) -> pd.DataFrame:
    valores = data["indicator"]["values"]
    df = pd.DataFrame(valores)
    df = df.rename(columns={"value": nombre})
    df["datetime_utc"] = pd.to_datetime(df["datetime"], utc=True)
    return df[[nombre, "datetime_utc"]]


@task
def unir(tablas: list[pd.DataFrame]) -> pd.DataFrame:
    return reduce(lambda a, b: a.merge(b, on="datetime_utc", how="outer"), tablas)


@task
def guardar(tabla: pd.DataFrame) -> Path:
    INTERIM.mkdir(parents=True, exist_ok=True)
    destino = INTERIM / "tabla_ESIOS"
    if destino.exists():
        previa = pd.read_csv(destino, encoding="utf-8")
        previa["datetime_utc"] = pd.to_datetime(previa["datetime_utc"], utc=True)
        tabla = pd.concat([previa, tabla], ignore_index=True)
        tabla = tabla.drop_duplicates(subset=["datetime_utc"], keep="last")
    tabla = tabla.sort_values("datetime_utc").reset_index(drop=True)
    tabla.to_csv(destino, index=False, encoding="utf-8")
    return destino


@flow(name="ingesta-esios", log_prints=True)
def ingesta_esios(inicio: str | None = None, fin: str | None = None):
    if inicio is None:
        hoy = date.today()
        inicio = (hoy - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
        fin    = (hoy + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    tablas = []
    for nombre, id_ind in INDICADORES_PREDICTIVO.items():
        data = descargar_indicador(nombre, id_ind, inicio, fin)
        tablas.append(parsear_indicador(data, nombre))
    tabla = unir(tablas)
    ruta = guardar(tabla)
    print(f"Guardado {tabla.shape} en {ruta}")


if __name__ == "__main__":
    ingesta_esios()