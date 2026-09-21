from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from prefect import flow, task

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "omie"
INTERIM = Path(__file__).resolve().parents[2] / "data" / "interim"


@task(retries=3, retry_delay_seconds=5)
def descargar_dia(fecha: str) -> Path:
    url = f"https://www.omie.es/es/file-download?parents=marginalpdbc&filename=marginalpdbc_{fecha}.1"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    RAW.mkdir(parents=True, exist_ok=True)
    destino = RAW / f"marginalpdbc_{fecha}.1"
    destino.write_bytes(r.content)
    return destino


@task
def parsear(fichero: Path) -> pd.DataFrame:
    names = ["ano", "mes", "dia", "hora", "precio_espana", "precio_portugal", "vacia"]
    tabla = pd.read_csv(
        fichero,
        sep=";",
        skiprows=1,
        skipfooter=1,
        names=names,
        engine="python",
    )
    return tabla


@task
def guardar(tabla: pd.DataFrame) -> Path:
    INTERIM.mkdir(parents=True, exist_ok=True)
    destino = INTERIM / "tabla_OMIE"

    if destino.exists():
        previa = pd.read_csv(destino)
        tabla = pd.concat([previa, tabla], ignore_index=True)
        tabla = tabla.drop_duplicates(subset=["ano", "mes", "dia", "hora"], keep="last")

    tabla.to_csv(destino, index=False, encoding="utf-8")
    return destino


@flow(name="ingesta-omie", log_prints=True)
def ingesta_omie(fecha: str | None = None):
    if fecha is None:
        fecha = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
    fichero = descargar_dia(fecha)
    tabla = parsear(fichero)
    ruta = guardar(tabla)
    print(f"Guardado en {ruta}")


if __name__ == "__main__":
    ingesta_omie()