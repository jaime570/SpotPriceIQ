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
def ingesta_omie(inicio: date | None = None, fin: date | None = None):
    hoy = date.today()
    if inicio is None:
        inicio = hoy - timedelta(days=7)
    if fin is None:
        fin = hoy + timedelta(days=1)
    rango_dias = pd.date_range(inicio, fin, freq="D")
    tablas = []
    for dia in rango_dias:
        fecha_str= dia.strftime("%Y%m%d")
        try:
            fichero = descargar_dia(fecha_str)
            tabla = parsear(fichero)
            tablas.append(tabla)
        except Exception:
            if dia.date() == hoy + timedelta(days=1):
                print(f"OMIE {fecha_str}: aún no publicado (normal antes de las 13h)")
            else:
                print(f"Omie no pudo descargar: {fecha_str}")
    if tablas: 
        tabla_final =pd.concat(tablas, ignore_index=True)
        guardar(tabla_final)
        print(f"se han guardado {len(tablas)} días")
    else:
        print("No se ha descargado ningún dia ")


if __name__ == "__main__":
    ingesta_omie()




            



