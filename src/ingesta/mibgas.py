"""Ingesta de MIBGAS: precio de referencia del gas en España (producto GDAES_D+1).

MIBGAS publica un CSV por año. El del año en curso no es del año natural, sino
de los últimos 12 meses móviles, así que al unir ficheros hay fechas repetidas
(se quitan por `fecha_entrega`). Para la ingesta diaria basta ese fichero; el
histórico desde 2023 se descarga una vez con `ingesta_mibgas(anios=[...])`.

La tabla de interim guarda el precio por fecha de entrega. El desplazamiento
point-in-time (gas_d1 del día T = entrega T−1) se hace en las features, no aquí.
"""

import io
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from prefect import flow, task

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "mibgas"
INTERIM = ROOT / "data" / "interim"
DESTINO = INTERIM / "tabla_MIBGAS"

URL = "https://www.mibgas.es/es/file-access/MIBGAS_Data_{anio}.csv?path=AGNO_{anio}/XLS"
PRODUCTO = "GDAES_D+1"


@task(retries=3, retry_delay_seconds=5)
def descargar_anio(anio: int) -> Path:
    r = requests.get(URL.format(anio=anio), timeout=30)
    r.raise_for_status()
    RAW.mkdir(parents=True, exist_ok=True)
    destino = RAW / f"MIBGAS_Data_{anio}.csv"
    # Se sobrescribe: el fichero del año en curso cambia cada día
    destino.write_bytes(r.content)
    return destino


def parsear_texto(texto: str) -> pd.DataFrame:
    """CSV de MIBGAS (texto) -> DataFrame [fecha_entrega, precio_gas], ordenado.

    Separada de la task para poder testearla sin ficheros ni Prefect.
    """
    # La primera línea es "Fecha Emisión :..."; la cabecera está en la segunda
    df = pd.read_csv(io.StringIO(texto), sep=";", skiprows=1)
    df = df[df["Product"] == PRODUCTO]

    # El nombre de la columna de precio lleva la unidad, se busca por contenido
    col_precio = [c for c in df.columns if "Price" in c][0]
    df = df[["First Day Delivery", col_precio]].rename(
        columns={"First Day Delivery": "fecha_entrega", col_precio: "precio_gas"}
    )
    df["fecha_entrega"] = pd.to_datetime(df["fecha_entrega"], format="%d/%m/%Y")
    df["precio_gas"] = pd.to_numeric(df["precio_gas"], errors="coerce")
    df = df.dropna(subset=["precio_gas"])
    return df.sort_values("fecha_entrega").reset_index(drop=True)


@task
def parsear(fichero: Path) -> pd.DataFrame:
    # El fichero es UTF-8; con la codificación por defecto de Windows la "ó" sale rota
    return parsear_texto(fichero.read_text(encoding="utf-8"))


def unir(previa: pd.DataFrame, nueva: pd.DataFrame) -> pd.DataFrame:
    """Une dos tablas de gas sin duplicar fechas; si una fecha se repite, gana la nueva."""
    tabla = pd.concat([previa, nueva], ignore_index=True)
    tabla = tabla.drop_duplicates(subset="fecha_entrega", keep="last")
    return tabla.sort_values("fecha_entrega").reset_index(drop=True)


@task
def guardar(tabla: pd.DataFrame) -> Path:
    INTERIM.mkdir(parents=True, exist_ok=True)
    if DESTINO.exists():
        previa = pd.read_csv(DESTINO, parse_dates=["fecha_entrega"])
        tabla = unir(previa, tabla)
    tabla.to_csv(DESTINO, index=False, encoding="utf-8", date_format="%Y-%m-%d")
    return DESTINO


@flow(name="ingesta-mibgas", log_prints=True)
def ingesta_mibgas(anios: list[int] | None = None):
    """Descarga los ficheros anuales de MIBGAS y actualiza tabla_MIBGAS.

    Por defecto solo el año en curso (últimos 12 meses). En enero se pide también
    el año anterior, por si el fichero del año nuevo todavía no está publicado.
    Falla solo si no se puede descargar ningún año: la decisión de bloquear o
    seguir con el último precio la toma la cobertura (caducidad de 3 días).
    """
    if anios is None:
        hoy = date.today()
        anios = [hoy.year - 1, hoy.year] if hoy.month == 1 else [hoy.year]

    tablas, fallidos = [], []
    for anio in anios:
        try:
            tablas.append(parsear(descargar_anio(anio)))
        except Exception as e:
            print(f"MIBGAS {anio}: no se pudo descargar ({e})")
            fallidos.append(anio)

    if not tablas:
        raise RuntimeError(f"MIBGAS: no se pudo descargar ningún año ({fallidos})")

    nueva = tablas[0]
    for t in tablas[1:]:
        nueva = unir(nueva, t)
    destino = guardar(nueva)
    print(f"MIBGAS: {len(nueva)} días descargados, de {nueva['fecha_entrega'].min().date()} "
          f"a {nueva['fecha_entrega'].max().date()} -> {destino.name}")
    return nueva


if __name__ == "__main__":
    ingesta_mibgas()
