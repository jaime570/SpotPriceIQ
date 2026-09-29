"""
Comprobación: ¿las previsiones ESIOS de MAÑANA cambian después de la subasta?

Cada ejecución guarda una "foto" de los valores de mañana en data/monitoring/check_esios/.
Si hay 2 o más fotos del mismo día objetivo, compara la primera con la última.

Uso (desde la raíz):
  1) antes de las 12:00  -> python -m scripts.check_publicacion_esios
  2) por la noche (~21h) -> python -m scripts.check_publicacion_esios
"""
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.ingesta.esios import descargar_indicador, parsear_indicador, INDICADORES_PREDICTIVO

ROOT = Path(__file__).resolve().parents[1]            # raíz del proyecto
CARPETA = ROOT / "data" / "monitoring" / "check_esios"
CARPETA.mkdir(parents=True, exist_ok=True)

USADAS = ["demanda_prevista_diaria", "eolica_prevista_d1",
          "solar_fv_prevista", "solar_termica_prevista"]

ahora = datetime.now()
sello = ahora.strftime("%Y%m%d_%H%M")                  
fecha_mañana = date.today() + timedelta(days=1)
inicio, fin = f"{fecha_mañana}T00:00:00", f"{fecha_mañana}T23:59:59"

print(f"Ejecutado: {ahora:%Y-%m-%d %H:%M} | Día objetivo: {fecha_mañana}\n")

# ---------- 1) Foto de los valores de mañana ----------
for nombre in USADAS:
    id_ind = INDICADORES_PREDICTIVO[nombre]
    data = descargar_indicador.fn(nombre, id_ind, inicio, fin)
    tabla = parsear_indicador.fn(data, nombre)

    nombre_oficial = data["indicator"].get("name", "?")
    actualizado = data["indicator"].get("values_updated_at", "?")
    print(f"{nombre:<25} id={id_ind:<5} horas: {tabla[nombre].notna().sum():>3}  "
          f"actualizado: {actualizado}\n{'':<25} nombre oficial: {nombre_oficial}")

    destino = CARPETA / f"{nombre}__obj{fecha_mañana}__{sello}.csv"
    tabla.to_csv(destino, index=False)

# ---------- 2) Comparar primera vs última foto del mismo día objetivo ----------
print("\n--- Comparación de fotos (mismo día objetivo) ---")
for nombre in USADAS:
    fotos = sorted(CARPETA.glob(f"{nombre}__obj{fecha_mañana}__*.csv"))
    if len(fotos) < 2:
        print(f"{nombre:<25} solo {len(fotos)} foto(s): vuelve a ejecutarlo más tarde")
        continue
    primera = pd.read_csv(fotos[0])
    ultima = pd.read_csv(fotos[-1])
    cmp = primera.merge(ultima, on="datetime_utc", suffixes=("_antes", "_despues"))
    diff = (cmp[f"{nombre}_despues"] - cmp[f"{nombre}_antes"]).abs()
    print(f"{nombre:<25} {fotos[0].stem[-13:]} -> {fotos[-1].stem[-13:]} | "
          f"horas que cambian: {(diff > 0).sum():>2}/24 | cambio máx: {diff.max():.1f} | "
          f"cambio medio: {diff.mean():.1f}")