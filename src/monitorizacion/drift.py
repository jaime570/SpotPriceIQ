from evidently import Report
from evidently.presets import DataDriftPreset
import pandas as pd 
import numpy as np
from pathlib import Path
from src.features import feature_sets

PATH = Path(__file__).resolve().parents[2]
PROCESSED = PATH / "data" / "processed"
TABLA_FEATURES = PROCESSED / "tabla_features.parquet"
REPORTS = PATH / "reports"
def cargar_partir():
    df = pd.read_parquet(TABLA_FEATURES)
    df["datetime_utc"]= pd.to_datetime(df["datetime_utc"], utc=True)
    referencia = df[df["datetime_utc"] < pd.Timestamp("2025-01-01", tz="UTC")]
    actual = df[df["datetime_utc"] >= pd.Timestamp("2025-01-01", tz="UTC")]
    return(referencia,actual)

def generar_report_drift():
    referencia, actual = cargar_partir()
    feats = feature_sets.get_feature_sets(referencia)["predictivo"]
    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=referencia[feats], current_data=actual[feats])
    snapshot.save_html(str(REPORTS / "data_drift.html"))
    return(snapshot)

if __name__ == "__main__":
    generar_report_drift()
