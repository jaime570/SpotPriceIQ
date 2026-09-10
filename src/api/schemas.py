from pydantic import BaseModel

class PredictRequest(BaseModel):
    features: dict[str, float]          # nombre_feature -> valor

class PredictResponse(BaseModel):
    prediccion_eur_mwh: float
    modelo: str
    version: str