import json, pandas as pd
from pathlib import Path
from src.features.feature_sets import get_feature_sets

feats = get_feature_sets(pd.read_parquet("data/processed/tabla_features.parquet"))["predictivo"]
p = Path("model_champion/model_meta.json")
meta = json.loads(p.read_text())
meta["features"] = feats
p.write_text(json.dumps(meta, indent=2))
print("features añadidas:", len(feats))