import os
import sys
import yaml
import pandas as pd

# Ensure project root in path so 'app' package resolves when run as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.utils.data_loader import load_groundwater_data
from app.utils.features import build_monthly_features
from app.models.forecasting import train_per_district, save_models


def main():
    with open("config/app_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    excel_path = cfg["data"]["groundwater_stations"]
    model_path = cfg["model"]["forecast_model_path"]

    df = load_groundwater_data(excel_path)
    
    # Build features for exogenous SARIMAX
    features, sources = build_monthly_features(df, processed_dir=cfg["data"]["processed_folder"])  
    print(f"Loaded features: {list(sources.keys())}")
    print(f"Features shape: {features.shape}")

    models = train_per_district(df, exog_features=features)
    save_models(models, model_path)
    print(f"Saved forecast models to {model_path}")


if __name__ == "__main__":
    main()



