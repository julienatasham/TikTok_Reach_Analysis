# main.py
import sys
from pathlib import Path
import os

# ==============================
# Project Paths
# ==============================
PROJECT_ROOT = Path(__file__).parent.resolve()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "Data"
RAW_DIR = DATA_DIR / "Raw"
PROCESSED_DIR = DATA_DIR / "Processed"
OUTPUT_DIR = PROJECT_ROOT / "TikTok_Reach_Analysis_outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "saved_models"
STATISTICS_DIR = OUTPUT_DIR / "statistics"

# Create output folders if they don't exist
for folder in [PROCESSED_DIR, FIGURES_DIR, MODELS_DIR, STATISTICS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Add scripts to Python path
sys.path.insert(0, str(SCRIPTS_DIR))

# ==============================
# Imports from scripts
# ==============================
try:
    from load_data import DataLoader
    from clean_data import DataCleaner
    from eda import TikTokEDA
    from models import (
        classification_model,
        regression_model,
        time_series_model
    )
except ModuleNotFoundError as e:
    print(f"Module import error: {e}")
    sys.exit(1)

# ==============================
# Main Pipeline
# ==============================
def main():
    print("Starting TikTok Reach Analysis Pipeline...")
    
    # ------------------------------
    # 1. Load Data
    # ------------------------------
    loader = DataLoader(data_path=RAW_DIR / "TikTok_Reach_Analysis_data.xlsx")
    df = loader.load_data()
    if df is None:
        print("Data loading failed. Exiting...")
        return
    
    # ------------------------------
    # 2. Clean Data
    # ------------------------------
    cleaner = DataCleaner(df)
    cleaned_df = cleaner.clean_data(
        save_to_processed=True,
        filename="tiktok_cleaned.csv"
    )
    
    # ------------------------------
    # 3. Exploratory Data Analysis (EDA)
    # ------------------------------
    eda = TikTokEDA(cleaned_df)
    eda.run_all(save_figures=True, output_dir=FIGURES_DIR)
    
    # ------------------------------
    # 4. Regression Model
    # ------------------------------
    regression = TikTokRegressionModel(cleaned_df)
    regression.prepare_data()
    regression.train_model()
    regression.evaluate_model()
    regression.save_model(MODELS_DIR / "regression_model.pkl")
    
    # ------------------------------
    # 5. Classification Model
    # ------------------------------
    classification = TikTokClassificationModel(cleaned_df)
    classification.prepare_data()
    classification.train_model()
    classification.evaluate_model()
    classification.save_model(MODELS_DIR / "classification_model.pkl")
    
    # ------------------------------
    # 6. Time Series Model
    # ------------------------------
    timeseries = TikTokTimeSeriesModel(cleaned_df)
    timeseries.prepare_data()
    timeseries.train_model()
    timeseries.evaluate_model()
    timeseries.save_model(MODELS_DIR / "timeseries_model.pkl")
    
    print("Pipeline completed successfully.")
    print(f"Processed data saved in: {PROCESSED_DIR}")
    print(f"Figures saved in: {FIGURES_DIR}")
    print(f"Models saved in: {MODELS_DIR}")
    print(f"Statistics report saved in: {STATISTICS_DIR}")

# ==============================
# Entry point
# ==============================
if __name__ == "__main__":
    main()
