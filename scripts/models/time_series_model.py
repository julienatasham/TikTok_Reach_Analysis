import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

# ==============================
# Paths
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "Data", "Processed", "tiktok_cleaned.csv")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "TikTok_Reach_Analysis_outputs", "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# Load dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df["Date_Posted"] = pd.to_datetime(df["Date_Posted"])

# ==============================
# Time Series Class
# ==============================
class TikTokTimeSeriesModel:
    def __init__(self, df, target="Reach"):
        self.df = df.copy()
        self.target = target
        self.model = None

    def prepare_data(self):
        if "Date_Posted" not in self.df.columns:
            raise ValueError("Date_Posted column required for time series")
        self.ts = self.df.set_index("Date_Posted")[self.target].asfreq('D').fillna(method="ffill")

    def train(self):
        self.model = ExponentialSmoothing(self.ts, trend="add", seasonal=None).fit()

    def forecast_and_plot(self, steps=14):
        preds = self.model.forecast(steps)
        print(f"Forecast for next {steps} days:\n{preds}")

        plt.figure()
        self.ts.plot(label="Actual")
        preds.plot(label="Forecast", color="red")
        plt.title("Time Series Forecast")
        plt.xlabel("Date")
        plt.ylabel(self.target)
        plt.legend()
        plt.savefig(os.path.join(SAVED_MODELS_DIR, "timeseries_forecast.png"), dpi=300, bbox_inches="tight")
        plt.close()

# ==============================
# Run Time Series
# ==============================
ts_model = TikTokTimeSeriesModel(df)
ts_model.prepare_data()
ts_model.train()
ts_model.forecast_and_plot()
