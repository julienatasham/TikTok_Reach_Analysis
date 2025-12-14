import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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

# ==============================
# Regression Class
# ==============================
class TikTokRegressionModel:
    def __init__(self, df, target="Reach"):
        self.df = df.copy()
        self.target = target
        self.model = None

    def prepare_data(self, features=None):
        if features is None:
            features = [c for c in self.df.select_dtypes(include="number").columns if c != self.target]
        self.X = self.df[features]
        self.y = self.df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train(self):
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        preds = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, preds)
        r2 = r2_score(self.y_test, preds)
        print(f"Regression MSE: {mse:.2f}, R²: {r2:.2f}")

        # Save plot
        plt.figure()
        plt.scatter(self.y_test, preds)
        plt.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()], 'r--')
        plt.xlabel("Actual Reach")
        plt.ylabel("Predicted Reach")
        plt.title("Regression: Actual vs Predicted")
        plt.savefig(os.path.join(SAVED_MODELS_DIR, "regression_actual_vs_predicted.png"), dpi=300, bbox_inches="tight")
        plt.close()

# ==============================
# Run Regression
# ==============================
reg = TikTokRegressionModel(df)
reg.prepare_data()
reg.train()
reg.evaluate()
