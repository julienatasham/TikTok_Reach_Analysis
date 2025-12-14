import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os

# ==============================
# Paths
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "Data", "Processed", "tiktok_cleaned.csv")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "TikTok_Reach_Analysis_outputs", "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# ==============================
# Classification Class
# ==============================
class TikTokClassificationModel:
    def __init__(self, df, target="Engagement_Category"):
        self.df = df.copy()
        self.target = target
        self.model = None

    def prepare_data(self, features=None):
        if features is None:
            features = [c for c in self.df.select_dtypes(include="number").columns]
        self.X = self.df[features]
        self.y = self.df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        preds = self.model.predict(self.X_test)
        print(classification_report(self.y_test, preds))

        cm = confusion_matrix(self.y_test, preds)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Classification Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(SAVED_MODELS_DIR, "classification_confusion_matrix.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Save model
        joblib.dump(self.model, os.path.join(SAVED_MODELS_DIR, "classification_model.pkl"))
        print("✅ Classification model saved as .pkl")

# ==============================
# Run Classification
# ==============================
clf = TikTokClassificationModel(df)
clf.prepare_data()
clf.train()
clf.evaluate()
