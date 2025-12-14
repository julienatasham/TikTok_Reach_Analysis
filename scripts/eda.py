"""
TikTok Reach Analysis – Exploratory Data Analysis (EDA)

Description:
A full, class-based EDA pipeline for TikTok reach and engagement data.
All visualizations are automatically saved to the project output directory.
"""

# ==============================
# Imports
# ==============================
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
pd.set_option("display.max_columns", None)

# ==============================
# Resolve Project Paths
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "Data", "Processed", "tiktok_cleaned.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "TikTok_Reach_Analysis_outputs"
)

FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

REPORT_PATH = os.path.join(
    BASE_DIR, "Data", "Processed", "eda_report.txt"
)

os.makedirs(FIGURES_DIR, exist_ok=True)

# ==============================
# Load Dataset
# ==============================
df = pd.read_csv(DATA_PATH)

if "Date_Posted" in df.columns:
    df["Date_Posted"] = pd.to_datetime(df["Date_Posted"])

# ==============================
# EDA CLASS
# ==============================
class TikTokEDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()

        print("\n📊 TikTok EDA Initialized")
        print(f"Dataset Shape: {self.df.shape}")

    # ==============================
    # Utility: Save Figures
    # ==============================
    def save_figure(self, filename: str):
        path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"📸 Saved: {path}")

    # ==============================
    # Data Overview
    # ==============================
    def data_overview(self):
        print("\n" + "=" * 80)
        print("DATA OVERVIEW")
        print("=" * 80)
        print(self.df.info())
        print("\nStatistical Summary:")
        print(self.df.describe().round(2))

    # ==============================
    # Numerical Analysis
    # ==============================
    def numerical_analysis(self):
        metrics = [
            "Reach", "Likes", "Comments", "Shares",
            "Total_Engagement", "Engagement_Rate",
            "Duration_Seconds", "Hashtags_Count"
        ]
        metrics = [m for m in metrics if m in self.df.columns]

        for col in metrics:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            sns.histplot(self.df[col], bins=30, ax=axes[0])
            axes[0].set_title(f"Distribution of {col}")

            sns.boxplot(x=self.df[col], ax=axes[1])
            axes[1].set_title(f"Outliers in {col}")

            plt.tight_layout()
            self.save_figure(f"{col.lower()}_distribution_outliers.png")
            plt.show()

    # ==============================
    # Categorical Analysis
    # ==============================
    def categorical_analysis(self):
        categories = [
            "Time_Category", "Content_Type", "Engagement_Category"
        ]
        categories = [c for c in categories if c in self.df.columns]

        for col in categories:
            counts = self.df[col].value_counts()

            plt.figure(figsize=(8, 4))
            sns.barplot(x=counts.index, y=counts.values)
            plt.title(f"{col} Distribution")
            plt.xticks(rotation=45)
            plt.ylabel("Count")

            self.save_figure(f"{col.lower()}_distribution.png")
            plt.show()

    # ==============================
    # Correlation Analysis
    # ==============================
    def correlation_analysis(self):
        key_metrics = [
            "Reach", "Likes", "Comments", "Shares",
            "Total_Engagement", "Engagement_Rate",
            "Hashtags_Count", "Duration_Seconds"
        ]
        key_metrics = [k for k in key_metrics if k in self.df.columns]

        corr = self.df[key_metrics].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")

        self.save_figure("correlation_heatmap.png")
        plt.show()

    # ==============================
    # Time Series Analysis
    # ==============================
    def time_series_analysis(self):
        if "Date_Posted" not in self.df.columns:
            return

        ts = (
            self.df
            .set_index("Date_Posted")
            .resample("D")[["Reach", "Total_Engagement"]]
            .mean()
        )

        ts.plot(figsize=(12, 5))
        plt.title("Daily Average Reach & Engagement")
        plt.ylabel("Value")

        self.save_figure("time_series_reach_engagement.png")
        plt.show()

    # ==============================
    # Report Generation
    # ==============================
    def generate_report(self):
        lines = [
            "TIKTOK REACH EDA REPORT",
            "=" * 60,
            f"Rows: {len(self.df)}",
            f"Columns: {len(self.df.columns)}",
        ]

        if "Reach" in self.df.columns:
            lines.append(f"Average Reach: {self.df['Reach'].mean():.2f}")

        if "Engagement_Rate" in self.df.columns and "Time_Category" in self.df.columns:
            best_time = (
                self.df.groupby("Time_Category")["Engagement_Rate"]
                .mean()
                .idxmax()
            )
            lines.append(f"Best Time Category: {best_time}")

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"\n📄 EDA report saved to: {REPORT_PATH}")

    # ==============================
    # Run Full EDA
    # ==============================
    def run_all(self):
        self.data_overview()
        self.numerical_analysis()
        self.categorical_analysis()
        self.correlation_analysis()
        self.time_series_analysis()
        self.generate_report()


# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    eda = TikTokEDA(df)
    eda.run_all()
