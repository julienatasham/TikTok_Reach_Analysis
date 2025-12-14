import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================
# Resolve project base directory
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==============================
# Build data path safely
# ==============================
DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "Processed",
    "tiktok_cleaned.csv"
)

# ==============================
# Output folder for figures
# ==============================
FIGURES_DIR = os.path.join(BASE_DIR, "TikTok_Reach_Analysis_outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ==============================
# Load dataset
# ==============================
df = pd.read_csv(DATA_PATH)

# ==============================
# Basic Overview
# ==============================
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# ==============================
# Date Feature Engineering
# ==============================
df["Date_Posted"] = pd.to_datetime(df["Date_Posted"])
df["Day_Posted"] = df["Date_Posted"].dt.day_name()

# ==============================
# Helper function to save plots
# ==============================
def save_plot(fig, filename):
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"📸 Saved: {path}")

# ==============================
# EDA Visualizations
# ==============================

# 1. Reach Distribution
fig = plt.figure()
df["Reach"].hist(bins=30)
plt.title("Reach Distribution")
plt.xlabel("Reach")
plt.ylabel("Frequency")
save_plot(fig, "reach_distribution.png")
plt.close(fig)

# 2. Engagement vs Reach
fig = plt.figure()
plt.scatter(df["Reach"], df["Total_Engagement"])
plt.title("Total Engagement vs Reach")
plt.xlabel("Reach")
plt.ylabel("Total Engagement")
save_plot(fig, "engagement_vs_reach.png")
plt.close(fig)

# 3. Average Reach by Post Hour
fig = plt.figure()
df.groupby("Post_Hour")["Reach"].mean().plot(kind="bar")
plt.title("Average Reach by Post Hour")
plt.xlabel("Hour of Posting")
plt.ylabel("Average Reach")
save_plot(fig, "average_reach_by_post_hour.png")
plt.close(fig)

# 4. Hashtags Count vs Reach
fig = plt.figure()
plt.scatter(df["Hashtags_Count"], df["Reach"])
plt.title("Hashtags Count vs Reach")
plt.xlabel("Hashtags Count")
plt.ylabel("Reach")
save_plot(fig, "hashtags_vs_reach.png")
plt.close(fig)

# 5. Video Duration vs Reach
fig = plt.figure()
plt.scatter(df["Duration_Seconds"], df["Reach"])
plt.title("Video Duration vs Reach")
plt.xlabel("Duration (seconds)")
plt.ylabel("Reach")
save_plot(fig, "duration_vs_reach.png")
plt.close(fig)

print("\nEDA completed successfully. All plots saved under 'figures' folder.")
