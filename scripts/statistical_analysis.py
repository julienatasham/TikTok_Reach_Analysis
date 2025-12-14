"""
TikTok Full Statistical Analysis 
-----------------------------------------------
Comprehensive statistical analysis of TikTok reach and engagement data.
Automatically skips tests on constant or invalid columns to avoid errors/warnings.
Includes:
- Descriptive statistics
- Correlation analysis (Pearson, Spearman, Kendall)
- Normality tests (Shapiro-Wilk, KS)
- Variance homogeneity (Levene)
- Group comparisons (t-test, Mann-Whitney, ANOVA, Kruskal-Wallis)
- Regression statistics
All outputs saved to a text report.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, pearsonr, spearmanr, kendalltau
from scipy.stats import shapiro, kstest, levene, ttest_ind, mannwhitneyu, f_oneway, kruskal
import statsmodels.api as sm
import warnings

# ==============================
# Suppress unnecessary warnings
# ==============================
warnings.filterwarnings("ignore")

# ==============================
# Paths
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Data", "Processed", "tiktok_cleaned.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "TikTok_Reach_Analysis_outputs", "statistics")
os.makedirs(OUTPUT_DIR, exist_ok=True)
REPORT_PATH = os.path.join(OUTPUT_DIR, "full_statistical_report_safe.txt")

# ==============================
# Load Data
# ==============================
df = pd.read_csv(DATA_PATH)
if "Date_Posted" in df.columns:
    df["Date_Posted"] = pd.to_datetime(df["Date_Posted"])

numeric_cols = df.select_dtypes(include='number').columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# ==============================
# Safe utility functions
# ==============================
def safe_skew_kurt(col):
    if np.all(col == col.iloc[0]) or len(col) < 3:
        return np.nan, np.nan
    return skew(col), kurtosis(col)

def safe_correlation(x, y, method="pearson"):
    if np.all(x == x.iloc[0]) or np.all(y == y.iloc[0]):
        return np.nan, np.nan
    if method == "pearson":
        return pearsonr(x, y)
    elif method == "spearman":
        return spearmanr(x, y)
    elif method == "kendall":
        return kendalltau(x, y)
    return np.nan, np.nan

def safe_shapiro(col):
    if len(col) < 3 or np.all(col == col.iloc[0]):
        return np.nan, np.nan
    return shapiro(col)

def safe_levene(*groups):
    non_constant_groups = [g for g in groups if len(np.unique(g)) > 1]
    if len(non_constant_groups) < 2:
        return np.nan, np.nan
    return levene(*non_constant_groups)

def safe_kruskal(*groups):
    non_constant_groups = [g for g in groups if len(np.unique(g)) > 1]
    if len(non_constant_groups) < 2:
        return np.nan, np.nan
    return kruskal(*non_constant_groups)

# ==============================
# Open Report File
# ==============================
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("TIKTOK FULL STATISTICAL ANALYSIS (SAFE VERSION)\n")
    f.write("="*80 + "\n\n")

    # ------------------------------
    # 1. Descriptive Statistics
    # ------------------------------
    f.write("1. Descriptive Statistics\n")
    f.write("-"*60 + "\n")
    for col in numeric_cols:
        mean = df[col].mean()
        median = df[col].median()
        std = df[col].std()
        skew_val, kurt_val = safe_skew_kurt(df[col])
        f.write(f"{col}: mean={mean:.2f}, median={median:.2f}, std={std:.2f}, skew={skew_val}, kurt={kurt_val}\n")
    f.write("\n")

    # ------------------------------
    # 2. Correlation Analysis
    # ------------------------------
    f.write("2. Correlation Analysis\n")
    f.write("-"*60 + "\n")
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            r_pearson, p_pearson = safe_correlation(df[col1], df[col2], "pearson")
            r_spearman, p_spearman = safe_correlation(df[col1], df[col2], "spearman")
            r_kendall, p_kendall = safe_correlation(df[col1], df[col2], "kendall")
            f.write(f"{col1} vs {col2}:\n")
            f.write(f"  Pearson r={r_pearson}, p={p_pearson}\n")
            f.write(f"  Spearman r={r_spearman}, p={p_spearman}\n")
            f.write(f"  Kendall tau={r_kendall}, p={p_kendall}\n")
    f.write("\n")

    # ------------------------------
    # 3. Normality Tests
    # ------------------------------
    f.write("3. Normality Tests\n")
    f.write("-"*60 + "\n")
    for col in numeric_cols:
        shapiro_stat, shapiro_p = safe_shapiro(df[col])
        f.write(f"{col}: Shapiro-Wilk stat={shapiro_stat}, p={shapiro_p}\n")
    f.write("\n")

    # ------------------------------
    # 4. Variance Homogeneity (Levene)
    # ------------------------------
    f.write("4. Variance Homogeneity (Levene)\n")
    f.write("-"*60 + "\n")
    for cat in categorical_cols:
        for num in numeric_cols:
            groups = [df[df[cat]==val][num].dropna() for val in df[cat].unique()]
            stat, p = safe_levene(*groups)
            f.write(f"{num} by {cat}: Levene stat={stat}, p={p}\n")
    f.write("\n")

    # ------------------------------
    # 5. Group Comparisons
    # ------------------------------
    f.write("5. Group Comparisons\n")
    f.write("-"*60 + "\n")
    for cat in categorical_cols:
        for num in numeric_cols:
            groups = [df[df[cat]==val][num].dropna() for val in df[cat].unique()]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) == 2:
                t_stat, t_p = ttest_ind(*groups)
                mw_stat, mw_p = mannwhitneyu(*groups)
                f.write(f"{num} by {cat} (2 groups): t-test stat={t_stat}, p={t_p}; Mann-Whitney stat={mw_stat}, p={mw_p}\n")
            elif len(groups) > 2:
                f_stat, f_p = f_oneway(*groups)
                kw_stat, kw_p = safe_kruskal(*groups)
                f.write(f"{num} by {cat} (>2 groups): ANOVA F={f_stat}, p={f_p}; Kruskal-Wallis H={kw_stat}, p={kw_p}\n")
    f.write("\n")

    # ------------------------------
    # 6. Regression Statistics
    # ------------------------------
    f.write("6. Regression Statistics\n")
    f.write("-"*60 + "\n")
    if all(x in df.columns for x in ["Reach","Likes","Comments","Shares"]):
        X = df[["Likes","Comments","Shares"]]
        y = df["Reach"]
        X = sm.add_constant(X)
        model = sm.OLS(y,X).fit()
        f.write(model.summary().as_text())
        f.write("\n")

print(f"📄 Full safe statistical analysis report saved to: {REPORT_PATH}")
