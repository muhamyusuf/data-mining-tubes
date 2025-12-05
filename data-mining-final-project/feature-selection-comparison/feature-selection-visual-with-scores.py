"""
================================================================================
VISUALISASI FITUR TERPILIH DENGAN SCORE/NILAI
================================================================================
Membuat visualisasi fitur yang dipilih dengan menampilkan score importance
dari setiap metode feature selection untuk Dataset 1 dan 2

Output: Visualisasi dengan ranking dan score untuk setiap fitur
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest, RFECV
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FEATURE SELECTION WITH IMPORTANCE SCORES")
print("=" * 80)

# Setup output directory
output_dir = Path(__file__).parent / 'outputs-comparison'
output_dir.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', 
          '#34495e', '#e67e22', '#95a5a6', '#d35400']

# ============================================================================
# DATASET 1: PHARMACY TRANSACTION
# ============================================================================
print("\n[1/2] 📊 Processing Dataset 1: Pharmacy Transaction...")

# Load and preprocess
data_path1 = Path(__file__).parent.parent / 'dataset-type-1'
dfs = []
for year in ['2021', '2022', '2023']:
    df = pd.read_csv(data_path1 / f'{year}.csv')
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], format='%d-%m-%y', errors='coerce')
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True).dropna()

# Feature engineering
df_agg = df_all.groupby(['KODE', 'TANGGAL']).agg({
    'QTY_MSK': 'sum',
    'NILAI_MSK': 'sum'
}).reset_index().sort_values(['KODE', 'TANGGAL'])

for lag in [1, 2, 3, 7, 14, 21, 28]:
    df_agg[f'qty_lag_{lag}'] = df_agg.groupby('KODE')['QTY_MSK'].shift(lag)

for window in [3, 7, 14, 21, 30]:
    df_agg[f'qty_roll_mean_{window}'] = df_agg.groupby('KODE')['QTY_MSK'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean())
    df_agg[f'qty_roll_std_{window}'] = df_agg.groupby('KODE')['QTY_MSK'].transform(
        lambda x: x.rolling(window=window, min_periods=1).std())
    df_agg[f'qty_roll_max_{window}'] = df_agg.groupby('KODE')['QTY_MSK'].transform(
        lambda x: x.rolling(window=window, min_periods=1).max())
    df_agg[f'qty_roll_min_{window}'] = df_agg.groupby('KODE')['QTY_MSK'].transform(
        lambda x: x.rolling(window=window, min_periods=1).min())

for span in [3, 7, 14]:
    df_agg[f'qty_ewma_{span}'] = df_agg.groupby('KODE')['QTY_MSK'].transform(
        lambda x: x.ewm(span=span, adjust=False).mean())

for period in [1, 3, 7]:
    df_agg[f'qty_change_{period}'] = df_agg.groupby('KODE')['QTY_MSK'].pct_change(periods=period)

for window in [7, 14, 30]:
    mean = df_agg.groupby('KODE')['QTY_MSK'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean())
    std = df_agg.groupby('KODE')['QTY_MSK'].transform(
        lambda x: x.rolling(window=window, min_periods=1).std())
    df_agg[f'qty_cv_{window}'] = std / (mean + 1e-8)

df_agg['nilai_per_unit'] = df_agg['NILAI_MSK'] / (df_agg['QTY_MSK'] + 1)
df_agg = df_agg.dropna()

median_demand = df_agg['QTY_MSK'].median()
df_agg['demand_class'] = (df_agg['QTY_MSK'] > median_demand).astype(int)

feature_cols_d1 = [col for col in df_agg.columns if col not in 
                   ['KODE', 'TANGGAL', 'QTY_MSK', 'NILAI_MSK', 'demand_class']]

X_d1 = df_agg[feature_cols_d1].copy()
y_d1 = df_agg['demand_class'].copy()

X_train_d1, X_test_d1, y_train_d1, y_test_d1 = train_test_split(
    X_d1, y_d1, test_size=0.2, random_state=42, stratify=y_d1)

scaler_d1 = StandardScaler()
X_train_scaled_d1 = scaler_d1.fit_transform(X_train_d1)
X_test_scaled_d1 = scaler_d1.transform(X_test_d1)

print(f"✅ Dataset 1: {len(feature_cols_d1)} features, {len(X_d1)} samples")

# Calculate scores for Dataset 1
K_FEATURES_D1 = 10

# Information Gain
mi_scores_d1 = mutual_info_classif(X_train_scaled_d1, y_train_d1, random_state=42)
mi_ranking_d1 = sorted(zip(feature_cols_d1, mi_scores_d1), key=lambda x: x[1], reverse=True)
top_mi_d1 = mi_ranking_d1[:K_FEATURES_D1]

# Chi-Square
scaler_chi_d1 = MinMaxScaler()
X_train_nonneg_d1 = scaler_chi_d1.fit_transform(X_train_d1)
chi2_scores_d1, _ = chi2(X_train_nonneg_d1, y_train_d1)
chi2_ranking_d1 = sorted(zip(feature_cols_d1, chi2_scores_d1), key=lambda x: x[1], reverse=True)
top_chi_d1 = chi2_ranking_d1[:K_FEATURES_D1]

# Pearson Correlation
corr_scores_d1 = []
for i, col in enumerate(feature_cols_d1):
    corr = abs(np.corrcoef(X_train_scaled_d1[:, i], y_train_d1)[0, 1])
    corr_scores_d1.append((col, corr))
corr_ranking_d1 = sorted(corr_scores_d1, key=lambda x: x[1], reverse=True)
top_corr_d1 = corr_ranking_d1[:K_FEATURES_D1]

# RFECV
rfecv_d1 = RFECV(
    estimator=DecisionTreeClassifier(max_depth=8, min_samples_split=50, random_state=42),
    step=1, cv=5, scoring='f1_weighted', min_features_to_select=5, n_jobs=-1
)
rfecv_d1.fit(X_train_scaled_d1, y_train_d1)
rfecv_ranking_d1 = [(feature_cols_d1[i], rfecv_d1.ranking_[i]) 
                     for i in range(len(feature_cols_d1)) if rfecv_d1.support_[i]]
# Sort by ranking (lower is better for RFECV)
rfecv_ranking_d1 = sorted(rfecv_ranking_d1, key=lambda x: x[1])
# Convert to score (inverse of ranking for visualization)
rfecv_scores_d1 = [(feat, 1.0/rank) for feat, rank in rfecv_ranking_d1]

print(f"✅ Calculated feature scores for all methods")

# ============================================================================
# DATASET 2: WAVE MEASUREMENT
# ============================================================================
print("\n[2/2] 📊 Processing Dataset 2: Wave Measurement...")

data_path2 = Path(__file__).parent.parent / 'dataset-type-2'
files = list(data_path2.glob('*.xlsx'))
dfs2 = [pd.read_excel(f) for f in files]
df_wave = pd.concat(dfs2, ignore_index=True)

print(f"   Loaded {len(df_wave)} samples initially")

# Identify target column (usually 'Hsig' or last column with numeric data)
target_col = None
for col in df_wave.columns:
    if 'hsig' in str(col).lower() or 'Hsig' in str(col):
        target_col = col
        break
if target_col is None:
    # Use last numeric column
    numeric_cols = df_wave.select_dtypes(include=[np.number]).columns
    target_col = numeric_cols[-1]

print(f"   Target column: {target_col}")

# Remove columns with all missing values
df_clean = df_wave.dropna(axis=1, how='all')
print(f"   After removing empty columns: {df_clean.shape}")

# Remove rows with missing target
df_clean = df_clean.dropna(subset=[target_col])
print(f"   After removing missing target: {df_clean.shape}")

# Keep only numeric columns but ensure target is kept
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
if target_col not in numeric_cols:
    # Target might not be numeric initially, try to convert
    try:
        df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[target_col])
    except:
        pass
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
df_clean = df_clean[numeric_cols]
print(f"   After keeping only numeric columns: {df_clean.shape}")

# Fill missing features with median
for col in df_clean.columns:
    if col != target_col and df_clean[col].isna().any():
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Remove constant columns
cols_to_drop = []
for col in df_clean.columns:
    if col != target_col:
        if df_clean[col].nunique() == 1:
            cols_to_drop.append(col)
if cols_to_drop:
    df_clean = df_clean.drop(columns=cols_to_drop)
print(f"   After removing constant columns: {df_clean.shape}")

# Remove outliers (keep 99% of data)
for col in df_clean.select_dtypes(include=[np.number]).columns:
    if col != target_col:
        top_threshold = df_clean[col].quantile(0.99)
        df_clean = df_clean[df_clean[col] <= top_threshold]
print(f"   After removing outliers: {df_clean.shape}")

df_clean.columns = df_clean.columns.astype(str)
feature_cols_d2 = [col for col in df_clean.columns if col != str(target_col)]
X_d2 = df_clean[feature_cols_d2].copy()
y_reg_d2 = df_clean[str(target_col)].copy()
median_hsig = y_reg_d2.median()
y_d2 = (y_reg_d2 > median_hsig).astype(int)

X_train_d2, X_test_d2, y_train_d2, y_test_d2 = train_test_split(
    X_d2, y_d2, test_size=0.2, random_state=42, stratify=y_d2)

scaler_d2 = StandardScaler()
X_train_scaled_d2 = scaler_d2.fit_transform(X_train_d2)
X_test_scaled_d2 = scaler_d2.transform(X_test_d2)
X_train_scaled_d2 = np.nan_to_num(X_train_scaled_d2, nan=0.0)

print(f"✅ Dataset 2: {len(feature_cols_d2)} features, {len(X_d2)} samples")

# Calculate scores for Dataset 2
K_FEATURES_D2 = 7

# Information Gain
mi_scores_d2 = mutual_info_classif(X_train_scaled_d2, y_train_d2, random_state=42)
mi_ranking_d2 = sorted(zip(feature_cols_d2, mi_scores_d2), key=lambda x: x[1], reverse=True)
top_mi_d2 = mi_ranking_d2[:K_FEATURES_D2]

# Chi-Square
scaler_chi_d2 = MinMaxScaler()
X_train_nonneg_d2 = scaler_chi_d2.fit_transform(X_train_d2)
chi2_scores_d2, _ = chi2(X_train_nonneg_d2, y_train_d2)
chi2_ranking_d2 = sorted(zip(feature_cols_d2, chi2_scores_d2), key=lambda x: x[1], reverse=True)
top_chi_d2 = chi2_ranking_d2[:K_FEATURES_D2]

# Pearson Correlation
corr_scores_d2 = []
for i, col in enumerate(feature_cols_d2):
    corr = abs(np.corrcoef(X_train_scaled_d2[:, i], y_train_d2)[0, 1])
    corr_scores_d2.append((col, corr))
corr_ranking_d2 = sorted(corr_scores_d2, key=lambda x: x[1], reverse=True)
top_corr_d2 = corr_ranking_d2[:K_FEATURES_D2]

# RFECV
rfecv_d2 = RFECV(
    estimator=DecisionTreeClassifier(max_depth=10, min_samples_split=50, random_state=42),
    step=3, cv=5, scoring='f1_weighted', min_features_to_select=5, n_jobs=-1
)
rfecv_d2.fit(X_train_scaled_d2, y_train_d2)
rfecv_ranking_d2 = [(feature_cols_d2[i], rfecv_d2.ranking_[i]) 
                     for i in range(len(feature_cols_d2)) if rfecv_d2.support_[i]]
rfecv_ranking_d2 = sorted(rfecv_ranking_d2, key=lambda x: x[1])
rfecv_scores_d2 = [(feat, 1.0/rank) for feat, rank in rfecv_ranking_d2]

print(f"✅ Calculated feature scores for all methods")

# ============================================================================
# VISUALIZATION: DATASET 1 WITH SCORES
# ============================================================================
print("\n📊 Creating visualizations with scores...")

dataset1_data = {
    'Information Gain': top_mi_d1,
    'Chi-Square': top_chi_d1,
    'Pearson Correlation': top_corr_d1,
    'RFECV': rfecv_scores_d1
}

fig1, axes1 = plt.subplots(2, 2, figsize=(18, 14))
fig1.suptitle('Dataset 1: Pharmacy Transaction - Feature Importance Scores', 
              fontsize=18, fontweight='bold', y=0.995)

methods = ['Information Gain', 'Chi-Square', 'Pearson Correlation', 'RFECV']
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for method, pos in zip(methods, positions):
    ax = axes1[pos]
    data = dataset1_data[method]
    
    # Reverse for display (highest at top)
    features = [f[0] for f in data][::-1]
    scores = [f[1] for f in data][::-1]
    
    n_features = len(features)
    y_pos = np.arange(n_features)
    
    # Normalize scores for visualization
    if max(scores) > 0:
        scores_norm = [s / max(scores) for s in scores]
    else:
        scores_norm = scores
    
    # Create bars with varying colors
    bars = ax.barh(y_pos, scores_norm, alpha=0.7, edgecolor='black')
    
    # Color gradient
    for i, bar in enumerate(bars):
        bar.set_color(colors[i % len(colors)])
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel('Normalized Importance Score', fontsize=10, fontweight='bold')
    ax.set_title(f'{method}\n({n_features} features selected)', 
                fontsize=12, fontweight='bold', pad=10)
    
    # Add score values
    for i, (bar, score, score_norm) in enumerate(zip(bars, scores[::-1], scores_norm)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
               f'{score:.4f}',
               ha='left', va='center', fontsize=8, fontweight='bold',
               color='darkgreen')
    
    ax.set_xlim(0, max(scores_norm) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
output_path1 = output_dir / 'dataset1_features_with_scores.png'
plt.savefig(output_path1, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path1.name}")
plt.close()

# ============================================================================
# VISUALIZATION: DATASET 2 WITH SCORES
# ============================================================================
dataset2_data = {
    'Information Gain': top_mi_d2,
    'Chi-Square': top_chi_d2,
    'Pearson Correlation': top_corr_d2,
    'RFECV': rfecv_scores_d2
}

fig2, axes2 = plt.subplots(2, 2, figsize=(18, 14))
fig2.suptitle('Dataset 2: Wave Measurement - Feature Importance Scores', 
              fontsize=18, fontweight='bold', y=0.995)

for method, pos in zip(methods, positions):
    ax = axes2[pos]
    data = dataset2_data[method]
    
    features = [f[0] for f in data][::-1]
    scores = [f[1] for f in data][::-1]
    
    n_features = len(features)
    y_pos = np.arange(n_features)
    
    if max(scores) > 0:
        scores_norm = [s / max(scores) for s in scores]
    else:
        scores_norm = scores
    
    bars = ax.barh(y_pos, scores_norm, alpha=0.7, edgecolor='black')
    
    for i, bar in enumerate(bars):
        bar.set_color(colors[i % len(colors)])
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel('Normalized Importance Score', fontsize=10, fontweight='bold')
    ax.set_title(f'{method}\n({n_features} features selected)', 
                fontsize=12, fontweight='bold', pad=10)
    
    for i, (bar, score, score_norm) in enumerate(zip(bars, scores[::-1], scores_norm)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
               f'{score:.4f}',
               ha='left', va='center', fontsize=9, fontweight='bold',
               color='darkgreen')
    
    ax.set_xlim(0, max(scores_norm) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
output_path2 = output_dir / 'dataset2_features_with_scores.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path2.name}")
plt.close()

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("📊 FEATURE IMPORTANCE SUMMARY")
print("=" * 80)

print("\n🔹 DATASET 1: TOP 5 FEATURES BY SCORE")
print("-" * 80)
for method in methods:
    data = dataset1_data[method]
    print(f"\n{method}:")
    for i, (feat, score) in enumerate(data[:5], 1):
        print(f"  {i}. {feat:30s} | Score: {score:.6f}")

print("\n\n🔹 DATASET 2: ALL FEATURES BY SCORE")
print("-" * 80)
for method in methods:
    data = dataset2_data[method]
    print(f"\n{method}:")
    for i, (feat, score) in enumerate(data, 1):
        print(f"  {i}. {feat:30s} | Score: {score:.6f}")

print("\n" + "=" * 80)
print("✅ VISUALIZATION WITH SCORES COMPLETE!")
print("=" * 80)
print(f"\n📁 Output files:")
print(f"   1. {output_path1.name}")
print(f"   2. {output_path2.name}")
print("\n" + "=" * 80)
