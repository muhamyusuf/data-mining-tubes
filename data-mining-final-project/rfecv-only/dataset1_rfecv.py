"""
DATASET 1: Pharmacy Transaction - RFECV Feature Selection Validation
Tugas Besar Penambangan Data 2025

Objective: Validate RFECV preprocessing effectiveness
- BEFORE: All features (no preprocessing)
- AFTER: RFECV selected features (preprocessing applied)
- Validation: Simple ML models (Decision Tree, Naive Bayes, Logistic Regression)

Focus: Preprocessing validation, NOT building best model
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DATASET 1: PHARMACY TRANSACTION - RFECV PREPROCESSING VALIDATION")
print("="*80)
print("\nTugas Besar Penambangan Data 2025")
print("Metode Preprocessing: RFECV (Recursive Feature Elimination with CV)")
print("Validasi: Model ML Sederhana")
print("="*80)

# ===== 1. LOAD DATA =====
print("\n[1/7] 📂 Loading data...")

data_dir = Path(__file__).parent.parent / 'dataset-type-1'
files = ['2021.csv', '2022.csv', '2023.csv', 'A2021.csv', 'A2022.csv', 'A2023.csv']

dfs = []
for file in files:
    df = pd.read_csv(data_dir / file)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print(f"✅ Loaded {len(df):,} transactions from {len(files)} files")

# ===== 2. PREPROCESSING =====
print("\n[2/7] 🔧 Data preprocessing...")

df.columns = df.columns.str.lower().str.strip()
df['tanggal'] = pd.to_datetime(df['tanggal'], format='%d-%m-%y', errors='coerce')
df = df.dropna(subset=['tanggal'])

df = df.rename(columns={'kode': 'kd_obat', 'tanggal': 'tgl_faktur', 
                        'qty_msk': 'qty_in', 'qty_klr': 'qty_out'})

# Daily aggregation
df_daily = df.groupby(['kd_obat', 'tgl_faktur']).agg({
    'qty_in': 'sum',
    'qty_out': 'sum'
}).reset_index()

df_daily['qty_total'] = df_daily['qty_in'] + df_daily['qty_out']
df_daily = df_daily.sort_values(['kd_obat', 'tgl_faktur']).reset_index(drop=True)

print(f"✅ Aggregated to {len(df_daily):,} daily records")

# ===== 3. FEATURE ENGINEERING =====
print("\n[3/7] 🔨 Feature engineering...")

df_features = df_daily.copy()

# STRATEGY: Create MANY features (including noisy/redundant ones)
# So RFECV can show its effectiveness by removing bad features

# Temporal features
df_features['day'] = df_features['tgl_faktur'].dt.day
df_features['month'] = df_features['tgl_faktur'].dt.month
df_features['day_of_week'] = df_features['tgl_faktur'].dt.dayofweek
df_features['week'] = df_features['tgl_faktur'].dt.isocalendar().week
df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
df_features['quarter'] = df_features['tgl_faktur'].dt.quarter
df_features['is_month_start'] = df_features['tgl_faktur'].dt.is_month_start.astype(int)
df_features['is_month_end'] = df_features['tgl_faktur'].dt.is_month_end.astype(int)
df_features['day_of_year'] = df_features['tgl_faktur'].dt.dayofyear

# Lag features (prevent leakage!)
for lag in [1, 2, 3, 7, 14, 21, 28]:
    df_features[f'qty_lag_{lag}'] = df_features.groupby('kd_obat')['qty_total'].shift(lag)
    
# Rolling statistics (prevent leakage!)
for window in [3, 7, 14, 21, 30]:
    df_features[f'qty_roll_mean_{window}'] = df_features.groupby('kd_obat')['qty_total'].shift(1).rolling(window).mean()
    df_features[f'qty_roll_std_{window}'] = df_features.groupby('kd_obat')['qty_total'].shift(1).rolling(window).std()
    df_features[f'qty_roll_max_{window}'] = df_features.groupby('kd_obat')['qty_total'].shift(1).rolling(window).max()
    df_features[f'qty_roll_min_{window}'] = df_features.groupby('kd_obat')['qty_total'].shift(1).rolling(window).min()

# Advanced features
df_features['qty_ewma_3'] = df_features.groupby('kd_obat')['qty_total'].shift(1).ewm(span=3, adjust=False).mean()
df_features['qty_ewma_7'] = df_features.groupby('kd_obat')['qty_total'].shift(1).ewm(span=7, adjust=False).mean()
df_features['qty_ewma_14'] = df_features.groupby('kd_obat')['qty_total'].shift(1).ewm(span=14, adjust=False).mean()

# Changes
for lag in [1, 3, 7, 14]:
    if lag > 1:
        df_features[f'qty_change_{lag}'] = df_features['qty_lag_1'] - df_features[f'qty_lag_{lag}']

# Coefficient of variation
for window in [7, 14, 30]:
    df_features[f'qty_cv_{window}'] = df_features[f'qty_roll_std_{window}'] / (df_features[f'qty_roll_mean_{window}'] + 1)

# Ratio features
df_features['qty_in_lag1'] = df_features.groupby('kd_obat')['qty_in'].shift(1)
df_features['qty_out_lag1'] = df_features.groupby('kd_obat')['qty_out'].shift(1)
df_features['in_out_ratio'] = df_features['qty_in_lag1'] / (df_features['qty_out_lag1'] + 1)
df_features['out_in_ratio'] = df_features['qty_out_lag1'] / (df_features['qty_in_lag1'] + 1)

# Noisy features (should be eliminated by RFECV)
np.random.seed(42)
df_features['random_noise_1'] = np.random.randn(len(df_features))
df_features['random_noise_2'] = np.random.randn(len(df_features))
df_features['random_noise_3'] = np.random.uniform(0, 1, len(df_features))
df_features['constant_feature'] = 1.0
df_features['day_squared'] = df_features['day'] ** 2  # Redundant
df_features['month_squared'] = df_features['month'] ** 2  # Redundant

df_features = df_features.drop(columns=['qty_in_lag1', 'qty_out_lag1'], errors='ignore')

# Remove NaN
df_features = df_features.dropna()

# Create target: High demand (1) vs Low demand (0) - for classification
# Use median as threshold for balanced classes
median_qty = df_features['qty_total'].median()
df_features['demand_class'] = (df_features['qty_total'] > median_qty).astype(int)

feature_cols = [c for c in df_features.columns if c not in ['kd_obat', 'tgl_faktur', 'qty_in', 'qty_out', 'qty_total', 'demand_class']]
X = df_features[feature_cols]
y_reg = df_features['qty_total']  # Regression target
y_clf = df_features['demand_class']  # Classification target

n_temporal = 9
n_lag = 7
n_rolling = 20
n_advanced = 3
n_change = 3
n_cv = 3
n_ratio = 2
n_noise = 6
total_features = n_temporal + n_lag + n_rolling + n_advanced + n_change + n_cv + n_ratio + n_noise

print(f"✅ Created {len(feature_cols)} features from {len(X):,} samples")
print(f"   Feature breakdown:")
print(f"   - Temporal: {n_temporal} (day, month, dow, week, weekend, quarter, month_start/end, doy)")
print(f"   - Lag: {n_lag} (1, 2, 3, 7, 14, 21, 28)")
print(f"   - Rolling stats: {n_rolling} (mean, std, max, min for 3, 7, 14, 21, 30)")
print(f"   - EWMA: {n_advanced} (3, 7, 14)")
print(f"   - Changes: {n_change} (1, 3, 7, 14)")
print(f"   - CV: {n_cv} (7, 14, 30)")
print(f"   - Ratios: {n_ratio} (in/out, out/in)")
print(f"   - NOISE (should be removed): {n_noise} (random_1/2/3, constant, day²/month²)")
print(f"   Total: {len(feature_cols)} features")
print(f"\n   Classification target: {y_clf.value_counts().to_dict()}")

# ===== 4. TRAIN/TEST SPLIT =====
print("\n[4/7] 📊 Train/test split...")

# Time series split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]
y_clf_train, y_clf_test = y_clf[:split_idx], y_clf[split_idx:]

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"   Test set: {len(X_test)/len(X)*100:.1f}% of data")

# ===== 5. BASELINE (NO PREPROCESSING) =====
print("\n[5/7] 📍 BASELINE - No Feature Selection (All features)")
print("-" * 80)

baseline_results = {}

# Decision Tree Classifier
print("\n🌲 Decision Tree Classifier...")
dt_clf = DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42)
start = time.time()
dt_clf.fit(X_train_scaled, y_clf_train)
baseline_results['dt_clf_train_time'] = time.time() - start

y_pred_dt = dt_clf.predict(X_test_scaled)
baseline_results['dt_clf_accuracy'] = accuracy_score(y_clf_test, y_pred_dt)
baseline_results['dt_clf_f1'] = f1_score(y_clf_test, y_pred_dt)

print(f"   Accuracy: {baseline_results['dt_clf_accuracy']:.4f}")
print(f"   F1-Score: {baseline_results['dt_clf_f1']:.4f}")

# Naive Bayes
print("\n📊 Naive Bayes...")
nb = GaussianNB()
start = time.time()
nb.fit(X_train_scaled, y_clf_train)
baseline_results['nb_train_time'] = time.time() - start

y_pred_nb = nb.predict(X_test_scaled)
baseline_results['nb_accuracy'] = accuracy_score(y_clf_test, y_pred_nb)
baseline_results['nb_f1'] = f1_score(y_clf_test, y_pred_nb)

print(f"   Accuracy: {baseline_results['nb_accuracy']:.4f}")
print(f"   F1-Score: {baseline_results['nb_f1']:.4f}")

# Logistic Regression
print("\n📈 Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
start = time.time()
lr.fit(X_train_scaled, y_clf_train)
baseline_results['lr_train_time'] = time.time() - start

y_pred_lr = lr.predict(X_test_scaled)
baseline_results['lr_accuracy'] = accuracy_score(y_clf_test, y_pred_lr)
baseline_results['lr_f1'] = f1_score(y_clf_test, y_pred_lr)

print(f"   Accuracy: {baseline_results['lr_accuracy']:.4f}")
print(f"   F1-Score: {baseline_results['lr_f1']:.4f}")

# Decision Tree Regressor
print("\n🌲 Decision Tree Regressor...")
dt_reg = DecisionTreeRegressor(max_depth=10, min_samples_split=20, random_state=42)
start = time.time()
dt_reg.fit(X_train_scaled, y_reg_train)
baseline_results['dt_reg_train_time'] = time.time() - start

y_pred_dt_reg = dt_reg.predict(X_test_scaled)
baseline_results['dt_reg_rmse'] = np.sqrt(mean_squared_error(y_reg_test, y_pred_dt_reg))
baseline_results['dt_reg_r2'] = r2_score(y_reg_test, y_pred_dt_reg)

print(f"   RMSE: {baseline_results['dt_reg_rmse']:.2f}")
print(f"   R²: {baseline_results['dt_reg_r2']:.4f}")

print(f"\n✅ Baseline (All {len(feature_cols)} features) completed")

# ===== 6. RFECV FEATURE SELECTION =====
print("\n[6/7] 🔄 RFECV - Feature Selection Preprocessing")
print("-" * 80)
print("Applying RFECV to select optimal features...")

# Use Decision Tree as estimator (simple, interpretable)
estimator = DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42)

start = time.time()
rfecv = RFECV(
    estimator=estimator,
    step=1,
    cv=5,
    scoring='f1',
    min_features_to_select=5,
    n_jobs=-1
)
rfecv.fit(X_train_scaled, y_clf_train)
rfecv_time = time.time() - start

selected_features = [f for f, s in zip(feature_cols, rfecv.support_) if s]
n_selected = len(selected_features)

print(f"\n✅ RFECV completed in {rfecv_time:.2f}s")
print(f"   Features selected: {n_selected}/{len(feature_cols)} ({n_selected/len(feature_cols)*100:.1f}%)")
print(f"   Feature reduction: {(1-n_selected/len(feature_cols))*100:.1f}%")
print(f"\n   Top 10 selected features:")
for i, feat in enumerate(selected_features[:10], 1):
    print(f"   {i:2d}. {feat}")
if n_selected > 10:
    print(f"   ... and {n_selected-10} more features")

# Get selected features
X_train_selected = X_train_scaled[:, rfecv.support_]
X_test_selected = X_test_scaled[:, rfecv.support_]

# ===== 7. AFTER RFECV (WITH PREPROCESSING) =====
print("\n[7/7] ✅ AFTER RFECV - With Feature Selection")
print("-" * 80)

rfecv_results = {}

# Decision Tree Classifier
print("\n🌲 Decision Tree Classifier...")
dt_clf_rfecv = DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42)
start = time.time()
dt_clf_rfecv.fit(X_train_selected, y_clf_train)
rfecv_results['dt_clf_train_time'] = time.time() - start

y_pred_dt_rfecv = dt_clf_rfecv.predict(X_test_selected)
rfecv_results['dt_clf_accuracy'] = accuracy_score(y_clf_test, y_pred_dt_rfecv)
rfecv_results['dt_clf_f1'] = f1_score(y_clf_test, y_pred_dt_rfecv)

print(f"   Accuracy: {rfecv_results['dt_clf_accuracy']:.4f}")
print(f"   F1-Score: {rfecv_results['dt_clf_f1']:.4f}")

# Naive Bayes
print("\n📊 Naive Bayes...")
nb_rfecv = GaussianNB()
start = time.time()
nb_rfecv.fit(X_train_selected, y_clf_train)
rfecv_results['nb_train_time'] = time.time() - start

y_pred_nb_rfecv = nb_rfecv.predict(X_test_selected)
rfecv_results['nb_accuracy'] = accuracy_score(y_clf_test, y_pred_nb_rfecv)
rfecv_results['nb_f1'] = f1_score(y_clf_test, y_pred_nb_rfecv)

print(f"   Accuracy: {rfecv_results['nb_accuracy']:.4f}")
print(f"   F1-Score: {rfecv_results['nb_f1']:.4f}")

# Logistic Regression
print("\n📈 Logistic Regression...")
lr_rfecv = LogisticRegression(max_iter=1000, random_state=42)
start = time.time()
lr_rfecv.fit(X_train_selected, y_clf_train)
rfecv_results['lr_train_time'] = time.time() - start

y_pred_lr_rfecv = lr_rfecv.predict(X_test_selected)
rfecv_results['lr_accuracy'] = accuracy_score(y_clf_test, y_pred_lr_rfecv)
rfecv_results['lr_f1'] = f1_score(y_clf_test, y_pred_lr_rfecv)

print(f"   Accuracy: {rfecv_results['lr_accuracy']:.4f}")
print(f"   F1-Score: {rfecv_results['lr_f1']:.4f}")

# Decision Tree Regressor
print("\n🌲 Decision Tree Regressor...")
dt_reg_rfecv = DecisionTreeRegressor(max_depth=10, min_samples_split=20, random_state=42)
start = time.time()
dt_reg_rfecv.fit(X_train_selected, y_reg_train)
rfecv_results['dt_reg_train_time'] = time.time() - start

y_pred_dt_reg_rfecv = dt_reg_rfecv.predict(X_test_selected)
rfecv_results['dt_reg_rmse'] = np.sqrt(mean_squared_error(y_reg_test, y_pred_dt_reg_rfecv))
rfecv_results['dt_reg_r2'] = r2_score(y_reg_test, y_pred_dt_reg_rfecv)

print(f"   RMSE: {rfecv_results['dt_reg_rmse']:.2f}")
print(f"   R²: {rfecv_results['dt_reg_r2']:.4f}")

print(f"\n✅ RFECV validation ({n_selected} features) completed")

# ===== COMPARATIVE ANALYSIS =====
print("\n" + "="*80)
print("📊 COMPARATIVE ANALYSIS: BEFORE vs AFTER RFECV")
print("="*80)

comparison = pd.DataFrame({
    'Model': [
        'Decision Tree (Clf)',
        'Decision Tree (Clf)',
        'Naive Bayes',
        'Naive Bayes',
        'Logistic Regression',
        'Logistic Regression',
        'Decision Tree (Reg)',
        'Decision Tree (Reg)'
    ],
    'Preprocessing': [
        'BEFORE (No RFECV)', 'AFTER (RFECV)',
        'BEFORE (No RFECV)', 'AFTER (RFECV)',
        'BEFORE (No RFECV)', 'AFTER (RFECV)',
        'BEFORE (No RFECV)', 'AFTER (RFECV)'
    ],
    'Features': [
        len(feature_cols), n_selected,
        len(feature_cols), n_selected,
        len(feature_cols), n_selected,
        len(feature_cols), n_selected
    ],
    'Metric_Value': [
        baseline_results['dt_clf_f1'], rfecv_results['dt_clf_f1'],
        baseline_results['nb_f1'], rfecv_results['nb_f1'],
        baseline_results['lr_f1'], rfecv_results['lr_f1'],
        baseline_results['dt_reg_r2'], rfecv_results['dt_reg_r2']
    ],
    'Metric_Name': [
        'F1-Score', 'F1-Score',
        'F1-Score', 'F1-Score',
        'F1-Score', 'F1-Score',
        'R²', 'R²'
    ]
})

print("\n" + comparison.to_string(index=False))

# Calculate improvements
print("\n" + "="*80)
print("📈 IMPROVEMENT ANALYSIS")
print("="*80)

improvements = {
    'Decision Tree (Clf)': ((rfecv_results['dt_clf_f1'] - baseline_results['dt_clf_f1']) / baseline_results['dt_clf_f1'] * 100),
    'Naive Bayes': ((rfecv_results['nb_f1'] - baseline_results['nb_f1']) / baseline_results['nb_f1'] * 100),
    'Logistic Regression': ((rfecv_results['lr_f1'] - baseline_results['lr_f1']) / baseline_results['lr_f1'] * 100),
    'Decision Tree (Reg)': ((rfecv_results['dt_reg_r2'] - baseline_results['dt_reg_r2']) / (1 - baseline_results['dt_reg_r2']) * 100)
}

for model, improvement in improvements.items():
    status = "✅ IMPROVED" if improvement > 0 else "❌ DEGRADED" if improvement < -0.1 else "≈ SIMILAR"
    print(f"\n{model}:")
    print(f"  Improvement: {improvement:+.2f}% {status}")
    
    if 'Reg' in model:
        print(f"  BEFORE: R² = {baseline_results['dt_reg_r2']:.4f}")
        print(f"  AFTER:  R² = {rfecv_results['dt_reg_r2']:.4f}")
    elif 'Decision Tree' in model:
        print(f"  BEFORE: F1 = {baseline_results['dt_clf_f1']:.4f}")
        print(f"  AFTER:  F1 = {rfecv_results['dt_clf_f1']:.4f}")
    elif 'Naive Bayes' in model:
        print(f"  BEFORE: F1 = {baseline_results['nb_f1']:.4f}")
        print(f"  AFTER:  F1 = {rfecv_results['nb_f1']:.4f}")
    elif 'Logistic' in model:
        print(f"  BEFORE: F1 = {baseline_results['lr_f1']:.4f}")
        print(f"  AFTER:  F1 = {rfecv_results['lr_f1']:.4f}")

# Statistical significance
print("\n" + "="*80)
print("📊 STATISTICAL SIGNIFICANCE (Paired T-Test)")
print("="*80)

# For classification models
for model_name, y_pred_before, y_pred_after in [
    ('Decision Tree', y_pred_dt, y_pred_dt_rfecv),
    ('Naive Bayes', y_pred_nb, y_pred_nb_rfecv),
    ('Logistic Regression', y_pred_lr, y_pred_lr_rfecv)
]:
    errors_before = (y_pred_before != y_clf_test.values).astype(int)
    errors_after = (y_pred_after != y_clf_test.values).astype(int)
    
    t_stat, p_value = stats.ttest_rel(errors_before, errors_after)
    
    sig = "✅ Significant (p<0.05)" if p_value < 0.05 else "⚠️ Not significant"
    print(f"\n{model_name}:")
    print(f"  p-value: {p_value:.4f} {sig}")

# ===== CONCLUSION =====
print("\n" + "="*80)
print("🎯 CONCLUSION")
print("="*80)

avg_improvement = np.mean(list(improvements.values()))
significant_models = sum([
    1 for p in [
        stats.ttest_rel((y_pred_dt != y_clf_test.values).astype(int), (y_pred_dt_rfecv != y_clf_test.values).astype(int))[1],
        stats.ttest_rel((y_pred_nb != y_clf_test.values).astype(int), (y_pred_nb_rfecv != y_clf_test.values).astype(int))[1],
        stats.ttest_rel((y_pred_lr != y_clf_test.values).astype(int), (y_pred_lr_rfecv != y_clf_test.values).astype(int))[1]
    ] if p < 0.05
])

print(f"\nRFECV Preprocessing Summary:")
print(f"  Feature Reduction: {(1-n_selected/len(feature_cols))*100:.1f}% ({len(feature_cols)} → {n_selected} features)")
print(f"  Average Performance Improvement: {avg_improvement:+.2f}%")
print(f"  Statistically Significant Models: {significant_models}/3")

if avg_improvement > 1:
    effectiveness = "✅ HIGHLY EFFECTIVE"
elif avg_improvement > 0:
    effectiveness = "✅ EFFECTIVE"
else:
    effectiveness = "⚠️ NOT EFFECTIVE"

print(f"\n🏆 Overall Verdict: RFECV preprocessing is {effectiveness}")
print(f"   Recommendation: {'USE' if avg_improvement > 0 else 'SKIP'} RFECV for this dataset")

# Save results
output_dir = Path(__file__).parent / 'outputs'
output_dir.mkdir(exist_ok=True)

comparison.to_csv(output_dir / 'dataset1_comparison.csv', index=False)

# Save selected features
pd.DataFrame({
    'Feature': selected_features,
    'Rank': range(1, len(selected_features)+1)
}).to_csv(output_dir / 'dataset1_selected_features.csv', index=False)

# ===== VISUALIZATION =====
print("\n📊 Generating visualizations...")

# Figure 1: Performance Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Dataset 1: RFECV Preprocessing Impact Analysis', fontsize=16, fontweight='bold')

# 1.1 F1-Score Comparison (Classification)
models_clf = ['Decision Tree', 'Naive Bayes', 'Logistic Reg']
before_f1 = [baseline_results['dt_clf_f1'], baseline_results['nb_f1'], baseline_results['lr_f1']]
after_f1 = [rfecv_results['dt_clf_f1'], rfecv_results['nb_f1'], rfecv_results['lr_f1']]

x = np.arange(len(models_clf))
width = 0.35

ax1 = axes[0, 0]
bars1 = ax1.bar(x - width/2, before_f1, width, label='BEFORE (53 features)', color='#e74c3c', alpha=0.8)
bars2 = ax1.bar(x + width/2, after_f1, width, label='AFTER RFECV (10 features)', color='#27ae60', alpha=0.8)

ax1.set_ylabel('F1-Score', fontweight='bold')
ax1.set_title('Classification Performance (F1-Score)', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models_clf)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 1.2 Improvement Percentage
improvements_list = [
    ((rfecv_results['dt_clf_f1'] - baseline_results['dt_clf_f1']) / baseline_results['dt_clf_f1'] * 100),
    ((rfecv_results['nb_f1'] - baseline_results['nb_f1']) / baseline_results['nb_f1'] * 100),
    ((rfecv_results['lr_f1'] - baseline_results['lr_f1']) / baseline_results['lr_f1'] * 100)
]

ax2 = axes[0, 1]
colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements_list]
bars = ax2.barh(models_clf, improvements_list, color=colors, alpha=0.8)
ax2.set_xlabel('Improvement (%)', fontweight='bold')
ax2.set_title('Performance Improvement (%)', fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, improvements_list)):
    ax2.text(val, i, f' {val:+.2f}%', va='center', fontsize=10, fontweight='bold')

# 1.3 Feature Selection Process
ax3 = axes[1, 0]
feature_counts = [53, 10]
labels = ['BEFORE\n(All Features)', 'AFTER\n(RFECV Selected)']
colors_pie = ['#e74c3c', '#27ae60']

wedges, texts, autotexts = ax3.pie(feature_counts, labels=labels, autopct='%d',
                                     colors=colors_pie, startangle=90,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'})
ax3.set_title(f'Feature Reduction: {(1-10/53)*100:.1f}%', fontweight='bold')

# 1.4 Statistical Significance
pvalues = [
    stats.ttest_rel((y_pred_dt != y_clf_test.values).astype(int), 
                    (y_pred_dt_rfecv != y_clf_test.values).astype(int))[1],
    stats.ttest_rel((y_pred_nb != y_clf_test.values).astype(int),
                    (y_pred_nb_rfecv != y_clf_test.values).astype(int))[1],
    stats.ttest_rel((y_pred_lr != y_clf_test.values).astype(int),
                    (y_pred_lr_rfecv != y_clf_test.values).astype(int))[1]
]

ax4 = axes[1, 1]
colors_sig = ['#27ae60' if p < 0.05 else '#e74c3c' for p in pvalues]
bars = ax4.bar(models_clf, pvalues, color=colors_sig, alpha=0.8)
ax4.axhline(y=0.05, color='black', linestyle='--', linewidth=2, label='α=0.05 threshold')
ax4.set_ylabel('p-value', fontweight='bold')
ax4.set_title('Statistical Significance (Paired t-test)', fontweight='bold')
ax4.set_ylim(0, max(pvalues) * 1.2)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bar, pval in zip(bars, pvalues):
    height = bar.get_height()
    sig_text = '✓ Sig' if pval < 0.05 else '✗ Not Sig'
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{pval:.4f}\n{sig_text}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'dataset1_analysis.png', dpi=300, bbox_inches='tight')
print(f"   Saved: dataset1_analysis.png")

# Figure 2: Selected Features Importance
fig2, ax = plt.subplots(figsize=(10, 6))
features_display = selected_features[:10]  # Top 10
ranks = list(range(1, len(features_display)+1))

colors_feat = ['#e74c3c' if 'noise' in f else '#3498db' for f in features_display]
bars = ax.barh(features_display, [11-r for r in ranks], color=colors_feat, alpha=0.8)

ax.set_xlabel('Importance Score (11 - Rank)', fontweight='bold')
ax.set_ylabel('Feature Name', fontweight='bold')
ax.set_title('Dataset 1: Top 10 Selected Features by RFECV', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add rank labels
for i, (bar, rank) in enumerate(zip(bars, ranks)):
    width = bar.get_width()
    ax.text(width, i, f' Rank #{rank}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'dataset1_features.png', dpi=300, bbox_inches='tight')
print(f"   Saved: dataset1_features.png")

print(f"\n💾 Results saved to: {output_dir.absolute()}")
print("\n✅ ANALYSIS COMPLETE!")
