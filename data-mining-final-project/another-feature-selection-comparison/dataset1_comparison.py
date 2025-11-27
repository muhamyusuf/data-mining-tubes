"""
DATASET 1: Pharmacy Transaction - Feature Selection Comparison
Objective: Compare RFECV vs Mutual Information feature selection

Dataset Characteristics:
- Time series pharmacy transactions (2021-2023)
- Target: qty_total (daily transaction volume)
- Features: Temporal + lag + rolling statistics
- Challenge: Highly skewed distribution (median=3, max=12,073)

Feature Selection Methods:
1. RFECV: Recursive Feature Elimination with Cross-Validation
2. Mutual Information: Information-theoretic dependency measure

Model: Ridge Regression (simple, anti-overfitting via L2 regularization)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFECV, mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("📊 DATASET 1: PHARMACY TRANSACTION - FEATURE SELECTION COMPARISON")
print("="*80)

# ===== 1. DATA LOADING =====
print("\n[1/8] 📂 Loading pharmacy transaction data...")

data_dir = Path(__file__).parent.parent / 'dataset-type-1'
files = ['2021.csv', '2022.csv', '2023.csv', 'A2021.csv', 'A2022.csv', 'A2023.csv']

dfs = []
for file in files:
    df = pd.read_csv(data_dir / file)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print(f"✅ Loaded {len(df):,} rows from {len(files)} files")

# ===== 2. PREPROCESSING =====
print("\n[2/8] 🔧 Data preprocessing...")

# Clean column names
df.columns = df.columns.str.lower().str.strip()

# Parse date
df['tanggal'] = pd.to_datetime(df['tanggal'], format='%d-%m-%y', errors='coerce')
df = df.dropna(subset=['tanggal'])

# Rename for consistency
df = df.rename(columns={'kode': 'kd_obat', 'tanggal': 'tgl_faktur', 
                        'qty_msk': 'qty_in', 'qty_klr': 'qty_out'})

# Daily aggregation by product
df_daily = df.groupby(['kd_obat', 'tgl_faktur']).agg({
    'qty_in': 'sum',
    'qty_out': 'sum'
}).reset_index()

df_daily['qty_total'] = df_daily['qty_in'] + df_daily['qty_out']
df_daily = df_daily.sort_values(['kd_obat', 'tgl_faktur']).reset_index(drop=True)

print(f"✅ Daily aggregated: {len(df_daily):,} rows")

# ===== 3. FEATURE ENGINEERING =====
print("\n[3/8] 🔨 Feature engineering...")

df_features = df_daily.copy()

# Temporal features
df_features['day'] = df_features['tgl_faktur'].dt.day
df_features['month'] = df_features['tgl_faktur'].dt.month
df_features['day_of_week'] = df_features['tgl_faktur'].dt.dayofweek
df_features['week'] = df_features['tgl_faktur'].dt.isocalendar().week
df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)

# Lag features (prevent leakage)
for lag in [1, 2, 3, 7]:
    df_features[f'qty_lag_{lag}'] = df_features.groupby('kd_obat')['qty_total'].shift(lag)

# Rolling statistics (prevent leakage)
for window in [7, 14]:
    df_features[f'qty_roll_mean_{window}'] = df_features.groupby('kd_obat')['qty_total'].shift(1).rolling(window).mean()
    df_features[f'qty_roll_std_{window}'] = df_features.groupby('kd_obat')['qty_total'].shift(1).rolling(window).std()

# Advanced features
df_features['qty_ewma_7'] = df_features.groupby('kd_obat')['qty_total'].shift(1).ewm(span=7, adjust=False).mean()
df_features['qty_change_1'] = df_features['qty_lag_1'] - df_features['qty_lag_2']
df_features['qty_cv_7'] = df_features['qty_roll_std_7'] / (df_features['qty_roll_mean_7'] + 1)

# Ratio features
df_features['qty_in_lag1'] = df_features.groupby('kd_obat')['qty_in'].shift(1)
df_features['qty_out_lag1'] = df_features.groupby('kd_obat')['qty_out'].shift(1)
df_features['in_out_ratio'] = df_features['qty_in_lag1'] / (df_features['qty_out_lag1'] + 1)
df_features = df_features.drop(columns=['qty_in_lag1', 'qty_out_lag1'], errors='ignore')

# Remove NaN
df_features = df_features.dropna()

feature_cols = [c for c in df_features.columns if c not in ['kd_obat', 'tgl_faktur', 'qty_in', 'qty_out', 'qty_total']]
X = df_features[feature_cols]
y = df_features['qty_total']

print(f"✅ Created {len(feature_cols)} features from {len(X):,} samples")
print(f"   Target distribution: median={y.median():.0f}, mean={y.mean():.0f}, max={y.max():.0f}")

# ===== 4. TRAIN/TEST SPLIT =====
print("\n[4/8] 📊 Train/test split (80/20)...")

# Time series split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,}")

# ===== 5. BASELINE (NO FEATURE SELECTION) =====
print("\n[5/8] 📍 BASELINE - All features (no selection)...")

model_baseline = Ridge(alpha=10.0, random_state=42)

start_time = time.time()
model_baseline.fit(X_train_scaled, y_train)
baseline_train_time = time.time() - start_time

y_pred_baseline = model_baseline.predict(X_test_scaled)

baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
baseline_r2 = r2_score(y_test, y_pred_baseline)

print(f"✅ Baseline: {len(feature_cols)} features")
print(f"   RMSE: {baseline_rmse:.2f} | R²: {baseline_r2:.4f} | Train time: {baseline_train_time:.4f}s")

# ===== 6. RFECV FEATURE SELECTION =====
print("\n[6/8] 🔄 RFECV - Recursive Feature Elimination...")

start_time = time.time()

rfecv = RFECV(
    estimator=Ridge(alpha=10.0, random_state=42),
    step=1,
    cv=5,
    min_features_to_select=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
rfecv.fit(X_train_scaled, y_train)

rfecv_selection_time = time.time() - start_time

selected_rfecv = [f for f, s in zip(feature_cols, rfecv.support_) if s]
print(f"✅ Selected {len(selected_rfecv)}/{len(feature_cols)} features ({(1-len(selected_rfecv)/len(feature_cols))*100:.1f}% reduction)")
print(f"   Features: {selected_rfecv[:5]}..." if len(selected_rfecv) > 5 else f"   Features: {selected_rfecv}")

# Train with selected features
model_rfecv = Ridge(alpha=10.0, random_state=42)

start_time = time.time()
model_rfecv.fit(X_train_scaled[:, rfecv.support_], y_train)
rfecv_train_time = time.time() - start_time

y_pred_rfecv = model_rfecv.predict(X_test_scaled[:, rfecv.support_])

rfecv_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rfecv))
rfecv_r2 = r2_score(y_test, y_pred_rfecv)

print(f"   RMSE: {rfecv_rmse:.2f} | R²: {rfecv_r2:.4f}")
print(f"   Selection: {rfecv_selection_time:.4f}s | Training: {rfecv_train_time:.4f}s")

# ===== 7. MUTUAL INFORMATION FEATURE SELECTION =====
print("\n[7/8] 📈 Mutual Information - Information-theoretic selection...")

start_time = time.time()

mi_scores = mutual_info_regression(X_train_scaled, y_train, random_state=42)
mi_df = pd.DataFrame({
    'feature': feature_cols,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

# Select same number as RFECV for fair comparison
selected_mi = mi_df.head(len(selected_rfecv))['feature'].tolist()
mi_selection_time = time.time() - start_time

print(f"✅ Selected {len(selected_mi)}/{len(feature_cols)} features ({(1-len(selected_mi)/len(feature_cols))*100:.1f}% reduction)")
print(f"   Top features: {selected_mi[:5]}..." if len(selected_mi) > 5 else f"   Features: {selected_mi}")

# Get column indices
mi_indices = [i for i, f in enumerate(feature_cols) if f in selected_mi]

# Train with selected features
model_mi = Ridge(alpha=10.0, random_state=42)

start_time = time.time()
model_mi.fit(X_train_scaled[:, mi_indices], y_train)
mi_train_time = time.time() - start_time

y_pred_mi = model_mi.predict(X_test_scaled[:, mi_indices])

mi_rmse = np.sqrt(mean_squared_error(y_test, y_pred_mi))
mi_r2 = r2_score(y_test, y_pred_mi)

print(f"   RMSE: {mi_rmse:.2f} | R²: {mi_r2:.4f}")
print(f"   Selection: {mi_selection_time:.4f}s | Training: {mi_train_time:.4f}s")

# ===== 8. STATISTICAL COMPARISON =====
print("\n[8/8] 📊 Statistical significance testing...")

errors_baseline = np.abs(y_test.values - y_pred_baseline)
errors_rfecv = np.abs(y_test.values - y_pred_rfecv)
errors_mi = np.abs(y_test.values - y_pred_mi)

t_rfecv, p_rfecv = stats.ttest_rel(errors_baseline, errors_rfecv)
t_mi, p_mi = stats.ttest_rel(errors_baseline, errors_mi)

print(f"✅ Paired t-test results:")
print(f"   RFECV vs Baseline:    p={p_rfecv:.4f} {'✅ Significant' if p_rfecv < 0.05 else '⚠️ Not significant'}")
print(f"   Mutual Info vs Baseline: p={p_mi:.4f} {'✅ Significant' if p_mi < 0.05 else '⚠️ Not significant'}")

# ===== RESULTS SUMMARY =====
print("\n" + "="*80)
print("📊 COMPARATIVE RESULTS")
print("="*80)

results = pd.DataFrame({
    'Method': ['Baseline (All Features)', 'RFECV', 'Mutual Information'],
    'Features': [len(feature_cols), len(selected_rfecv), len(selected_mi)],
    'RMSE': [baseline_rmse, rfecv_rmse, mi_rmse],
    'R2': [baseline_r2, rfecv_r2, mi_r2],
    'Selection_Time_s': [0.0, rfecv_selection_time, mi_selection_time],
    'Training_Time_s': [baseline_train_time, rfecv_train_time, mi_train_time],
    'p_value': ['-', f'{p_rfecv:.4f}', f'{p_mi:.4f}']
})

print("\n" + results.to_string(index=False))

# Calculate improvements
print("\n" + "="*80)
print("📈 IMPROVEMENT ANALYSIS")
print("="*80)

rfecv_rmse_improvement = ((baseline_rmse - rfecv_rmse) / baseline_rmse) * 100
mi_rmse_improvement = ((baseline_rmse - mi_rmse) / baseline_rmse) * 100

rfecv_r2_improvement = ((rfecv_r2 - baseline_r2) / (1 - baseline_r2)) * 100
mi_r2_improvement = ((mi_r2 - baseline_r2) / (1 - baseline_r2)) * 100

print(f"\nRFECV:")
print(f"  Feature reduction: {(1-len(selected_rfecv)/len(feature_cols))*100:.1f}%")
print(f"  RMSE improvement: {rfecv_rmse_improvement:+.2f}%")
print(f"  R² improvement: {rfecv_r2_improvement:+.2f}%")
print(f"  Training speedup: {baseline_train_time/rfecv_train_time:.2f}x")

print(f"\nMutual Information:")
print(f"  Feature reduction: {(1-len(selected_mi)/len(feature_cols))*100:.1f}%")
print(f"  RMSE improvement: {mi_rmse_improvement:+.2f}%")
print(f"  R² improvement: {mi_r2_improvement:+.2f}%")
print(f"  Training speedup: {baseline_train_time/mi_train_time:.2f}x")

# Winner determination
best_method = results.loc[results['RMSE'].idxmin(), 'Method']
print(f"\n🏆 Best method by RMSE: {best_method}")

# Save results
output_dir = Path(__file__).parent / 'outputs'
output_dir.mkdir(exist_ok=True)

results.to_csv(output_dir / 'dataset1_comparison.csv', index=False)

print(f"\n💾 Results saved to: {output_dir / 'dataset1_comparison.csv'}")
print("\n✅ ANALYSIS COMPLETE!")
