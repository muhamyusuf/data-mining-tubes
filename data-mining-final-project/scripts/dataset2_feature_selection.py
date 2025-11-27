"""
Dataset 1: Pharmacy Transaction Volume Prediction
FEATURE SELECTION AS PREPROCESSING: Baseline vs RFECV vs SelectKBest
Validation: Decision Tree (simple model)

Tujuan:
- Tunjukkan bahwa preprocessing (feature selection) meningkatkan performa model
- Perbandingan 3 skenario: NO preprocessing, RFECV, SelectKBest
- Validasi menggunakan model sederhana (Decision Tree)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV, SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("📊 FEATURE SELECTION AS PREPROCESSING - DATASET 1")
print("="*80)

# Get project root directory
project_root = Path(__file__).parent.parent
output_dir = project_root / 'outputs' / 'dataset1-output'
output_dir.mkdir(parents=True, exist_ok=True)

# ===== 1. LOAD DATA =====
print("\n[1/7] 📂 Loading transaction data...")
csv_files = list((project_root / 'dataset-type-1').glob('[!A]*.csv'))
df_list = [pd.read_csv(f) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)
print(f"✅ Loaded {len(df):,} rows from {len(csv_files)} files")

# ===== 2. PREPROCESSING =====
print("\n[2/7] 🔧 Basic preprocessing...")
df = df.rename(columns={
    'KODE': 'kd_obat',
    'TANGGAL': 'tgl_faktur',
    'QTY_MSK': 'qty_in',
    'QTY_KLR': 'qty_out'
})
df['tgl_faktur'] = pd.to_datetime(df['tgl_faktur'], format='%d-%m-%y', errors='coerce')
df = df.dropna(subset=['tgl_faktur'])

# Total transaction volume (IN + OUT)
df['qty_total'] = df['qty_in'].fillna(0) + df['qty_out'].fillna(0)
df = df[df['qty_total'] > 0]

# Top 30 products
top_30 = df.groupby('kd_obat')['qty_total'].sum().nlargest(30).index
df = df[df['kd_obat'].isin(top_30)]

# Daily aggregation
df_daily = df.groupby(['kd_obat', 'tgl_faktur']).agg({
    'qty_in': 'sum',
    'qty_out': 'sum',
    'qty_total': 'sum'
}).reset_index()

df_daily = df_daily.sort_values(['kd_obat', 'tgl_faktur']).reset_index(drop=True)
print(f"✅ Daily aggregated: {len(df_daily):,} rows")

# ===== 3. FEATURE ENGINEERING =====
print("\n[3/7] 🔨 Feature engineering...")

df_features = df_daily.copy()

# Temporal features
df_features['day'] = df_features['tgl_faktur'].dt.day
df_features['month'] = df_features['tgl_faktur'].dt.month
df_features['day_of_week'] = df_features['tgl_faktur'].dt.dayofweek
df_features['week'] = df_features['tgl_faktur'].dt.isocalendar().week
df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)

# Lag features (SHIFT to avoid leakage!)
for lag in [1, 2, 3, 7]:
    df_features[f'qty_lag_{lag}'] = df_features.groupby('kd_obat')['qty_total'].shift(lag)

# Rolling features (MUST shift first to avoid leakage!)
for window in [7, 14]:
    df_features[f'qty_roll_mean_{window}'] = df_features.groupby('kd_obat')['qty_total'].shift(1).rolling(window).mean()
    df_features[f'qty_roll_std_{window}'] = df_features.groupby('kd_obat')['qty_total'].shift(1).rolling(window).std()

# Ratio features (use previous day data only!)
df_features['qty_in_lag1'] = df_features.groupby('kd_obat')['qty_in'].shift(1)
df_features['qty_out_lag1'] = df_features.groupby('kd_obat')['qty_out'].shift(1)
df_features['in_out_ratio'] = df_features['qty_in_lag1'] / (df_features['qty_out_lag1'] + 1)

# Drop temporary columns
df_features = df_features.drop(columns=['qty_in_lag1', 'qty_out_lag1'], errors='ignore')

# Remove NaN from lag/rolling
df_features = df_features.dropna()

feature_cols = [c for c in df_features.columns if c not in ['kd_obat', 'tgl_faktur', 'qty_in', 'qty_out', 'qty_total']]
X = df_features[feature_cols]
y = df_features['qty_total']

print(f"✅ Features created: {len(feature_cols)} features, {len(X):,} samples")
print(f"   Feature names: {feature_cols}")
print(f"   Target range: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")

# ===== 4. TRAIN/TEST SPLIT =====
print("\n[4/7] 📊 Train/Test split (80/20, time series)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize features (important for Ridge)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

print(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"   Features standardized (mean=0, std=1)")

# ===== 5. BASELINE (NO FEATURE SELECTION) =====
print("\n[5/7] 📍 BASELINE - No Feature Selection (All features)...")
print(f"   Using ALL {len(feature_cols)} features")

model_baseline = Ridge(alpha=10.0, random_state=42)  # Increased regularization
model_baseline.fit(X_train_scaled, y_train)

y_pred_baseline = model_baseline.predict(X_test_scaled)
y_train_pred_baseline = model_baseline.predict(X_train_scaled)

rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
r2_baseline = r2_score(y_test, y_pred_baseline)
r2_train_baseline = r2_score(y_train, y_train_pred_baseline)

print(f"✅ BASELINE Results:")
print(f"   RMSE: {rmse_baseline:.4f} | MAE: {mae_baseline:.4f} | R²: {r2_baseline:.4f}")

# ===== 6. RFECV (FEATURE SELECTION PREPROCESSING) =====
print("\n[6/7] 🔄 RFECV - Feature Selection Preprocessing...")

estimator = Ridge(alpha=10.0, random_state=42)  # Match baseline regularization

rfecv = RFECV(
    estimator=estimator,
    step=1,
    cv=5,
    min_features_to_select=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
rfecv.fit(X_train_scaled, y_train)

selected_rfecv = [f for f, s in zip(feature_cols, rfecv.support_) if s]
print(f"✅ RFECV selected {len(selected_rfecv)}/{len(feature_cols)} features")
print(f"   Selected: {selected_rfecv}")

# Train with selected features
model_rfecv = Ridge(alpha=10.0, random_state=42)  # Match baseline regularization
model_rfecv.fit(X_train_scaled[selected_rfecv], y_train)

y_pred_rfecv = model_rfecv.predict(X_test_scaled[selected_rfecv])
y_train_pred_rfecv = model_rfecv.predict(X_train_scaled[selected_rfecv])

rmse_rfecv = np.sqrt(mean_squared_error(y_test, y_pred_rfecv))
mae_rfecv = mean_absolute_error(y_test, y_pred_rfecv)
r2_rfecv = r2_score(y_test, y_pred_rfecv)
r2_train_rfecv = r2_score(y_train, y_train_pred_rfecv)

print(f"✅ RFECV Results:")
print(f"   RMSE: {rmse_rfecv:.4f} | MAE: {mae_rfecv:.4f} | R²: {r2_rfecv:.4f}")

# ===== 7. SELECTKBEST (FEATURE SELECTION PREPROCESSING) =====
print("\n[7/7] 📈 SelectKBest - Feature Selection Preprocessing...")

selector = SelectKBest(score_func=f_regression, k=len(selected_rfecv))
selector.fit(X_train_scaled, y_train)

scores = pd.DataFrame({
    'feature': feature_cols,
    'score': selector.scores_
}).sort_values('score', ascending=False)

selected_kbest = scores.head(len(selected_rfecv))['feature'].tolist()
print(f"✅ SelectKBest selected {len(selected_kbest)}/{len(feature_cols)} features")
print(f"   Selected: {selected_kbest}")

# Train with selected features
model_kbest = Ridge(alpha=10.0, random_state=42)  # Match baseline regularization
model_kbest.fit(X_train_scaled[selected_kbest], y_train)

y_pred_kbest = model_kbest.predict(X_test_scaled[selected_kbest])
y_train_pred_kbest = model_kbest.predict(X_train_scaled[selected_kbest])

rmse_kbest = np.sqrt(mean_squared_error(y_test, y_pred_kbest))
mae_kbest = mean_absolute_error(y_test, y_pred_kbest)
r2_kbest = r2_score(y_test, y_pred_kbest)
r2_train_kbest = r2_score(y_train, y_train_pred_kbest)

print(f"✅ SelectKBest Results:")
print(f"   RMSE: {rmse_kbest:.4f} | MAE: {mae_kbest:.4f} | R²: {r2_kbest:.4f}")

# ===== RESULTS & ANALYSIS =====
print("\n" + "="*80)
print("📊 PREPROCESSING IMPACT ANALYSIS")
print("="*80)

results = pd.DataFrame({
    'Method': ['BASELINE\n(No Preprocessing)', 'RFECV\n(Preprocessing)', 'SelectKBest\n(Preprocessing)'],
    'N_Features': [len(feature_cols), len(selected_rfecv), len(selected_kbest)],
    'RMSE_Test': [rmse_baseline, rmse_rfecv, rmse_kbest],
    'MAE_Test': [mae_baseline, mae_rfecv, mae_kbest],
    'R2_Test': [r2_baseline, r2_rfecv, r2_kbest],
    'R2_Train': [r2_train_baseline, r2_train_rfecv, r2_train_kbest],
    'Overfit_Gap': [
        abs(r2_train_baseline - r2_baseline),
        abs(r2_train_rfecv - r2_rfecv),
        abs(r2_train_kbest - r2_kbest)
    ]
})

print("\n" + results.to_string(index=False))

# Improvement calculation
rmse_improvement_rfecv = ((rmse_baseline - rmse_rfecv) / rmse_baseline) * 100
rmse_improvement_kbest = ((rmse_baseline - rmse_kbest) / rmse_baseline) * 100
r2_improvement_rfecv = ((r2_rfecv - r2_baseline) / (1 - r2_baseline)) * 100
r2_improvement_kbest = ((r2_kbest - r2_baseline) / (1 - r2_baseline)) * 100

print("\n" + "="*80)
print("📈 PREPROCESSING BENEFITS")
print("="*80)
print(f"\nFeature Reduction:")
print(f"  RFECV:      {len(feature_cols)} → {len(selected_rfecv)} features ({(1 - len(selected_rfecv)/len(feature_cols))*100:.1f}% reduction)")
print(f"  SelectKBest: {len(feature_cols)} → {len(selected_kbest)} features ({(1 - len(selected_kbest)/len(feature_cols))*100:.1f}% reduction)")

print(f"\nPerformance Improvement (vs Baseline):")
print(f"  RFECV:")
print(f"    RMSE: {rmse_improvement_rfecv:+.2f}% {'✅ BETTER' if rmse_improvement_rfecv > 0 else '❌ WORSE'}")
print(f"    R²:   {r2_improvement_rfecv:+.2f}% {'✅ BETTER' if r2_improvement_rfecv > 0 else '❌ WORSE'}")
print(f"  SelectKBest:")
print(f"    RMSE: {rmse_improvement_kbest:+.2f}% {'✅ BETTER' if rmse_improvement_kbest > 0 else '❌ WORSE'}")
print(f"    R²:   {r2_improvement_kbest:+.2f}% {'✅ BETTER' if r2_improvement_kbest > 0 else '❌ WORSE'}")

print(f"\nOverfitting Reduction:")
print(f"  Baseline:    Gap = {abs(r2_train_baseline - r2_baseline):.4f}")
print(f"  RFECV:       Gap = {abs(r2_train_rfecv - r2_rfecv):.4f} ({((abs(r2_train_baseline - r2_baseline) - abs(r2_train_rfecv - r2_rfecv)) / abs(r2_train_baseline - r2_baseline) * 100):+.1f}%)")
print(f"  SelectKBest: Gap = {abs(r2_train_kbest - r2_kbest):.4f} ({((abs(r2_train_baseline - r2_baseline) - abs(r2_train_kbest - r2_kbest)) / abs(r2_train_baseline - r2_baseline) * 100):+.1f}%)")

# Statistical significance test
print("\n" + "="*80)
print("📊 STATISTICAL SIGNIFICANCE (Paired T-Test)")
print("="*80)

# Compare prediction errors
errors_baseline = np.abs(y_test.values - y_pred_baseline)
errors_rfecv = np.abs(y_test.values - y_pred_rfecv)
errors_kbest = np.abs(y_test.values - y_pred_kbest)

t_stat_rfecv, p_value_rfecv = stats.ttest_rel(errors_baseline, errors_rfecv)
t_stat_kbest, p_value_kbest = stats.ttest_rel(errors_baseline, errors_kbest)

print(f"\nBaseline vs RFECV:")
print(f"  t-statistic: {t_stat_rfecv:.4f}")
print(f"  p-value:     {p_value_rfecv:.6f} {'✅ Significant (p < 0.05)' if p_value_rfecv < 0.05 else '⚠️ Not significant'}")

print(f"\nBaseline vs SelectKBest:")
print(f"  t-statistic: {t_stat_kbest:.4f}")
print(f"  p-value:     {p_value_kbest:.6f} {'✅ Significant (p < 0.05)' if p_value_kbest < 0.05 else '⚠️ Not significant'}")

# Best method
best_method = results.loc[results['RMSE_Test'].idxmin(), 'Method'].strip()
print(f"\n🏆 BEST METHOD: {best_method}")

# ===== SAVE OUTPUTS =====
print("\n💾 Saving results...")

results.to_csv(output_dir / 'preprocessing_comparison.csv', index=False)

pd.DataFrame({
    'Baseline_Features': pd.Series(feature_cols),
    'RFECV_Features': pd.Series(selected_rfecv),
    'SelectKBest_Features': pd.Series(selected_kbest)
}).to_csv(output_dir / 'selected_features.csv', index=False)

pd.DataFrame({
    'Actual': y_test.values,
    'Baseline_Pred': y_pred_baseline,
    'RFECV_Pred': y_pred_rfecv,
    'SelectKBest_Pred': y_pred_kbest
}).to_csv(output_dir / 'test_predictions.csv', index=False)

scores.to_csv(output_dir / 'all_feature_scores.csv', index=False)

# ===== VISUALIZATION =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Metrics comparison
methods = ['Baseline', 'RFECV', 'SelectKBest']
metrics = {
    'RMSE': [rmse_baseline, rmse_rfecv, rmse_kbest],
    'MAE': [mae_baseline, mae_rfecv, mae_kbest],
    'R² (Test)': [r2_baseline, r2_rfecv, r2_kbest]
}

x = np.arange(len(methods))
width = 0.25

for i, (metric_name, values) in enumerate(metrics.items()):
    axes[0,0].bar(x + i*width, values, width, label=metric_name, alpha=0.8)

axes[0,0].set_ylabel('Score')
axes[0,0].set_title('Performance Comparison')
axes[0,0].set_xticks(x + width)
axes[0,0].set_xticklabels(methods)
axes[0,0].legend()
axes[0,0].grid(axis='y', alpha=0.3)

# 2. Feature count
axes[0,1].bar(methods, [len(feature_cols), len(selected_rfecv), len(selected_kbest)], alpha=0.8, color=['red', 'steelblue', 'coral'])
axes[0,1].set_ylabel('Number of Features')
axes[0,1].set_title('Feature Count After Preprocessing')
axes[0,1].grid(axis='y', alpha=0.3)

# 3. Overfitting comparison
overfit_gaps = [
    abs(r2_train_baseline - r2_baseline),
    abs(r2_train_rfecv - r2_rfecv),
    abs(r2_train_kbest - r2_kbest)
]
axes[1,0].bar(methods, overfit_gaps, alpha=0.8, color=['red', 'steelblue', 'coral'])
axes[1,0].set_ylabel('Overfitting Gap (|R²_train - R²_test|)')
axes[1,0].set_title('Overfitting Control')
axes[1,0].axhline(y=0.05, color='green', linestyle='--', label='Good (<0.05)')
axes[1,0].axhline(y=0.15, color='orange', linestyle='--', label='Acceptable (<0.15)')
axes[1,0].legend()
axes[1,0].grid(axis='y', alpha=0.3)

# 4. Predictions scatter (best method only)
best_idx = results['RMSE_Test'].idxmin()
if best_idx == 0:
    best_pred = y_pred_baseline
    best_name = 'Baseline'
elif best_idx == 1:
    best_pred = y_pred_rfecv
    best_name = 'RFECV'
else:
    best_pred = y_pred_kbest
    best_name = 'SelectKBest'

axes[1,1].scatter(y_test, y_pred_baseline, alpha=0.3, s=20, label='Baseline', color='red')
axes[1,1].scatter(y_test, best_pred, alpha=0.5, s=30, label=f'{best_name} (Best)')
axes[1,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axes[1,1].set_xlabel('Actual')
axes[1,1].set_ylabel('Predicted')
axes[1,1].set_title('Predictions: Baseline vs Best Method')
axes[1,1].legend()
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'preprocessing_impact.png', dpi=150)

# Summary report
summary = f"""FEATURE SELECTION AS PREPROCESSING - DATASET 1 (PHARMACY)
{'='*80}

MODEL: Ridge Regression (Simple linear model for validation)
PREPROCESSING METHODS: Feature Selection (RFECV, SelectKBest)
NOTE: Ridge is a SIMPLE model - feature selection should show CLEAR impact

DATA:
- Total samples: {len(X):,}
- Train/Test: {len(X_train):,} / {len(X_test):,}
- Original features: {len(feature_cols)}

RESULTS:
{'Method':<20} {'Features':<12} {'RMSE':<10} {'R²':<10} {'Overfit Gap':<12}
{'-'*80}
{'Baseline (No Prep)':<20} {len(feature_cols):<12} {rmse_baseline:<10.4f} {r2_baseline:<10.4f} {abs(r2_train_baseline - r2_baseline):<12.4f}
{'RFECV':<20} {len(selected_rfecv):<12} {rmse_rfecv:<10.4f} {r2_rfecv:<10.4f} {abs(r2_train_rfecv - r2_rfecv):<12.4f}
{'SelectKBest':<20} {len(selected_kbest):<12} {rmse_kbest:<10.4f} {r2_kbest:<10.4f} {abs(r2_train_kbest - r2_kbest):<12.4f}

PREPROCESSING IMPACT:
- RFECV:      RMSE {rmse_improvement_rfecv:+.2f}% | R² {r2_improvement_rfecv:+.2f}% | {len(feature_cols)-len(selected_rfecv)} features removed
- SelectKBest: RMSE {rmse_improvement_kbest:+.2f}% | R² {r2_improvement_kbest:+.2f}% | {len(feature_cols)-len(selected_kbest)} features removed

STATISTICAL SIGNIFICANCE (p-value):
- Baseline vs RFECV:      p = {p_value_rfecv:.6f} {'✅ Significant' if p_value_rfecv < 0.05 else '❌ Not significant'}
- Baseline vs SelectKBest: p = {p_value_kbest:.6f} {'✅ Significant' if p_value_kbest < 0.05 else '❌ Not significant'}

BEST METHOD: {best_method}

CONCLUSION:
Feature selection preprocessing {'DOES' if (rmse_improvement_rfecv > 0 or rmse_improvement_kbest > 0) else 'DOES NOT'} improve model performance.
{'RFECV' if rmse_rfecv < rmse_kbest else 'SelectKBest'} shows better results with {abs(rmse_rfecv - rmse_kbest):.4f} lower RMSE.
"""

with open(output_dir / 'analysis_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"\n✅ Saved to {output_dir}/")
print("\n✅ ANALYSIS COMPLETE!\n")
