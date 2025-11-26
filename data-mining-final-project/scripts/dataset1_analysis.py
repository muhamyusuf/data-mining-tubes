"""
Dataset 1: Pharmacy Analysis - RFECV vs SelectKBest (F-regression)
IMPROVED VERSION:
- Forces min 5 features in RFECV
- Uses transaction volume (QTY_MSK + QTY_KLR)
- Stronger regularization
- Proper time series split
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV, SelectKBest, f_regression
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("📊 DATASET 1: PHARMACY TRANSACTION VOLUME PREDICTION")
print("="*70)

# Get project root directory
project_root = Path(__file__).parent.parent
output_dir = project_root / 'outputs' / 'dataset1-output'
output_dir.mkdir(parents=True, exist_ok=True)

# ===== 1. LOAD DATA =====
print("\n[1/6] 📂 Loading transaction data...")
csv_files = list((project_root / 'dataset-type-1').glob('[!A]*.csv'))  # Exclude A*.csv (stock files)
df_list = [pd.read_csv(f) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)
print(f"✅ Loaded {len(df):,} rows from {len(csv_files)} files")

# ===== 2. PREPROCESSING =====
print("\n[2/6] 🔧 Preprocessing...")
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

# Keep only actual transactions
df = df[df['qty_total'] > 0]
print(f"   Transactions with volume > 0: {len(df):,}")

# Top 30 products by total volume
top_30 = df.groupby('kd_obat')['qty_total'].sum().nlargest(30).index
df = df[df['kd_obat'].isin(top_30)]
print(f"   Top 30 products: {len(df):,} transactions")

# Daily aggregation
df_daily = df.groupby(['kd_obat', 'tgl_faktur']).agg({
    'qty_total': 'sum',
    'qty_in': 'sum',
    'qty_out': 'sum'
}).reset_index()
print(f"   Daily level: {len(df_daily):,} rows")

# Feature engineering
print("   Creating features...")
features_list = []
for product in top_30:
    df_prod = df_daily[df_daily['kd_obat'] == product].sort_values('tgl_faktur').copy()
    
    # Lag features
    for lag in [1, 2, 3, 7, 14]:
        df_prod[f'lag_{lag}'] = df_prod['qty_total'].shift(lag)
    
    # Rolling statistics  
    for window in [7, 14]:
        df_prod[f'roll_{window}_mean'] = df_prod['qty_total'].rolling(window).mean()
        df_prod[f'roll_{window}_std'] = df_prod['qty_total'].rolling(window).std()
    
    # Ratio
    df_prod['in_out_ratio'] = df_prod['qty_in'] / (df_prod['qty_out'] + 0.1)
    
    # Temporal
    df_prod['day'] = df_prod['tgl_faktur'].dt.day
    df_prod['month'] = df_prod['tgl_faktur'].dt.month
    df_prod['day_of_week'] = df_prod['tgl_faktur'].dt.dayofweek
    df_prod['week'] = df_prod['tgl_faktur'].dt.isocalendar().week
    df_prod['is_weekend'] = (df_prod['day_of_week'] >= 5).astype(int)
    
    features_list.append(df_prod)

df_features = pd.concat(features_list, ignore_index=True).dropna()
print(f"✅ Feature matrix: {len(df_features):,} rows")

# ===== 3. PREPARE DATA =====
print("\n[3/6] 📊 Train/Test split...")
feature_cols = [c for c in df_features.columns if c not in ['kd_obat', 'tgl_faktur', 'qty_total', 'qty_in', 'qty_out']]
X = df_features[feature_cols]
y = df_features['qty_total']

# Time series split (no shuffle!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {len(feature_cols)}")

# ===== 4. RFECV =====
print("\n[4/6] 🔄 RFECV (min 5 features)...")
estimator = LGBMRegressor(
    n_estimators=150, learning_rate=0.05, max_depth=4,
    min_child_samples=30, reg_alpha=0.3, reg_lambda=0.3,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1
)
rfecv = RFECV(estimator, step=1, cv=5, min_features_to_select=5, 
              scoring='neg_mean_squared_error', n_jobs=-1)
rfecv.fit(X_train, y_train)

selected_rfecv = [f for f, s in zip(feature_cols, rfecv.support_) if s]
print(f"✅ Selected {len(selected_rfecv)} features")

# Train final model
model_rfecv = LGBMRegressor(
    n_estimators=150, learning_rate=0.05, max_depth=4,
    min_child_samples=30, reg_alpha=0.3, reg_lambda=0.3,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1
)
model_rfecv.fit(X_train[selected_rfecv], y_train)
y_pred_rfecv = model_rfecv.predict(X_test[selected_rfecv])
y_train_pred_rfecv = model_rfecv.predict(X_train[selected_rfecv])

rmse_rfecv = np.sqrt(mean_squared_error(y_test, y_pred_rfecv))
mae_rfecv = mean_absolute_error(y_test, y_pred_rfecv)
r2_rfecv = r2_score(y_test, y_pred_rfecv)
r2_train_rfecv = r2_score(y_train, y_train_pred_rfecv)

# ===== 5. SELECTKBEST =====
print("\n[5/6] 📈 SelectKBest (F-regression)...")
selector = SelectKBest(f_regression, k=len(selected_rfecv))
selector.fit(X_train, y_train)

scores = pd.DataFrame({'feature': feature_cols, 'score': selector.scores_}).sort_values('score', ascending=False)
selected_kbest = scores.head(len(selected_rfecv))['feature'].tolist()
print(f"✅ Selected {len(selected_kbest)} features")

model_kbest = LGBMRegressor(
    n_estimators=150, learning_rate=0.05, max_depth=4,
    min_child_samples=30, reg_alpha=0.3, reg_lambda=0.3,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1
)
model_kbest.fit(X_train[selected_kbest], y_train)
y_pred_kbest = model_kbest.predict(X_test[selected_kbest])
y_train_pred_kbest = model_kbest.predict(X_train[selected_kbest])

rmse_kbest = np.sqrt(mean_squared_error(y_test, y_pred_kbest))
mae_kbest = mean_absolute_error(y_test, y_pred_kbest)
r2_kbest = r2_score(y_test, y_pred_kbest)
r2_train_kbest = r2_score(y_train, y_train_pred_kbest)

# ===== 6. RESULTS =====
print("\n[6/6] 📊 RESULTS")
print("="*70)
print(f"{'Metric':<25} {'RFECV':>15} {'SelectKBest':>15} {'Winner':>15}")
print("-"*70)
print(f"{'Features':<25} {len(selected_rfecv):>15} {len(selected_kbest):>15} {'=':>15}")
print(f"{'RMSE (Test)':<25} {rmse_rfecv:>15.2f} {rmse_kbest:>15.2f} {'RFECV' if rmse_rfecv < rmse_kbest else 'SelectKBest':>15}")
print(f"{'MAE (Test)':<25} {mae_rfecv:>15.2f} {mae_kbest:>15.2f} {'RFECV' if mae_rfecv < mae_kbest else 'SelectKBest':>15}")
print(f"{'R2 (Test)':<25} {r2_rfecv:>15.4f} {r2_kbest:>15.4f} {'RFECV' if r2_rfecv > r2_kbest else 'SelectKBest':>15}")
print(f"{'R2 (Train)':<25} {r2_train_rfecv:>15.4f} {r2_train_kbest:>15.4f} {'-':>15}")
print(f"{'Overfit Gap':<25} {abs(r2_train_rfecv - r2_rfecv):>15.4f} {abs(r2_train_kbest - r2_kbest):>15.4f} {'RFECV' if abs(r2_train_rfecv - r2_rfecv) < abs(r2_train_kbest - r2_kbest) else 'SelectKBest':>15}")
print("="*70)

max_gap = max(abs(r2_train_rfecv - r2_rfecv), abs(r2_train_kbest - r2_kbest))
if max_gap > 0.15:
    print("\n⚠️  HIGH overfitting!")
elif max_gap > 0.05:
    print("\n⚠️  Moderate overfitting (acceptable)")
else:
    print("\n✅ Good generalization!")

winner = "RFECV" if rmse_rfecv < rmse_kbest else "SelectKBest"
print(f"\n🏆 WINNER: {winner}")

# Save outputs
print("\n💾 Saving...")

pd.DataFrame({
    'Method': ['RFECV', 'SelectKBest'],
    'N_Features': [len(selected_rfecv), len(selected_kbest)],
    'RMSE_Test': [rmse_rfecv, rmse_kbest],
    'MAE_Test': [mae_rfecv, mae_kbest],
    'R2_Test': [r2_rfecv, r2_kbest],
    'R2_Train': [r2_train_rfecv, r2_train_kbest]
}).to_csv(output_dir / 'comparison_summary.csv', index=False)

pd.DataFrame({
    'RFECV_Features': pd.Series(selected_rfecv),
    'SelectKBest_Features': pd.Series(selected_kbest)
}).to_csv(output_dir / 'selected_features.csv', index=False)

pd.DataFrame({
    'Actual': y_test.values,
    'RFECV_Pred': y_pred_rfecv,
    'SelectKBest_Pred': y_pred_kbest
}).to_csv(output_dir / 'test_predictions.csv', index=False)

scores.to_csv(output_dir / 'all_feature_scores.csv', index=False)

# Viz
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Metrics
metrics = ['RMSE', 'MAE', 'R2\n(Test)', 'R2\n(Train)']
rfecv_vals = [rmse_rfecv/100, mae_rfecv/100, r2_rfecv, r2_train_rfecv]
kbest_vals = [rmse_kbest/100, mae_kbest/100, r2_kbest, r2_train_kbest]
x = np.arange(len(metrics))
width = 0.35
axes[0,0].bar(x - width/2, rfecv_vals, width, label='RFECV', alpha=0.8, color='steelblue')
axes[0,0].bar(x + width/2, kbest_vals, width, label='SelectKBest', alpha=0.8, color='coral')
axes[0,0].set_ylabel('Score')
axes[0,0].set_title('Performance Comparison')
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(metrics)
axes[0,0].legend()
axes[0,0].grid(axis='y', alpha=0.3)

# RFECV importance
imp = model_rfecv.feature_importances_
idx = np.argsort(imp)[::-1][:10]
axes[0,1].barh([selected_rfecv[i] for i in idx], imp[idx], alpha=0.8, color='steelblue')
axes[0,1].set_xlabel('Importance')
axes[0,1].set_title('RFECV - Top 10')
axes[0,1].grid(axis='x', alpha=0.3)

# SelectKBest scores
top = scores.head(10)
axes[1,0].barh(top['feature'], top['score'], alpha=0.8, color='coral')
axes[1,0].set_xlabel('F-Score')
axes[1,0].set_title('SelectKBest - Top 10')
axes[1,0].grid(axis='x', alpha=0.3)

# Predictions
axes[1,1].scatter(y_test, y_pred_rfecv, alpha=0.5, label='RFECV', s=30)
axes[1,1].scatter(y_test, y_pred_kbest, alpha=0.5, label='SelectKBest', s=30)
axes[1,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axes[1,1].set_xlabel('Actual')
axes[1,1].set_ylabel('Predicted')
axes[1,1].set_title('Predictions')
axes[1,1].legend()
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'results_visualization.png', dpi=150)

# Summary
summary = f"""PHARMACY TRANSACTION VOLUME PREDICTION
{'='*70}

DATA: {len(df):,} transactions -> {len(df_features):,} after features
SPLIT: {len(X_train):,} train / {len(X_test):,} test
FEATURES: {len(feature_cols)} available

RFECV: {', '.join(selected_rfecv)}
SelectKBest: {', '.join(selected_kbest)}

PERFORMANCE:
                RFECV      SelectKBest
RMSE          {rmse_rfecv:>7.2f}     {rmse_kbest:>7.2f}
MAE           {mae_rfecv:>7.2f}     {mae_kbest:>7.2f}
R2 (Test)     {r2_rfecv:>7.4f}     {r2_kbest:>7.4f}
R2 (Train)    {r2_train_rfecv:>7.4f}     {r2_train_kbest:>7.4f}

WINNER: {winner}
"""
with open(output_dir / 'analysis_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"✅ Saved to dataset1-output/")
print("\n✅ DONE!\n")
