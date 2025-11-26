"""
Dataset 2: Wave Analysis - RFECV vs SelectKBest (F-regression)
IMPROVED VERSION:
- Forces min 3 features in RFECV
- Auto-detects numeric columns
- StandardScaler for wave data
- Stronger regularization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🌊 DATASET 2: WAVE PARAMETER PREDICTION")
print("="*70)

# Get project root directory
project_root = Path(__file__).parent.parent
output_dir = project_root / 'outputs' / 'dataset2-output'
output_dir.mkdir(parents=True, exist_ok=True)

# ===== 1. LOAD DATA =====
print("\n[1/6] 📂 Loading Excel files...")
excel_files = list((project_root / 'dataset-type-2').glob('*.xlsx')) + list((project_root / 'dataset-type-2').glob('*.xls'))

if len(excel_files) == 0:
    print("❌ No Excel files found!")
    exit(1)

df_list = []
proper_columns = None

for i, file in enumerate(excel_files):
    try:
        # Read with row 4 as header (skip 3 metadata rows)
        df_temp = pd.read_excel(file, header=3)
        
        # First file: use first row as column names
        if i == 0:
            proper_columns = df_temp.iloc[0].values.astype(str)
            df_temp = df_temp.iloc[1:].reset_index(drop=True)
            df_temp.columns = proper_columns
        else:
            # Other files: use saved column names, skip first row (it's data)
            df_temp.columns = proper_columns
            df_temp = df_temp.reset_index(drop=True)
        
        df_list.append(df_temp)
    except Exception as e:
        print(f"⚠️  Skipping {file.name}: {e}")

if len(df_list) == 0:
    print("❌ No data loaded!")
    exit(1)

df_raw = pd.concat(df_list, ignore_index=True)
print(f"✅ Loaded {len(df_raw):,} rows from {len(df_list)} files")

# ===== 2. CLEAN DATA =====
print("\n[2/6] 🔧 Cleaning data...")

# Drop completely empty rows
df_raw = df_raw.dropna(how='all')

# Auto-drop text columns (time, compass, etc.)
text_patterns = ['time', 'date', 'utc', 'gmt', 'compass', 'scale', 'dir(', 'unnamed']
cols_to_drop = []

for col in df_raw.columns:
    col_lower = str(col).lower()
    # Check if column name contains text patterns
    if any(pattern in col_lower for pattern in text_patterns):
        cols_to_drop.append(col)
    # Or check if column is mostly non-numeric
    elif df_raw[col].dtype == 'object':
        try:
            pd.to_numeric(df_raw[col], errors='raise')
        except:
            cols_to_drop.append(col)

if cols_to_drop:
    print(f"   Dropping {len(cols_to_drop)} text columns: {cols_to_drop[:3]}...")
    df_numeric = df_raw.drop(columns=cols_to_drop, errors='ignore')
else:
    df_numeric = df_raw.copy()

# Convert all remaining columns to numeric
for col in df_numeric.columns:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

# Remove rows with any NaN
df_numeric = df_numeric.dropna()

# Get numeric columns
numeric_cols = df_numeric.select_dtypes(include=[np.number]).columns.tolist()
print(f"✅ Found {len(numeric_cols)} numeric columns")

if len(numeric_cols) < 2:
    print(f"❌ Need at least 2 columns, found {len(numeric_cols)}")
    exit(1)

# Auto-assign: last column = target, rest = features
target_col = numeric_cols[-1]
feature_cols = numeric_cols[:-1]

print(f"   🎯 Target: {target_col}")
print(f"   📊 Features ({len(feature_cols)}): {feature_cols[:5]}...")

df = df_numeric[numeric_cols].copy()
print(f"✅ Clean data: {len(df):,} rows")

# ===== 3. PREPARE DATA =====
print("\n[3/6] 📊 Standardizing and splitting...")
X = df[feature_cols]
y = df[target_col]

# IMPORTANT: Standardize wave data (different scales)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

# Random split (wave data is not time series)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {len(feature_cols)}")

# ===== 4. RFECV =====
print("\n[4/6] 🔄 RFECV (min 3 features)...")
estimator = LGBMRegressor(
    n_estimators=150, 
    learning_rate=0.05, 
    max_depth=4,
    min_child_samples=20,
    reg_alpha=0.3,
    reg_lambda=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42, 
    verbose=-1
)

# Determine min features (at least 3, or half of available features)
min_features = max(3, len(feature_cols) // 2)
print(f"   Setting min_features_to_select={min_features}")

rfecv = RFECV(
    estimator=estimator,
    step=1,
    cv=5,
    min_features_to_select=min_features,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
rfecv.fit(X_train, y_train)

selected_rfecv = [f for f, s in zip(feature_cols, rfecv.support_) if s]
print(f"✅ Selected {len(selected_rfecv)} features: {selected_rfecv}")

# Train final model
model_rfecv = LGBMRegressor(
    n_estimators=150, learning_rate=0.05, max_depth=4,
    min_child_samples=20, reg_alpha=0.3, reg_lambda=0.3,
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
print(f"   Selecting top {len(selected_rfecv)} features for fair comparison")

selector = SelectKBest(score_func=f_regression, k=len(selected_rfecv))
selector.fit(X_train, y_train)

scores = pd.DataFrame({
    'feature': feature_cols,
    'score': selector.scores_
}).sort_values('score', ascending=False)

selected_kbest = scores.head(len(selected_rfecv))['feature'].tolist()
print(f"✅ Selected {len(selected_kbest)} features: {selected_kbest}")

# Train final model
model_kbest = LGBMRegressor(
    n_estimators=150, learning_rate=0.05, max_depth=4,
    min_child_samples=20, reg_alpha=0.3, reg_lambda=0.3,
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
print(f"{'RMSE (Test)':<25} {rmse_rfecv:>15.4f} {rmse_kbest:>15.4f} {'RFECV' if rmse_rfecv < rmse_kbest else 'SelectKBest':>15}")
print(f"{'MAE (Test)':<25} {mae_rfecv:>15.4f} {mae_kbest:>15.4f} {'RFECV' if mae_rfecv < mae_kbest else 'SelectKBest':>15}")
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

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Metrics comparison
metrics = ['RMSE', 'MAE', 'R2\n(Test)', 'R2\n(Train)']
rfecv_vals = [rmse_rfecv, mae_rfecv, r2_rfecv, r2_train_rfecv]
kbest_vals = [rmse_kbest, mae_kbest, r2_kbest, r2_train_kbest]
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

# RFECV feature importance
imp = model_rfecv.feature_importances_
idx = np.argsort(imp)[::-1]
axes[0,1].barh([selected_rfecv[i] for i in idx], imp[idx], alpha=0.8, color='steelblue')
axes[0,1].set_xlabel('Importance')
axes[0,1].set_title('RFECV - Feature Importance')
axes[0,1].grid(axis='x', alpha=0.3)

# SelectKBest scores
top = scores.head(len(selected_kbest))
axes[1,0].barh(top['feature'], top['score'], alpha=0.8, color='coral')
axes[1,0].set_xlabel('F-Score')
axes[1,0].set_title('SelectKBest - Top Features')
axes[1,0].grid(axis='x', alpha=0.3)

# Predictions scatter
axes[1,1].scatter(y_test, y_pred_rfecv, alpha=0.5, label='RFECV', s=30)
axes[1,1].scatter(y_test, y_pred_kbest, alpha=0.5, label='SelectKBest', s=30)
axes[1,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axes[1,1].set_xlabel('Actual')
axes[1,1].set_ylabel('Predicted')
axes[1,1].set_title('Predictions vs Actual')
axes[1,1].legend()
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'results_visualization.png', dpi=150)

# Summary
summary = f"""WAVE PARAMETER PREDICTION
{'='*70}

DATA: {len(df_raw):,} rows -> {len(df):,} after cleaning
TARGET: {target_col}
SPLIT: {len(X_train):,} train / {len(X_test):,} test
FEATURES: {len(feature_cols)} available

RFECV: {', '.join(selected_rfecv)}
SelectKBest: {', '.join(selected_kbest)}

PERFORMANCE:
                RFECV      SelectKBest
RMSE          {rmse_rfecv:>7.4f}     {rmse_kbest:>7.4f}
MAE           {mae_rfecv:>7.4f}     {mae_kbest:>7.4f}
R2 (Test)     {r2_rfecv:>7.4f}     {r2_kbest:>7.4f}
R2 (Train)    {r2_train_rfecv:>7.4f}     {r2_train_kbest:>7.4f}

WINNER: {winner}
"""
with open(output_dir / 'analysis_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"✅ Saved to dataset2-output/")
print("\n✅ DONE!\n")
