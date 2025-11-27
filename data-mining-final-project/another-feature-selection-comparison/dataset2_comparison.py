"""
DATASET 2: Wave Data - Feature Selection Comparison
Objective: Compare RFECV vs Mutual Information feature selection

Dataset Characteristics:
- Ocean wave measurements (6 Excel files)
- Auto-detection of features and target
- Challenge: Multiple numeric features, unknown domain knowledge

Feature Selection Methods:
1. RFECV: Recursive Feature Elimination with Cross-Validation
2. Mutual Information: Information-theoretic dependency measure

Model: Ridge Regression (simple, anti-overfitting via L2 regularization)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
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
print("🌊 DATASET 2: WAVE DATA - FEATURE SELECTION COMPARISON")
print("="*80)

# ===== 1. DATA LOADING =====
print("\n[1/7] 📂 Loading wave data from Excel files...")

data_dir = Path(__file__).parent.parent / 'dataset-type-2'
excel_files = sorted(glob.glob(str(data_dir / '*.xlsx')))

if not excel_files:
    print("❌ No Excel files found!")
    exit(1)

print(f"   Found {len(excel_files)} Excel files")

# Try loading with different approaches
dfs = []
for file in excel_files:
    try:
        # Excel has header at row 4 (index 4), data starts at row 5
        df = pd.read_excel(file, engine='openpyxl', header=4)
        
        # Remove completely empty rows/columns
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        dfs.append(df)
        print(f"   ✅ Loaded: {Path(file).name} ({len(df)} rows, {len(df.columns)} cols)")
    except Exception as e:
        print(f"   ⚠️ Failed to load {Path(file).name}: {e}")
        continue

if not dfs:
    print("❌ Error: Cannot load any Excel files.")
    exit(1)

df = pd.concat(dfs, ignore_index=True)
print(f"\n✅ Combined dataset: {len(df):,} rows, {len(df.columns)} columns")

# ===== 2. AUTO-DETECT FEATURES =====
print("\n[2/7] 🔍 Auto-detecting features and target...")

# Get numeric columns only (auto-convert if needed)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"   Found {len(numeric_cols)} numeric columns")

if len(numeric_cols) < 2:
    print("❌ Error: Need at least 2 numeric columns (1 target + 1 feature)")
    exit(1)

# Auto-assign: Last column = target, others = features
target_col = numeric_cols[-1]
feature_cols = numeric_cols[:-1]

print(f"✅ Auto-detected:")
print(f"   Target: '{target_col}'")
print(f"   Features ({len(feature_cols)}): {feature_cols[:3]}..." if len(feature_cols) > 3 else f"   Features: {feature_cols}")

# Remove rows with NaN in target ONLY (not all features)
df_clean = df[numeric_cols].copy()

# Check NaN stats before cleaning
print(f"   NaN stats: {df_clean.isnull().sum().sum()} total NaN values")

# Remove rows where target is NaN
df_clean = df_clean.dropna(subset=[target_col])
print(f"   After removing NaN target: {len(df_clean):,} rows")

# Fill remaining NaN in features with median
for col in feature_cols:
    if df_clean[col].isnull().any():
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

print(f"   Final samples: {len(df_clean):,}")

X = df_clean[feature_cols]
y = df_clean[target_col]

# Convert column names to strings (sklearn requirement)
X.columns = X.columns.astype(str)
feature_cols_str = X.columns.tolist()

print(f"   Target distribution: median={y.median():.2f}, mean={y.mean():.2f}, std={y.std():.2f}")

# ===== 3. TRAIN/TEST SPLIT =====
print("\n[3/7] 📊 Train/test split (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardization
scaler = StandardScaler()

# Handle any remaining NaN by filling with 0 before scaling
X_train_filled = X_train.fillna(0)
X_test_filled = X_test.fillna(0)

X_train_scaled = scaler.fit_transform(X_train_filled)
X_test_scaled = scaler.transform(X_test_filled)

print(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,}")

# ===== 4. BASELINE (NO FEATURE SELECTION) =====
print("\n[4/7] 📍 BASELINE - All features (no selection)...")

model_baseline = Ridge(alpha=10.0, random_state=42)

start_time = time.time()
model_baseline.fit(X_train_scaled, y_train)
baseline_train_time = time.time() - start_time

y_pred_baseline = model_baseline.predict(X_test_scaled)

baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
baseline_r2 = r2_score(y_test, y_pred_baseline)

print(f"✅ Baseline: {len(feature_cols)} features")
print(f"   RMSE: {baseline_rmse:.4f} | R²: {baseline_r2:.4f} | Train time: {baseline_train_time:.4f}s")

# ===== 5. RFECV FEATURE SELECTION =====
print("\n[5/7] 🔄 RFECV - Recursive Feature Elimination...")

start_time = time.time()

# Determine min features (at least 1, max 50% of features)
min_features = max(1, min(3, len(feature_cols) // 2))

rfecv = RFECV(
    estimator=Ridge(alpha=10.0, random_state=42),
    step=1,
    cv=5,
    min_features_to_select=min_features,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
rfecv.fit(X_train_scaled, y_train)

rfecv_selection_time = time.time() - start_time

selected_rfecv = [f for f, s in zip(feature_cols_str, rfecv.support_) if s]
print(f"✅ Selected {len(selected_rfecv)}/{len(feature_cols_str)} features ({(1-len(selected_rfecv)/len(feature_cols_str))*100:.1f}% reduction)")
print(f"   Features: {selected_rfecv[:5]}..." if len(selected_rfecv) > 5 else f"   Features: {selected_rfecv}")

# Train with selected features
model_rfecv = Ridge(alpha=10.0, random_state=42)

start_time = time.time()
model_rfecv.fit(X_train_scaled[:, rfecv.support_], y_train)
rfecv_train_time = time.time() - start_time

y_pred_rfecv = model_rfecv.predict(X_test_scaled[:, rfecv.support_])

rfecv_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rfecv))
rfecv_r2 = r2_score(y_test, y_pred_rfecv)

print(f"   RMSE: {rfecv_rmse:.4f} | R²: {rfecv_r2:.4f}")
print(f"   Selection: {rfecv_selection_time:.4f}s | Training: {rfecv_train_time:.4f}s")

# ===== 6. MUTUAL INFORMATION FEATURE SELECTION =====
print("\n[6/7] 📈 Mutual Information - Information-theoretic selection...")

start_time = time.time()

mi_scores = mutual_info_regression(X_train_scaled, y_train, random_state=42)
mi_df = pd.DataFrame({
    'feature': feature_cols_str,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

# Select same number as RFECV for fair comparison
selected_mi = mi_df.head(len(selected_rfecv))['feature'].tolist()
mi_selection_time = time.time() - start_time

print(f"✅ Selected {len(selected_mi)}/{len(feature_cols_str)} features ({(1-len(selected_mi)/len(feature_cols_str))*100:.1f}% reduction)")
print(f"   Top features: {selected_mi[:5]}..." if len(selected_mi) > 5 else f"   Features: {selected_mi}")

# Get column indices
mi_indices = [i for i, f in enumerate(feature_cols_str) if f in selected_mi]

# Train with selected features
model_mi = Ridge(alpha=10.0, random_state=42)

start_time = time.time()
model_mi.fit(X_train_scaled[:, mi_indices], y_train)
mi_train_time = time.time() - start_time

y_pred_mi = model_mi.predict(X_test_scaled[:, mi_indices])

mi_rmse = np.sqrt(mean_squared_error(y_test, y_pred_mi))
mi_r2 = r2_score(y_test, y_pred_mi)

print(f"   RMSE: {mi_rmse:.4f} | R²: {mi_r2:.4f}")
print(f"   Selection: {mi_selection_time:.4f}s | Training: {mi_train_time:.4f}s")

# ===== 7. STATISTICAL COMPARISON =====
print("\n[7/7] 📊 Statistical significance testing...")

errors_baseline = np.abs(y_test.values - y_pred_baseline)
errors_rfecv = np.abs(y_test.values - y_pred_rfecv)
errors_mi = np.abs(y_test.values - y_pred_mi)

t_rfecv, p_rfecv = stats.ttest_rel(errors_baseline, errors_rfecv)
t_mi, p_mi = stats.ttest_rel(errors_baseline, errors_mi)

print(f"✅ Paired t-test results:")
print(f"   RFECV vs Baseline:       p={p_rfecv:.4f} {'✅ Significant' if p_rfecv < 0.05 else '⚠️ Not significant'}")
print(f"   Mutual Info vs Baseline: p={p_mi:.4f} {'✅ Significant' if p_mi < 0.05 else '⚠️ Not significant'}")

# ===== RESULTS SUMMARY =====
print("\n" + "="*80)
print("📊 COMPARATIVE RESULTS")
print("="*80)

results = pd.DataFrame({
    'Method': ['Baseline (All Features)', 'RFECV', 'Mutual Information'],
    'Features': [len(feature_cols_str), len(selected_rfecv), len(selected_mi)],
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

rfecv_r2_improvement = ((rfecv_r2 - baseline_r2) / (1 - baseline_r2)) * 100 if baseline_r2 < 1 else 0
mi_r2_improvement = ((mi_r2 - baseline_r2) / (1 - baseline_r2)) * 100 if baseline_r2 < 1 else 0

print(f"\nRFECV:")
print(f"  Feature reduction: {(1-len(selected_rfecv)/len(feature_cols_str))*100:.1f}%")
print(f"  RMSE improvement: {rfecv_rmse_improvement:+.2f}%")
print(f"  R² improvement: {rfecv_r2_improvement:+.2f}%")
print(f"  Training speedup: {baseline_train_time/rfecv_train_time:.2f}x")

print(f"\nMutual Information:")
print(f"  Feature reduction: {(1-len(selected_mi)/len(feature_cols_str))*100:.1f}%")
print(f"  RMSE improvement: {mi_rmse_improvement:+.2f}%")
print(f"  R² improvement: {mi_r2_improvement:+.2f}%")
print(f"  Training speedup: {baseline_train_time/mi_train_time:.2f}x")

# Winner determination
best_method = results.loc[results['RMSE'].idxmin(), 'Method']
print(f"\n🏆 Best method by RMSE: {best_method}")

# Save results
output_dir = Path(__file__).parent / 'outputs'
output_dir.mkdir(exist_ok=True)

results.to_csv(output_dir / 'dataset2_comparison.csv', index=False)

print(f"\n💾 Results saved to: {output_dir / 'dataset2_comparison.csv'}")
print("\n✅ ANALYSIS COMPLETE!")
