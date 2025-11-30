"""
================================================================================
DATASET 1: PHARMACY DEMAND PREDICTION - RFECV FEATURE SELECTION
================================================================================
Tugas Besar Penambangan Data 2025

OBJECTIVE: 
    Memprediksi demand level (High/Low) berdasarkan pharmacy transaction patterns.
    
PREPROCESSING FOCUS:
    RFECV (Recursive Feature Elimination with Cross-Validation) untuk:
    - Mengurangi dimensionality dari banyak temporal & transaction features
    - Menghilangkan redundant/irrelevant features
    - Meningkatkan model generalization (reduce overfitting)

VALIDATION:
    Simple ML models (Decision Tree, Naive Bayes, Logistic Regression) 
    digunakan HANYA sebagai alat ukur kualitas data preprocessing, 
    BUKAN untuk mencari model terbaik.

================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import ttest_rel
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

print("=" * 80)
print("DATASET 1: PHARMACY DEMAND PREDICTION - RFECV FEATURE SELECTION")
print("=" * 80)
print("\nTugas Besar Penambangan Data 2025")
print("Preprocessing Method: RFECV (Feature Selection for Dimensionality Reduction)")
print("Validation: Simple ML models as data quality measuring tools")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n[1/7] 📂 Loading data...")

# Get script directory
script_dir = Path(__file__).parent
data_path = script_dir.parent / "dataset-type-1"
transaction_files = ['2021.csv', '2022.csv', '2023.csv']

dfs = []
for file in transaction_files:
    df = pd.read_csv(data_path / file)
    dfs.append(df)

df_raw = pd.concat(dfs, ignore_index=True)
print(f"✅ Loaded {len(df_raw):,} transactions from {len(transaction_files)} files")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n[2/7] 🔧 Data preprocessing...")

# Convert date
df_raw['TANGGAL'] = pd.to_datetime(df_raw['TANGGAL'], format='%d-%m-%y', errors='coerce')

# Aggregate daily transactions per product
df_agg = df_raw.groupby(['KODE', 'TANGGAL']).agg({
    'QTY_MSK': 'sum',
    'NILAI_MSK': 'sum'
}).reset_index()

# Focus on incoming quantity (purchases/restocking) - remove zero
df_agg = df_agg[df_agg['QTY_MSK'] > 0].copy()

# Remove extreme outliers (top 1% - likely data entry errors)
q99 = df_agg['QTY_MSK'].quantile(0.99)
df_agg = df_agg[df_agg['QTY_MSK'] <= q99].copy()

print(f"✅ Preprocessed to {len(df_agg):,} daily purchase records (removed extreme outliers & zero)")

# ============================================================================
# 3. FEATURE ENGINEERING (NATURAL FEATURES - NO ARTIFICIAL NOISE)
# ============================================================================
print("\n[3/7] 🔨 Feature engineering...")

df_agg = df_agg.sort_values(['KODE', 'TANGGAL']).reset_index(drop=True)

# ============== TEMPORAL FEATURES ==============
df_agg['day'] = df_agg['TANGGAL'].dt.day
df_agg['month'] = df_agg['TANGGAL'].dt.month
df_agg['day_of_week'] = df_agg['TANGGAL'].dt.dayofweek
df_agg['week_of_year'] = df_agg['TANGGAL'].dt.isocalendar().week
df_agg['is_weekend'] = (df_agg['day_of_week'] >= 5).astype(int)
df_agg['quarter'] = df_agg['TANGGAL'].dt.quarter
df_agg['is_month_start'] = (df_agg['day'] <= 5).astype(int)
df_agg['is_month_end'] = (df_agg['day'] >= 25).astype(int)
df_agg['day_of_year'] = df_agg['TANGGAL'].dt.dayofyear

# ============== LAG FEATURES (Historical demand) ==============
for lag in [1, 2, 3, 7, 14, 21, 28]:
    df_agg[f'qty_lag_{lag}'] = df_agg.groupby('KODE')['QTY_MSK'].shift(lag)

# ============== ROLLING STATISTICS (Trend & Volatility) ==============
for window in [3, 7, 14, 21, 30]:
    df_agg[f'qty_roll_mean_{window}'] = df_agg.groupby('KODE')['QTY_MSK'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    df_agg[f'qty_roll_std_{window}'] = df_agg.groupby('KODE')['QTY_MSK'].transform(
        lambda x: x.rolling(window=window, min_periods=1).std()
    )
    df_agg[f'qty_roll_max_{window}'] = df_agg.groupby('KODE')['QTY_MSK'].transform(
        lambda x: x.rolling(window=window, min_periods=1).max()
    )
    df_agg[f'qty_roll_min_{window}'] = df_agg.groupby('KODE')['QTY_MSK'].transform(
        lambda x: x.rolling(window=window, min_periods=1).min()
    )

# ============== EXPONENTIAL WEIGHTED MOVING AVERAGE ==============
for span in [3, 7, 14]:
    df_agg[f'qty_ewma_{span}'] = df_agg.groupby('KODE')['QTY_MSK'].transform(
        lambda x: x.ewm(span=span, adjust=False).mean()
    )

# ============== CHANGE FEATURES (Growth rate) ==============
for period in [1, 3, 7]:
    df_agg[f'qty_change_{period}'] = df_agg.groupby('KODE')['QTY_MSK'].pct_change(periods=period)

# ============== COEFFICIENT OF VARIATION (Relative volatility) ==============
for window in [7, 14, 30]:
    mean = df_agg.groupby('KODE')['QTY_MSK'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    std = df_agg.groupby('KODE')['QTY_MSK'].transform(
        lambda x: x.rolling(window=window, min_periods=1).std()
    )
    df_agg[f'qty_cv_{window}'] = std / (mean + 1e-8)

# ============== TRANSACTION VALUE FEATURES ==============
df_agg['nilai_per_unit'] = df_agg['NILAI_MSK'] / (df_agg['QTY_MSK'] + 1)

# Drop NaN (from lag/rolling features at the beginning)
df_agg = df_agg.dropna()

# ============== TARGET: DEMAND CLASSIFICATION ==============
# Binary classification: High demand (1) vs Low demand (0)
median_demand = df_agg['QTY_MSK'].median()
df_agg['demand_class'] = (df_agg['QTY_MSK'] > median_demand).astype(int)

# Select features
feature_cols = [col for col in df_agg.columns if col not in 
                ['KODE', 'TANGGAL', 'QTY_MSK', 'NILAI_MSK', 'demand_class']]

X = df_agg[feature_cols].copy()
y = df_agg['demand_class'].copy()

print(f"✅ Created {len(feature_cols)} natural features from {len(df_agg):,} samples")
print(f"   Feature categories:")
print(f"   - Temporal: 9 features (day, month, dow, week, weekend, quarter, month edges, doy)")
print(f"   - Lag: 7 features (1, 2, 3, 7, 14, 21, 28 days)")
print(f"   - Rolling stats: 20 features (mean, std, max, min for windows 3, 7, 14, 21, 30)")
print(f"   - EWMA: 3 features (span 3, 7, 14)")
print(f"   - Changes: 3 features (growth rate 1, 3, 7)")
print(f"   - CV: 3 features (relative volatility 7, 14, 30)")
print(f"   - Transaction value: 1 feature (price per unit)")
print(f"   Total: {len(feature_cols)} features")
print(f"\n   Target distribution: {dict(y.value_counts().sort_index())}")

# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================
print("\n[4/7] 📊 Train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"   Test set: {len(X_test) / len(X) * 100:.1f}% of data")

# ============================================================================
# 5. BASELINE - NO FEATURE SELECTION (ALL FEATURES)
# ============================================================================
print("\n[5/7] 📍 BASELINE - No Feature Selection (All features)")
print("-" * 80)

baseline_results = {}

# Decision Tree Classifier
print("\n🌲 Decision Tree Classifier...")
dt_clf = DecisionTreeClassifier(max_depth=8, min_samples_split=50, random_state=42)
dt_clf.fit(X_train_scaled, y_train)
dt_pred = dt_clf.predict(X_test_scaled)
baseline_results['Decision Tree'] = {
    'model_type': 'Classification',
    'accuracy': accuracy_score(y_test, dt_pred),
    'f1': f1_score(y_test, dt_pred, average='weighted'),
    'predictions': dt_pred
}
print(f"   Accuracy: {baseline_results['Decision Tree']['accuracy']:.4f}")
print(f"   F1-Score: {baseline_results['Decision Tree']['f1']:.4f}")

# Naive Bayes
print("\n📊 Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
nb_pred = nb.predict(X_test_scaled)
baseline_results['Naive Bayes'] = {
    'model_type': 'Classification',
    'accuracy': accuracy_score(y_test, nb_pred),
    'f1': f1_score(y_test, nb_pred, average='weighted'),
    'predictions': nb_pred
}
print(f"   Accuracy: {baseline_results['Naive Bayes']['accuracy']:.4f}")
print(f"   F1-Score: {baseline_results['Naive Bayes']['f1']:.4f}")

# Logistic Regression
print("\n📈 Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
baseline_results['Logistic Regression'] = {
    'model_type': 'Classification',
    'accuracy': accuracy_score(y_test, lr_pred),
    'f1': f1_score(y_test, lr_pred, average='weighted'),
    'predictions': lr_pred
}
print(f"   Accuracy: {baseline_results['Logistic Regression']['accuracy']:.4f}")
print(f"   F1-Score: {baseline_results['Logistic Regression']['f1']:.4f}")

print(f"\n✅ Baseline (All {len(feature_cols)} features) completed")

# ============================================================================
# 6. RFECV - FEATURE SELECTION PREPROCESSING
# ============================================================================
print("\n[6/7] 🔄 RFECV - Feature Selection Preprocessing")
print("-" * 80)
print("Applying RFECV to select optimal features for demand prediction...")

# RFECV with Decision Tree estimator
rfecv = RFECV(
    estimator=DecisionTreeClassifier(max_depth=8, min_samples_split=50, random_state=42),
    step=1,
    cv=5,
    scoring='f1_weighted',
    min_features_to_select=5,
    n_jobs=-1
)

import time
start = time.time()
rfecv.fit(X_train_scaled, y_train)
rfecv_time = time.time() - start

# Get selected features and rankings
selected_features = [feature_cols[i] for i in range(len(feature_cols)) if rfecv.support_[i]]
selected_indices = [i for i in range(len(feature_cols)) if rfecv.support_[i]]

# Get feature rankings (1 = selected, higher = eliminated earlier)
feature_rankings = {}
for i, feat in enumerate(feature_cols):
    feature_rankings[feat] = rfecv.ranking_[i]

# Sort all features by ranking (best first)
all_features_ranked = sorted(feature_rankings.items(), key=lambda x: x[1])

# Transform data
X_train_selected = X_train_scaled[:, selected_indices]
X_test_selected = X_test_scaled[:, selected_indices]

reduction = (1 - len(selected_features) / len(feature_cols)) * 100

print(f"✅ RFECV completed in {rfecv_time:.2f}s")
print(f"   Features selected: {len(selected_features)}/{len(feature_cols)} ({len(selected_features)/len(feature_cols)*100:.1f}%)")
print(f"   Feature reduction: {reduction:.1f}%")
print(f"   Optimal CV score: {rfecv.cv_results_['mean_test_score'].max():.4f}")

print(f"\n   📊 Top 10 Features by RFECV Ranking:")
for i, (feat, rank) in enumerate(all_features_ranked[:10], 1):
    status = "✅ SELECTED" if rank == 1 else f"❌ Eliminated (rank {rank})"
    print(f"   {i:2d}. {feat:30s} | Rank: {rank:2d} | {status}")

# ============================================================================
# 7. AFTER RFECV - WITH FEATURE SELECTION
# ============================================================================
print("\n[7/7] ✅ AFTER RFECV - With Feature Selection")
print("-" * 80)

rfecv_results = {}

# Decision Tree Classifier
print("\n🌲 Decision Tree Classifier...")
dt_clf_rfecv = DecisionTreeClassifier(max_depth=8, min_samples_split=50, random_state=42)
dt_clf_rfecv.fit(X_train_selected, y_train)
dt_pred_rfecv = dt_clf_rfecv.predict(X_test_selected)
rfecv_results['Decision Tree'] = {
    'model_type': 'Classification',
    'accuracy': accuracy_score(y_test, dt_pred_rfecv),
    'f1': f1_score(y_test, dt_pred_rfecv, average='weighted'),
    'predictions': dt_pred_rfecv
}
print(f"   Accuracy: {rfecv_results['Decision Tree']['accuracy']:.4f}")
print(f"   F1-Score: {rfecv_results['Decision Tree']['f1']:.4f}")

# Naive Bayes
print("\n📊 Naive Bayes...")
nb_rfecv = GaussianNB()
nb_rfecv.fit(X_train_selected, y_train)
nb_pred_rfecv = nb_rfecv.predict(X_test_selected)
rfecv_results['Naive Bayes'] = {
    'model_type': 'Classification',
    'accuracy': accuracy_score(y_test, nb_pred_rfecv),
    'f1': f1_score(y_test, nb_pred_rfecv, average='weighted'),
    'predictions': nb_pred_rfecv
}
print(f"   Accuracy: {rfecv_results['Naive Bayes']['accuracy']:.4f}")
print(f"   F1-Score: {rfecv_results['Naive Bayes']['f1']:.4f}")

# Logistic Regression
print("\n📈 Logistic Regression...")
lr_rfecv = LogisticRegression(max_iter=1000, random_state=42)
lr_rfecv.fit(X_train_selected, y_train)
lr_pred_rfecv = lr_rfecv.predict(X_test_selected)
rfecv_results['Logistic Regression'] = {
    'model_type': 'Classification',
    'accuracy': accuracy_score(y_test, lr_pred_rfecv),
    'f1': f1_score(y_test, lr_pred_rfecv, average='weighted'),
    'predictions': lr_pred_rfecv
}
print(f"   Accuracy: {rfecv_results['Logistic Regression']['accuracy']:.4f}")
print(f"   F1-Score: {rfecv_results['Logistic Regression']['f1']:.4f}")

print(f"\n✅ RFECV validation ({len(selected_features)} features) completed")

# ============================================================================
# 8. COMPARATIVE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("📊 COMPARATIVE ANALYSIS: BEFORE vs AFTER RFECV")
print("=" * 80)

comparison_data = []
for model_name in baseline_results.keys():
    baseline_f1 = baseline_results[model_name]['f1']
    rfecv_f1 = rfecv_results[model_name]['f1']
    
    comparison_data.append({
        'Model': model_name,
        'Preprocessing': 'BEFORE (No RFECV)',
        'Features': len(feature_cols),
        'F1-Score': baseline_f1
    })
    comparison_data.append({
        'Model': model_name,
        'Preprocessing': 'AFTER (RFECV)',
        'Features': len(selected_features),
        'F1-Score': rfecv_f1
    })

df_comparison = pd.DataFrame(comparison_data)
print("\n", df_comparison.to_string(index=False))

# ============================================================================
# 9. IMPROVEMENT ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("📈 IMPROVEMENT ANALYSIS")
print("=" * 80)

improvements = []
p_values = []

for model_name in baseline_results.keys():
    baseline_f1 = baseline_results[model_name]['f1']
    rfecv_f1 = rfecv_results[model_name]['f1']
    improvement = ((rfecv_f1 - baseline_f1) / baseline_f1) * 100
    
    # Paired t-test
    baseline_preds = baseline_results[model_name]['predictions']
    rfecv_preds = rfecv_results[model_name]['predictions']
    
    baseline_correct = (baseline_preds == y_test).astype(int)
    rfecv_correct = (rfecv_preds == y_test).astype(int)
    
    t_stat, p_val = ttest_rel(rfecv_correct, baseline_correct)
    
    improvements.append(improvement)
    p_values.append(p_val)
    
    status = "✅ IMPROVED" if improvement > 0 else "❌ DEGRADED" if improvement < -0.1 else "≈ SIMILAR"
    
    print(f"\n{model_name}:")
    print(f"  Improvement: {improvement:+.2f}% {status}")
    print(f"  BEFORE: F1 = {baseline_f1:.4f}")
    print(f"  AFTER:  F1 = {rfecv_f1:.4f}")
    print(f"  p-value: {p_val:.4f} {'✅ Significant (p<0.05)' if p_val < 0.05 else '⚠️ Not significant'}")

avg_improvement = np.mean(improvements)

# ============================================================================
# 10. VERDICT
# ============================================================================
print("\n" + "=" * 80)
print("🎯 CONCLUSION")
print("=" * 80)

significant_count = sum(1 for p in p_values if p < 0.05)

print(f"\nRFECV Preprocessing Summary:")
print(f"  Feature Reduction: {reduction:.1f}% ({len(feature_cols)} → {len(selected_features)} features)")
print(f"  Average Performance Change: {avg_improvement:+.2f}%")
print(f"  Statistically Significant Models: {significant_count}/{len(baseline_results)}")

if avg_improvement > 2 and significant_count >= 2:
    verdict = "✅ HIGHLY EFFECTIVE"
    recommendation = "USE RFECV"
elif avg_improvement > 0 and significant_count >= 1:
    verdict = "✅ EFFECTIVE"
    recommendation = "USE RFECV"
elif abs(avg_improvement) <= 2:
    verdict = "≈ NEUTRAL"
    recommendation = "OPTIONAL (minimal impact)"
else:
    verdict = "❌ NOT EFFECTIVE"
    recommendation = "SKIP RFECV"

print(f"\n🏆 Overall Verdict: RFECV preprocessing is {verdict}")
print(f"   Recommendation: {recommendation} for this dataset")

# ============================================================================
# 11. SAVE RESULTS
# ============================================================================
output_dir = script_dir / "outputs"
output_dir.mkdir(exist_ok=True)

# Save comparison
df_comparison.to_csv(output_dir / "dataset1_comparison.csv", index=False)

# Save ALL features with rankings and interpretation
all_feature_data = []
for feat, rank in all_features_ranked:
    if 'lag' in feat:
        category = 'Lag (Historical)'
        interpretation = f"Past demand {feat.split('_')[-1]} days ago"
    elif 'roll_mean' in feat:
        category = 'Rolling Average'
        interpretation = f"Average demand over {feat.split('_')[-1]} days"
    elif 'roll_std' in feat:
        category = 'Rolling Volatility'
        interpretation = f"Demand volatility over {feat.split('_')[-1]} days"
    elif 'roll_max' in feat or 'roll_min' in feat:
        category = 'Rolling Extremes'
        interpretation = f"Peak/trough demand over {feat.split('_')[-1]} days"
    elif 'ewma' in feat:
        category = 'Exponential MA'
        interpretation = f"Weighted average (span {feat.split('_')[-1]})"
    elif 'change' in feat:
        category = 'Growth Rate'
        interpretation = f"Demand change over {feat.split('_')[-1]} periods"
    elif 'cv' in feat:
        category = 'Coefficient of Variation'
        interpretation = f"Relative volatility over {feat.split('_')[-1]} days"
    elif feat in ['day', 'month', 'day_of_week', 'week_of_year', 'quarter', 'day_of_year']:
        category = 'Temporal'
        interpretation = f"Time component: {feat.replace('_', ' ')}"
    elif feat in ['is_weekend', 'is_month_start', 'is_month_end']:
        category = 'Temporal Flag'
        interpretation = f"Binary indicator: {feat.replace('_', ' ')}"
    elif 'nilai' in feat:
        category = 'Transaction Value'
        interpretation = "Price per unit"
    else:
        category = 'Other'
        interpretation = feat
    
    all_feature_data.append({
        'Feature': feat,
        'RFECV_Ranking': rank,
        'Selected': 'YES' if rank == 1 else 'NO',
        'Category': category,
        'Interpretation': interpretation
    })

df_all_features = pd.DataFrame(all_feature_data)
df_all_features.to_csv(output_dir / "dataset1_all_feature_scores.csv", index=False)

# Save only selected features (for backward compatibility)
feature_interpretation = []
for feat in selected_features:
    if 'lag' in feat:
        category = 'Lag (Historical)'
        interpretation = f"Past demand {feat.split('_')[-1]} days ago"
    elif 'roll_mean' in feat:
        category = 'Rolling Average'
        interpretation = f"Average demand over {feat.split('_')[-1]} days"
    elif 'roll_std' in feat:
        category = 'Rolling Volatility'
        interpretation = f"Demand volatility over {feat.split('_')[-1]} days"
    elif 'roll_max' in feat or 'roll_min' in feat:
        category = 'Rolling Extremes'
        interpretation = f"Peak/trough demand over {feat.split('_')[-1]} days"
    elif 'ewma' in feat:
        category = 'Exponential MA'
        interpretation = f"Weighted average (span {feat.split('_')[-1]})"
    elif 'change' in feat:
        category = 'Growth Rate'
        interpretation = f"Demand change over {feat.split('_')[-1]} periods"
    elif 'cv' in feat:
        category = 'Coefficient of Variation'
        interpretation = f"Relative volatility over {feat.split('_')[-1]} days"
    elif feat in ['day', 'month', 'day_of_week', 'week_of_year', 'quarter', 'day_of_year']:
        category = 'Temporal'
        interpretation = f"Time component: {feat.replace('_', ' ')}"
    elif feat in ['is_weekend', 'is_month_start', 'is_month_end']:
        category = 'Temporal Flag'
        interpretation = f"Binary indicator: {feat.replace('_', ' ')}"
    elif 'nilai' in feat:
        category = 'Transaction Value'
        interpretation = "Price per unit"
    else:
        category = 'Other'
        interpretation = feat
    
    feature_interpretation.append({
        'Feature': feat,
        'Category': category,
        'Interpretation': interpretation
    })

df_features = pd.DataFrame(feature_interpretation)
df_features.to_csv(output_dir / "dataset1_selected_features.csv", index=False)

# ============================================================================
# 12. VISUALIZATION
# ============================================================================
print("\n📊 Generating visualizations...")

# Figure 1: Comprehensive Analysis (4 panels)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Dataset 1: Pharmacy Demand Prediction - RFECV Analysis', fontsize=16, fontweight='bold')

# Panel 1: F1-Score Comparison
ax1 = axes[0, 0]
models = list(baseline_results.keys())
baseline_scores = [baseline_results[m]['f1'] for m in models]
rfecv_scores = [rfecv_results[m]['f1'] for m in models]

x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x - width/2, baseline_scores, width, label=f'BEFORE ({len(feature_cols)} feat)', 
                color='coral', alpha=0.8)
bars2 = ax1.bar(x + width/2, rfecv_scores, width, label=f'AFTER ({len(selected_features)} feat)', 
                color='mediumseagreen', alpha=0.8)

ax1.set_xlabel('Model', fontweight='bold')
ax1.set_ylabel('F1-Score', fontweight='bold')
ax1.set_title('F1-Score Comparison: BEFORE vs AFTER RFECV', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Panel 2: Improvement Percentage
ax2 = axes[0, 1]
colors = ['green' if imp > 0 else 'red' for imp in improvements]
bars = ax2.barh(models, improvements, color=colors, alpha=0.7)
ax2.set_xlabel('Improvement (%)', fontweight='bold')
ax2.set_title('Performance Change After RFECV', fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax2.grid(axis='x', alpha=0.3)

# Semua label di kanan bar (positive position)
for i, (bar, imp) in enumerate(zip(bars, improvements)):
    x_pos = max(imp, 0) + 0.5  # Selalu di kanan, minimum di x=0
    ax2.text(x_pos, i, f'{imp:+.2f}%', 
            va='center', ha='left', fontweight='bold', fontsize=10)

# Panel 3: Feature Reduction
ax3 = axes[1, 0]
sizes = [len(selected_features), len(feature_cols) - len(selected_features)]
labels = [f'SELECTED\n{len(selected_features)} features', 
          f'REMOVED\n{len(feature_cols) - len(selected_features)} features']
colors_pie = ['mediumseagreen', 'lightcoral']
explode = (0.05, 0)

wedges, texts, autotexts = ax3.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                     autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
ax3.set_title(f'Feature Reduction: {reduction:.1f}%\n({len(feature_cols)} → {len(selected_features)} features)', 
              fontweight='bold')

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# Panel 4: Statistical Significance
ax4 = axes[1, 1]
colors_sig = ['green' if p < 0.05 else 'orange' for p in p_values]
bars = ax4.bar(models, p_values, color=colors_sig, alpha=0.7)
ax4.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
ax4.set_xlabel('Model', fontweight='bold')
ax4.set_ylabel('p-value', fontweight='bold')
ax4.set_title('Statistical Significance (Paired T-Test)', fontweight='bold')
ax4.set_xticklabels(models, rotation=15, ha='right')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

for bar, p in zip(bars, p_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{p:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "dataset1_analysis.png", dpi=300, bbox_inches='tight')
print("   Saved: dataset1_analysis.png")

# Figure 2: Selected Features by Category
fig2, ax = plt.subplots(figsize=(12, max(8, len(selected_features) * 0.4)))

categories = df_features['Category'].values
colors_map = {
    'Lag (Historical)': 'steelblue',
    'Rolling Average': 'seagreen',
    'Rolling Volatility': 'coral',
    'Rolling Extremes': 'gold',
    'Exponential MA': 'mediumpurple',
    'Growth Rate': 'crimson',
    'Coefficient of Variation': 'darkorange',
    'Temporal': 'teal',
    'Temporal Flag': 'darkturquoise',
    'Transaction Value': 'darkviolet',
    'Other': 'gray'
}
colors = [colors_map.get(cat, 'gray') for cat in categories]

y_pos = np.arange(len(selected_features))
bars = ax.barh(y_pos, [1] * len(selected_features), color=colors, alpha=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(selected_features, fontsize=10)
ax.set_xlabel('Selected for Demand Prediction', fontweight='bold')
ax.set_title(f'Selected Features by Category ({len(selected_features)} features)', 
             fontsize=14, fontweight='bold')
ax.set_xlim(0, 1.2)
ax.set_xticks([])

# Add category labels
for i, (feat, cat) in enumerate(zip(selected_features, categories)):
    ax.text(1.02, i, cat, va='center', fontsize=9, style='italic', color=colors[i])

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, label=cat, alpha=0.8) 
                   for cat, color in colors_map.items() if cat in categories]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "dataset1_features.png", dpi=300, bbox_inches='tight')
print("   Saved: dataset1_features.png")

# Figure 3: Feature Ranking Visualization (Top 20 features)
fig3, (ax_top, ax_cv) = plt.subplots(1, 2, figsize=(16, 8))
fig3.suptitle('Dataset 1: RFECV Feature Ranking & CV Performance', fontsize=16, fontweight='bold')

# Left panel: Top 20 features by ranking
top_n = min(20, len(all_features_ranked))
top_features = all_features_ranked[:top_n]
feat_names = [f[:25] for f, _ in top_features]  # Truncate long names
rankings = [r for _, r in top_features]
colors_rank = ['mediumseagreen' if r == 1 else 'lightcoral' for r in rankings]

y_pos = np.arange(len(feat_names))
bars = ax_top.barh(y_pos, [1/r if r <= 10 else 0.05 for r in rankings], color=colors_rank, alpha=0.8)
ax_top.set_yticks(y_pos)
ax_top.set_yticklabels(feat_names, fontsize=9)
ax_top.set_xlabel('Importance Score (1/Ranking)', fontweight='bold')
ax_top.set_title(f'Top {top_n} Features by RFECV Ranking', fontweight='bold')
ax_top.invert_yaxis()

# Add ranking numbers
for i, (bar, rank) in enumerate(zip(bars, rankings)):
    label = f"Rank {rank}" if rank > 1 else "✅ Selected"
    ax_top.text(bar.get_width() + 0.01, i, label, va='center', fontsize=8, fontweight='bold')

# Right panel: CV scores across feature elimination steps
n_features = range(rfecv.min_features_to_select, len(feature_cols) + 1, rfecv.step)
cv_scores = rfecv.cv_results_['mean_test_score']
cv_std = rfecv.cv_results_['std_test_score']

ax_cv.plot(n_features, cv_scores, 'o-', color='steelblue', linewidth=2, markersize=6, label='Mean CV Score')
ax_cv.fill_between(n_features, 
                    cv_scores - cv_std, 
                    cv_scores + cv_std, 
                    alpha=0.2, color='steelblue', label='± 1 Std Dev')
ax_cv.axvline(x=len(selected_features), color='green', linestyle='--', linewidth=2, 
              label=f'Optimal: {len(selected_features)} features')
ax_cv.set_xlabel('Number of Features', fontweight='bold')
ax_cv.set_ylabel('Cross-Validation F1-Score', fontweight='bold')
ax_cv.set_title('RFECV Cross-Validation Performance', fontweight='bold')
ax_cv.legend(fontsize=9)
ax_cv.grid(alpha=0.3)

# Mark optimal point
optimal_score = cv_scores[len(selected_features) - rfecv.min_features_to_select]
ax_cv.plot(len(selected_features), optimal_score, 'g*', markersize=20, 
          label=f'Optimal: {optimal_score:.4f}', zorder=5)
ax_cv.text(len(selected_features), optimal_score + 0.01, 
          f'{len(selected_features)} feat\n{optimal_score:.4f}', 
          ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "dataset1_ranking.png", dpi=300, bbox_inches='tight')
print("   Saved: dataset1_ranking.png")

print(f"\n💾 Results saved to: {output_dir.absolute()}")
print("\n✅ ANALYSIS COMPLETE!")
