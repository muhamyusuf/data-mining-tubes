"""
================================================================================
DATASET 2: WAVE HEIGHT PREDICTION - RFECV FEATURE SELECTION
================================================================================
Tugas Besar Penambangan Data 2025

OBJECTIVE: 
    Memprediksi significant wave height (Hsig) berdasarkan meteorological & 
    oceanographic measurements.
    
PREPROCESSING FOCUS:
    RFECV (Recursive Feature Elimination with Cross-Validation) untuk:
    - Mengurangi dimensionality dari banyak sensor measurements
    - Menghilangkan redundant/correlated features (sensor data often highly correlated)
    - Mengidentifikasi most predictive meteorological & oceanographic factors

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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, r2_score, mean_squared_error
from scipy.stats import ttest_rel
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

print("=" * 80)
print("DATASET 2: WAVE HEIGHT PREDICTION - RFECV FEATURE SELECTION")
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
data_path = script_dir.parent / "dataset-type-2"
excel_files = list(data_path.glob("*.xlsx"))

print(f"Found {len(excel_files)} Excel files")

dfs = []
for file in excel_files:
    print(f"  Loading: {file.name}...")
    df = pd.read_excel(file, engine='openpyxl')
    dfs.append(df)

df_raw = pd.concat(dfs, ignore_index=True)
print(f"✅ Loaded {len(df_raw):,} rows from {len(excel_files)} files")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n[2/7] 🔧 Data preprocessing...")

# Extract header from row 3 (0-indexed) and data from row 4 onwards
new_columns = df_raw.iloc[3].values
df_raw = df_raw.iloc[4:].reset_index(drop=True)
df_raw.columns = new_columns

# Convert all columns to numeric
df_numeric = df_raw.apply(pd.to_numeric, errors='coerce')

# Remove columns that are completely NaN
df_clean = df_numeric.dropna(axis=1, how='all').copy()

# Detect target column (Hsig or first numeric column)
hsig_candidates = [col for col in df_clean.columns if 'sig' in str(col).lower() and 'scale' not in str(col).lower()]
target_col = hsig_candidates[0] if hsig_candidates else [c for c in df_clean.columns if df_clean[c].dtype in [np.float64, np.int64]][0]
target_col = str(target_col)  # Ensure target_col is a string

print(f"✅ Detected target column: '{target_col}' (Wave Height)")

# Remove rows with NaN target
df_clean = df_clean.dropna(subset=[target_col])

# Fill remaining NaN in features with column median - simpler approach
feature_cols_to_fill = [c for c in df_clean.columns if c != target_col]
for col in feature_cols_to_fill:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Remove any columns that still have NaN values
df_clean = df_clean.dropna(axis=1, how='any')

# Remove zero-variance features (constant columns)
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if col != target_col:
        if df_clean[col].std() == 0:
            df_clean = df_clean.drop(columns=[col])

# Remove extreme outliers using top quantile method (gentle, like Dataset 1)
# This improves model performance by removing sensor errors/extreme anomalies
for col in df_clean.select_dtypes(include=[np.number]).columns:
    if col != target_col:  # Don't remove target outliers
        top_threshold = df_clean[col].quantile(0.99)  # Remove top 1%
        df_clean = df_clean[df_clean[col] <= top_threshold]

print(f"✅ Cleaned to {len(df_clean):,} samples with {len(df_clean.columns)} numeric features")

# ============================================================================
# 3. FEATURE SELECTION & TARGET DEFINITION
# ============================================================================
print("\n[3/7] 🎯 Defining features and target...")

# Convert column names to strings (some may be numeric)
df_clean.columns = df_clean.columns.astype(str)

# Features: all numeric columns except target
feature_cols = [col for col in df_clean.columns if col != str(target_col)]
X = df_clean[feature_cols].copy()
y_reg = df_clean[str(target_col)].copy()

# Classification target: High wave (1) vs Low wave (0)
median_hsig = y_reg.median()
y_clf = (y_reg > median_hsig).astype(int)

print(f"✅ Features: {len(feature_cols)} oceanographic & meteorological measurements")
print(f"   Target (regression): {target_col} [Range: {y_reg.min():.2f} - {y_reg.max():.2f}m]")
print(f"   Target (classification): High wave (>{median_hsig:.2f}m) vs Low wave")
print(f"   Class distribution: {dict(y_clf.value_counts().sort_index())}")

# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================
print("\n[4/7] 📊 Train/test split...")

X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg = train_test_split(
    X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Final safety check: replace any NaN with 0 (shouldn't happen but just in case)
X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

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
dt_clf = DecisionTreeClassifier(max_depth=10, min_samples_split=50, random_state=42)
dt_clf.fit(X_train_scaled, y_train_clf)
dt_pred = dt_clf.predict(X_test_scaled)
baseline_results['Decision Tree'] = {
    'model_type': 'Classification',
    'accuracy': accuracy_score(y_test_clf, dt_pred),
    'f1': f1_score(y_test_clf, dt_pred, average='weighted'),
    'predictions': dt_pred
}
print(f"   Accuracy: {baseline_results['Decision Tree']['accuracy']:.4f}")
print(f"   F1-Score: {baseline_results['Decision Tree']['f1']:.4f}")

# Naive Bayes
print("\n📊 Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train_scaled, y_train_clf)
nb_pred = nb.predict(X_test_scaled)
baseline_results['Naive Bayes'] = {
    'model_type': 'Classification',
    'accuracy': accuracy_score(y_test_clf, nb_pred),
    'f1': f1_score(y_test_clf, nb_pred, average='weighted'),
    'predictions': nb_pred
}
print(f"   Accuracy: {baseline_results['Naive Bayes']['accuracy']:.4f}")
print(f"   F1-Score: {baseline_results['Naive Bayes']['f1']:.4f}")

# Logistic Regression
print("\n📈 Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train_clf)
lr_pred = lr.predict(X_test_scaled)
baseline_results['Logistic Regression'] = {
    'model_type': 'Classification',
    'accuracy': accuracy_score(y_test_clf, lr_pred),
    'f1': f1_score(y_test_clf, lr_pred, average='weighted'),
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
print("Applying RFECV to select optimal oceanographic/meteorological features...")

# RFECV with Decision Tree estimator
rfecv = RFECV(
    estimator=DecisionTreeClassifier(max_depth=10, min_samples_split=50, random_state=42),
    step=3,  # Faster for many features
    cv=5,
    scoring='f1_weighted',
    min_features_to_select=5,
    n_jobs=-1
)

import time
start = time.time()
rfecv.fit(X_train_scaled, y_train_clf)
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

print(f"\n   📊 Top Features by RFECV Ranking:")
for i, (feat, rank) in enumerate(all_features_ranked[:min(10, len(all_features_ranked))], 1):
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
dt_clf_rfecv = DecisionTreeClassifier(max_depth=10, min_samples_split=50, random_state=42)
dt_clf_rfecv.fit(X_train_selected, y_train_clf)
dt_pred_rfecv = dt_clf_rfecv.predict(X_test_selected)
rfecv_results['Decision Tree'] = {
    'model_type': 'Classification',
    'accuracy': accuracy_score(y_test_clf, dt_pred_rfecv),
    'f1': f1_score(y_test_clf, dt_pred_rfecv, average='weighted'),
    'predictions': dt_pred_rfecv
}
print(f"   Accuracy: {rfecv_results['Decision Tree']['accuracy']:.4f}")
print(f"   F1-Score: {rfecv_results['Decision Tree']['f1']:.4f}")

# Naive Bayes
print("\n📊 Naive Bayes...")
nb_rfecv = GaussianNB()
nb_rfecv.fit(X_train_selected, y_train_clf)
nb_pred_rfecv = nb_rfecv.predict(X_test_selected)
rfecv_results['Naive Bayes'] = {
    'model_type': 'Classification',
    'accuracy': accuracy_score(y_test_clf, nb_pred_rfecv),
    'f1': f1_score(y_test_clf, nb_pred_rfecv, average='weighted'),
    'predictions': nb_pred_rfecv
}
print(f"   Accuracy: {rfecv_results['Naive Bayes']['accuracy']:.4f}")
print(f"   F1-Score: {rfecv_results['Naive Bayes']['f1']:.4f}")

# Logistic Regression
print("\n📈 Logistic Regression...")
lr_rfecv = LogisticRegression(max_iter=1000, random_state=42)
lr_rfecv.fit(X_train_selected, y_train_clf)
lr_pred_rfecv = lr_rfecv.predict(X_test_selected)
rfecv_results['Logistic Regression'] = {
    'model_type': 'Classification',
    'accuracy': accuracy_score(y_test_clf, lr_pred_rfecv),
    'f1': f1_score(y_test_clf, lr_pred_rfecv, average='weighted'),
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
    
    baseline_correct = (baseline_preds == y_test_clf).astype(int)
    rfecv_correct = (rfecv_preds == y_test_clf).astype(int)
    
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
df_comparison.to_csv(output_dir / "dataset2_comparison.csv", index=False)

# Save ALL features with rankings and categorization
all_feature_data = []
for feat, rank in all_features_ranked:
    feat_lower = str(feat).lower()
    
    if any(x in feat_lower for x in ['hsig', 'hmax', 'wave', 'height']):
        category = 'Wave Parameters'
        interpretation = "Wave characteristics"
    elif any(x in feat_lower for x in ['wind', 'gust']):
        category = 'Wind'
        interpretation = "Wind measurements"
    elif any(x in feat_lower for x in ['temp', 'sst']):
        category = 'Temperature'
        interpretation = "Water/air temperature"
    elif any(x in feat_lower for x in ['sal', 'psu']):
        category = 'Salinity'
        interpretation = "Water salinity"
    elif any(x in feat_lower for x in ['dir', 'compass']):
        category = 'Direction'
        interpretation = "Wind/wave direction"
    elif any(x in feat_lower for x in ['period', 'freq']):
        category = 'Wave Frequency'
        interpretation = "Wave oscillation"
    elif any(x in feat_lower for x in ['press', 'atm']):
        category = 'Pressure'
        interpretation = "Atmospheric pressure"
    elif any(x in feat_lower for x in ['current', 'surf']):
        category = 'Oceanographic'
        interpretation = "Ocean current measurements"
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
df_all_features.to_csv(output_dir / "dataset2_all_feature_scores.csv", index=False)

# Save only selected features (for backward compatibility)
feature_categorization = []
for feat in selected_features:
    feat_lower = str(feat).lower()
    
    if any(x in feat_lower for x in ['hsig', 'hmax', 'wave', 'height']):
        category = 'Wave Parameters'
        interpretation = "Wave characteristics"
    elif any(x in feat_lower for x in ['wind', 'gust']):
        category = 'Wind'
        interpretation = "Wind measurements"
    elif any(x in feat_lower for x in ['temp', 'sst']):
        category = 'Temperature'
        interpretation = "Water/air temperature"
    elif any(x in feat_lower for x in ['sal', 'psu']):
        category = 'Salinity'
        interpretation = "Water salinity"
    elif any(x in feat_lower for x in ['dir', 'compass']):
        category = 'Direction'
        interpretation = "Wind/wave direction"
    elif any(x in feat_lower for x in ['period', 'freq']):
        category = 'Wave Frequency'
        interpretation = "Wave oscillation"
    elif any(x in feat_lower for x in ['press', 'atm']):
        category = 'Pressure'
        interpretation = "Atmospheric pressure"
    else:
        category = 'Oceanographic'
        interpretation = "General ocean measurement"
    
    feature_categorization.append({
        'Feature': feat,
        'Category': category,
        'Interpretation': interpretation
    })

df_features = pd.DataFrame(feature_categorization)
df_features.to_csv(output_dir / "dataset2_selected_features.csv", index=False)

# ============================================================================
# 12. VISUALIZATION
# ============================================================================
print("\n📊 Generating visualizations...")

# Figure 1: Comprehensive Analysis (4 panels)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Dataset 2: Wave Height Prediction - RFECV Analysis', fontsize=16, fontweight='bold')

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
for i, (bar, imp, model) in enumerate(zip(bars, improvements, models)):
    x_pos = max(imp, 0) + 0.08  # Selalu di kanan, minimum di x=0
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
plt.savefig(output_dir / "dataset2_analysis.png", dpi=300, bbox_inches='tight')
print("   Saved: dataset2_analysis.png")

# Figure 2: Selected Features by Category
fig2, ax = plt.subplots(figsize=(12, max(8, len(selected_features) * 0.4)))

categories = df_features['Category'].values
colors_map = {
    'Wave Parameters': 'royalblue',
    'Wind': 'skyblue',
    'Temperature': 'coral',
    'Salinity': 'seagreen',
    'Direction': 'gold',
    'Wave Frequency': 'mediumpurple',
    'Pressure': 'crimson',
    'Oceanographic': 'teal',
    'Other': 'gray'
}
colors = [colors_map.get(cat, 'gray') for cat in categories]

y_pos = np.arange(len(selected_features))
bars = ax.barh(y_pos, [1] * len(selected_features), color=colors, alpha=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(selected_features, fontsize=10)
ax.set_xlabel('Selected for Wave Height Prediction', fontweight='bold')
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
plt.savefig(output_dir / "dataset2_features.png", dpi=300, bbox_inches='tight')
print("   Saved: dataset2_features.png")

# Figure 3: Feature Ranking Visualization (All features)
fig3, (ax_rank, ax_cv) = plt.subplots(1, 2, figsize=(16, max(8, len(all_features_ranked) * 0.35)))
fig3.suptitle('Dataset 2: RFECV Feature Ranking & CV Performance', fontsize=16, fontweight='bold')

# Left panel: All features by ranking
feat_names = [f[:25] for f, _ in all_features_ranked]  # Truncate long names
rankings = [r for _, r in all_features_ranked]
colors_rank = ['mediumseagreen' if r == 1 else 'lightcoral' for r in rankings]

y_pos = np.arange(len(feat_names))
bars = ax_rank.barh(y_pos, [1/r if r <= 10 else 0.05 for r in rankings], color=colors_rank, alpha=0.8)
ax_rank.set_yticks(y_pos)
ax_rank.set_yticklabels(feat_names, fontsize=9)
ax_rank.set_xlabel('Importance Score (1/Ranking)', fontweight='bold')
ax_rank.set_title(f'All Features by RFECV Ranking', fontweight='bold')
ax_rank.invert_yaxis()

# Add ranking numbers
for i, (bar, rank) in enumerate(zip(bars, rankings)):
    label = f"Rank {rank}" if rank > 1 else "✅ Selected"
    ax_rank.text(bar.get_width() + 0.01, i, label, va='center', fontsize=8, fontweight='bold')

# Right panel: CV scores across feature elimination steps
step_size = rfecv.step if hasattr(rfecv, 'step') else 1
cv_scores = rfecv.cv_results_['mean_test_score']
cv_std = rfecv.cv_results_['std_test_score']

# Calculate actual n_features based on CV results length
n_features_list = list(range(rfecv.min_features_to_select, 
                              rfecv.min_features_to_select + len(cv_scores) * step_size, 
                              step_size))

ax_cv.plot(n_features_list, cv_scores, 'o-', color='steelblue', linewidth=2, markersize=6, label='Mean CV Score')
ax_cv.fill_between(n_features_list, 
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
optimal_idx = (len(selected_features) - rfecv.min_features_to_select) // step_size
if optimal_idx < len(cv_scores):
    optimal_score = cv_scores[optimal_idx]
    ax_cv.plot(len(selected_features), optimal_score, 'g*', markersize=20, zorder=5)
    ax_cv.text(len(selected_features), optimal_score + 0.005, 
              f'{len(selected_features)} feat\n{optimal_score:.4f}', 
              ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "dataset2_ranking.png", dpi=300, bbox_inches='tight')
print("   Saved: dataset2_ranking.png")

print(f"\n💾 Results saved to: {output_dir.absolute()}")
print("\n✅ ANALYSIS COMPLETE!")
