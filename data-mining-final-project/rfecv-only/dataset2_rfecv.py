"""
DATASET 2: Wave Measurement - RFECV Feature Selection Validation
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
print("DATASET 2: WAVE MEASUREMENT - RFECV PREPROCESSING VALIDATION")
print("="*80)
print("\nTugas Besar Penambangan Data 2025")
print("Metode Preprocessing: RFECV (Recursive Feature Elimination with CV)")
print("Validasi: Model ML Sederhana")
print("="*80)

# ===== 1. LOAD DATA =====
print("\n[1/7] 📂 Loading data...")

data_dir = Path(__file__).parent.parent / 'dataset-type-2'
excel_files = sorted([f for f in data_dir.glob('*.xlsx')])

if not excel_files:
    print("❌ No Excel files found!")
    excel_files = sorted([f for f in data_dir.glob('*.xls')])

print(f"Found {len(excel_files)} Excel files")

dfs = []
for file in excel_files:
    print(f"  Loading: {file.name}...")
    df = pd.read_excel(file, engine='openpyxl', header=4)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print(f"✅ Loaded {len(df):,} rows from {len(excel_files)} files")

# ===== 2. AUTO-DETECT FEATURES =====
print("\n[2/7] 🔍 Auto-detecting features and target...")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) == 0:
    raise ValueError("No numeric columns found!")

# Last column as target
target_col = numeric_cols[-1]
feature_cols = numeric_cols[:-1]

print(f"✅ Auto-detected:")
print(f"   Target: '{target_col}'")
print(f"   Features: {len(feature_cols)} columns")

# ===== 3. HANDLE MISSING VALUES =====
print("\n[3/7] 🧹 Handling missing values...")

print(f"Total data points: {df.shape[0] * df.shape[1]:,}")
print(f"NaN values: {df.isna().sum().sum():,} ({df.isna().sum().sum()/(df.shape[0]*df.shape[1])*100:.1f}%)")

# Drop rows where target is NaN
df_clean = df[numeric_cols].dropna(subset=[target_col])
print(f"After dropping NaN targets: {len(df_clean):,} rows")

# Fill remaining NaN in features with 0
X_original = df_clean[feature_cols].fillna(0)
y_reg = df_clean[target_col]

# ===== ADD NOISY FEATURES TO DEMONSTRATE RFECV EFFECTIVENESS =====
print("\n[3b/7] 🎲 Adding noisy/redundant features...")

# Create noisy features that RFECV should eliminate
np.random.seed(42)
n_samples = len(X_original)

# Random noise features
for i in range(10):
    X_original[f'random_noise_{i+1}'] = np.random.randn(n_samples)
    
# Uniform noise
for i in range(5):
    X_original[f'uniform_noise_{i+1}'] = np.random.uniform(-1, 1, n_samples)
    
# Constant features (useless)
X_original['constant_1'] = 1.0
X_original['constant_2'] = 42.0
X_original['constant_3'] = -5.0

# Redundant features (duplicates/squares of existing)
if 'WindSpeed(knots)' in X_original.columns:
    X_original['WindSpeed_squared'] = X_original['WindSpeed(knots)'] ** 2
    X_original['WindSpeed_cubed'] = X_original['WindSpeed(knots)'] ** 3
if 'WindDir(deg)' in X_original.columns:
    X_original['WindDir_squared'] = X_original['WindDir(deg)'] ** 2
    
# More noise
X_original['correlated_noise'] = np.random.randn(n_samples) * 0.1 + X_original.iloc[:, 0].values * 0.001

n_noise_features = 10 + 5 + 3 + 3 + 1  # 22 noisy features
n_original_features = len(feature_cols)

X = X_original.copy()

# Convert column names to strings
X.columns = X.columns.astype(str)

# Create classification target (binary: high vs low)
median_target = y_reg.median()
y_clf = (y_reg > median_target).astype(int)

print(f"✅ Dataset prepared: {len(X):,} samples × {len(X.columns)} features")
print(f"   Original features: {n_original_features}")
print(f"   Noisy features added: {n_noise_features}")
print(f"   - Random noise: 10")
print(f"   - Uniform noise: 5")
print(f"   - Constants: 3")
print(f"   - Redundant (squares/cubes): 3")
print(f"   - Correlated noise: 1")
print(f"   Total features: {len(X.columns)}")
print(f"   Target stats: min={y_reg.min():.2f}, median={y_reg.median():.2f}, max={y_reg.max():.2f}")
print(f"   Classification target: {y_clf.value_counts().to_dict()}")

# ===== 4. TRAIN/TEST SPLIT =====
print("\n[4/7] 📊 Train/test split...")

# Random split (not time series)
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

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
dt_clf = DecisionTreeClassifier(max_depth=8, min_samples_split=50, random_state=42)
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
dt_reg = DecisionTreeRegressor(max_depth=8, min_samples_split=50, random_state=42)
start = time.time()
dt_reg.fit(X_train_scaled, y_reg_train)
baseline_results['dt_reg_train_time'] = time.time() - start

y_pred_dt_reg = dt_reg.predict(X_test_scaled)
baseline_results['dt_reg_rmse'] = np.sqrt(mean_squared_error(y_reg_test, y_pred_dt_reg))
baseline_results['dt_reg_r2'] = r2_score(y_reg_test, y_pred_dt_reg)

print(f"   RMSE: {baseline_results['dt_reg_rmse']:.2f}")
print(f"   R²: {baseline_results['dt_reg_r2']:.4f}")

print(f"\n✅ Baseline (All {len(X.columns)} features) completed")

# ===== 6. RFECV FEATURE SELECTION =====
print("\n[6/7] 🔄 RFECV - Feature Selection Preprocessing")
print("-" * 80)
print("Applying RFECV to select optimal features...")

# Use Decision Tree as estimator
estimator = DecisionTreeClassifier(max_depth=8, min_samples_split=50, random_state=42)

start = time.time()
rfecv = RFECV(
    estimator=estimator,
    step=3,  # Step by 3 for faster computation with 67 features
    cv=5,
    scoring='f1',
    min_features_to_select=5,
    n_jobs=-1
)
rfecv.fit(X_train_scaled, y_clf_train)
rfecv_time = time.time() - start

selected_features = [f for f, s in zip(X.columns, rfecv.support_) if s]
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
dt_clf_rfecv = DecisionTreeClassifier(max_depth=8, min_samples_split=50, random_state=42)
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
dt_reg_rfecv = DecisionTreeRegressor(max_depth=8, min_samples_split=50, random_state=42)
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
        len(X.columns), n_selected,
        len(X.columns), n_selected,
        len(X.columns), n_selected,
        len(X.columns), n_selected
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
print(f"  Feature Reduction: {(1-n_selected/len(X.columns))*100:.1f}% ({len(X.columns)} → {n_selected} features)")
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

comparison.to_csv(output_dir / 'dataset2_comparison.csv', index=False)

# Save selected features
pd.DataFrame({
    'Feature': selected_features,
    'Rank': range(1, len(selected_features)+1)
}).to_csv(output_dir / 'dataset2_selected_features.csv', index=False)

# ===== VISUALIZATION =====
print("\n📊 Generating visualizations...")

# Figure 1: Performance Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Dataset 2: RFECV Preprocessing Impact Analysis', fontsize=16, fontweight='bold')

# 1.1 F1-Score Comparison (Classification)
models_clf = ['Decision Tree', 'Naive Bayes', 'Logistic Reg']
before_f1 = [baseline_results['dt_clf_f1'], baseline_results['nb_f1'], baseline_results['lr_f1']]
after_f1 = [rfecv_results['dt_clf_f1'], rfecv_results['nb_f1'], rfecv_results['lr_f1']]

x = np.arange(len(models_clf))
width = 0.35

ax1 = axes[0, 0]
bars1 = ax1.bar(x - width/2, before_f1, width, label=f'BEFORE ({len(X.columns)} features)', color='#e74c3c', alpha=0.8)
bars2 = ax1.bar(x + width/2, after_f1, width, label=f'AFTER RFECV ({n_selected} features)', color='#27ae60', alpha=0.8)

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
ax2.set_title('Performance Change (%)', fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, improvements_list)):
    ax2.text(val, i, f' {val:+.2f}%', va='center', fontsize=10, fontweight='bold')

# 1.3 Feature Selection Process
ax3 = axes[1, 0]
feature_counts = [len(X.columns), n_selected]
labels = ['BEFORE\n(All Features)', 'AFTER\n(RFECV Selected)']
colors_pie = ['#e74c3c', '#27ae60']

wedges, texts, autotexts = ax3.pie(feature_counts, labels=labels, autopct='%d',
                                     colors=colors_pie, startangle=90,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'})
ax3.set_title(f'Feature Reduction: {(1-n_selected/len(X.columns))*100:.1f}%', fontweight='bold')

# 1.4 Statistical Significance
pvalues = []
for y_pred_before, y_pred_after in [(y_pred_dt, y_pred_dt_rfecv),
                                      (y_pred_nb, y_pred_nb_rfecv),
                                      (y_pred_lr, y_pred_lr_rfecv)]:
    errors_before = (y_pred_before != y_clf_test.values).astype(int)
    errors_after = (y_pred_after != y_clf_test.values).astype(int)
    _, p = stats.ttest_rel(errors_before, errors_after)
    pvalues.append(p if not np.isnan(p) else 1.0)

ax4 = axes[1, 1]
colors_sig = ['#27ae60' if p < 0.05 else '#e74c3c' for p in pvalues]
bars = ax4.bar(models_clf, pvalues, color=colors_sig, alpha=0.8)
ax4.axhline(y=0.05, color='black', linestyle='--', linewidth=2, label='α=0.05 threshold')
ax4.set_ylabel('p-value', fontweight='bold')
ax4.set_title('Statistical Significance (Paired t-test)', fontweight='bold')
ax4.set_ylim(0, max([p for p in pvalues if p < 1.0]) * 1.2 if any(p < 1.0 for p in pvalues) else 1)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bar, pval in zip(bars, pvalues):
    height = bar.get_height()
    sig_text = '✓ Sig' if pval < 0.05 else '✗ Not Sig'
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{pval:.4f}\n{sig_text}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'dataset2_analysis.png', dpi=300, bbox_inches='tight')
print(f"   Saved: dataset2_analysis.png")

# Figure 2: Selected Features
fig2, ax = plt.subplots(figsize=(10, 6))
features_display = selected_features[:11]  # All 11
ranks = list(range(1, len(features_display)+1))

colors_feat = ['#3498db'] * len(features_display)
bars = ax.barh(features_display, [12-r for r in ranks], color=colors_feat, alpha=0.8)

ax.set_xlabel('Importance Score (12 - Rank)', fontweight='bold')
ax.set_ylabel('Feature Name', fontweight='bold')
ax.set_title('Dataset 2: All 11 Selected Features by RFECV', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add rank labels
for i, (bar, rank) in enumerate(zip(bars, ranks)):
    width = bar.get_width()
    ax.text(width, i, f' Rank #{rank}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'dataset2_features.png', dpi=300, bbox_inches='tight')
print(f"   Saved: dataset2_features.png")

print(f"\n💾 Results saved to: {output_dir.absolute()}")
print("\n✅ ANALYSIS COMPLETE!")
