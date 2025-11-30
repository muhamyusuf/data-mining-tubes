"""
================================================================================
DATASET 2: COMPARISON OF FEATURE SELECTION METHODS
================================================================================
Tugas Besar Penambangan Data 2025

OBJECTIVE:
    Membandingkan 4 metode feature selection untuk prediksi tinggi gelombang:
    1. Information Gain (Filter Method)
    2. Chi-Square (Filter Method)
    3. Pearson Correlation (Filter Method)
    4. RFECV (Wrapper Method)
    
VALIDATION:
    Menggunakan Decision Tree, Naive Bayes, dan Logistic Regression untuk
    mengevaluasi efektivitas setiap metode feature selection.
    
REFERENCE:
    - Jurnal 1: Critical Factor Analysis for Diabetes Prediction
    - Jurnal 2: Ensemble Methods with Feature Selection for Code Smells
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import (
    mutual_info_classif,
    chi2,
    RFECV,
    SelectKBest
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DATASET 2: COMPARISON OF FEATURE SELECTION METHODS")
print("=" * 80)
print("\nMetode yang Dibandingkan:")
print("1. Information Gain (Mutual Information)")
print("2. Chi-Square Test")
print("3. Pearson Correlation")
print("4. RFECV (Recursive Feature Elimination with Cross-Validation)")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================
print("\n[1/6] 📂 Loading and preprocessing data...")

script_dir = Path(__file__).parent
data_path = script_dir.parent / "dataset-type-2"
excel_files = list(data_path.glob("*.xlsx"))

print(f"Found {len(excel_files)} Excel files")

dfs = []
for file in excel_files:
    df = pd.read_excel(file)
    dfs.append(df)

df_raw = pd.concat(dfs, ignore_index=True)

# Extract header and data
new_columns = df_raw.iloc[3].values
df_raw = df_raw.iloc[4:].reset_index(drop=True)
df_raw.columns = new_columns

# Convert to numeric
df_numeric = df_raw.apply(pd.to_numeric, errors='coerce')
df_clean = df_numeric.dropna(axis=1, how='all').copy()

# Detect target
hsig_candidates = [col for col in df_clean.columns if 'sig' in str(col).lower() and 'scale' not in str(col).lower()]
target_col = hsig_candidates[0] if hsig_candidates else [c for c in df_clean.columns if df_clean[c].dtype in [np.float64, np.int64]][0]
target_col = str(target_col)

# Clean data
df_clean = df_clean.dropna(subset=[target_col])

feature_cols_to_fill = [c for c in df_clean.columns if c != target_col]
for col in feature_cols_to_fill:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

df_clean = df_clean.dropna(axis=1, how='any')

# Remove zero-variance
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if col != target_col:
        if df_clean[col].std() == 0:
            df_clean = df_clean.drop(columns=[col])

# Remove outliers (top 1%)
for col in df_clean.select_dtypes(include=[np.number]).columns:
    if col != target_col:
        top_threshold = df_clean[col].quantile(0.99)
        df_clean = df_clean[df_clean[col] <= top_threshold]

print(f"✅ Cleaned to {len(df_clean):,} samples")

# ============================================================================
# 2. FEATURE & TARGET DEFINITION
# ============================================================================
print("\n[2/6] 🎯 Defining features and target...")

df_clean.columns = df_clean.columns.astype(str)

feature_cols = [col for col in df_clean.columns if col != str(target_col)]
X = df_clean[feature_cols].copy()
y_reg = df_clean[str(target_col)].copy()

# Classification target
median_hsig = y_reg.median()
y_clf = (y_reg > median_hsig).astype(int)

print(f"✅ Features: {len(feature_cols)} oceanographic measurements")
print(f"   Target: High wave (>{median_hsig:.2f}m) vs Low wave")
print(f"   Class distribution: {dict(y_clf.value_counts().sort_index())}")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================
print("\n[3/6] 📊 Train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Safety check
X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

print(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,}")

# ============================================================================
# 4. BASELINE - NO FEATURE SELECTION
# ============================================================================
print("\n[4/6] 📍 BASELINE - No Feature Selection...")
print("-" * 80)

def evaluate_models(X_tr, X_te, y_tr, y_te, method_name):
    """Evaluate 3 models and return results"""
    results = {}
    
    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=10, min_samples_split=50, random_state=42)
    dt.fit(X_tr, y_tr)
    dt_pred = dt.predict(X_te)
    results['Decision Tree'] = {
        'accuracy': accuracy_score(y_te, dt_pred),
        'f1': f1_score(y_te, dt_pred, average='weighted')
    }
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_tr, y_tr)
    nb_pred = nb.predict(X_te)
    results['Naive Bayes'] = {
        'accuracy': accuracy_score(y_te, nb_pred),
        'f1': f1_score(y_te, nb_pred, average='weighted')
    }
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr, y_tr)
    lr_pred = lr.predict(X_te)
    results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_te, lr_pred),
        'f1': f1_score(y_te, lr_pred, average='weighted')
    }
    
    # Print results
    print(f"\n{method_name}")
    for model_name, metrics in results.items():
        print(f"  {model_name:20s} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
    
    return results

baseline_results = evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, 
                                   f"BASELINE (All {len(feature_cols)} features)")

# ============================================================================
# 5. FEATURE SELECTION METHODS
# ============================================================================
print("\n[5/6] 🔄 Applying Feature Selection Methods...")
print("-" * 80)

# Jumlah fitur yang akan dipilih (konsisten untuk filter methods)
K_FEATURES = 7

# ===== METHOD 1: INFORMATION GAIN (MUTUAL INFORMATION) =====
print("\n1️⃣  Information Gain (Mutual Information)")
print("   Mengukur dependensi antara fitur dan target...")

mi_selector = SelectKBest(mutual_info_classif, k=K_FEATURES)
mi_selector.fit(X_train_scaled, y_train)

X_train_mi = mi_selector.transform(X_train_scaled)
X_test_mi = mi_selector.transform(X_test_scaled)

selected_features_mi = [feature_cols[i] for i in mi_selector.get_support(indices=True)]
print(f"   ✅ Selected {len(selected_features_mi)} features")
print(f"   Features: {selected_features_mi}")

mi_results = evaluate_models(X_train_mi, X_test_mi, y_train, y_test, 
                             f"   After IG ({K_FEATURES} features)")

# ===== METHOD 2: CHI-SQUARE =====
print("\n2️⃣  Chi-Square Test")
print("   Mengukur independensi antara fitur dan target...")

# Chi-square requires non-negative features
scaler_chi = MinMaxScaler()
X_train_nonneg = scaler_chi.fit_transform(X_train)
X_test_nonneg = scaler_chi.transform(X_test)

chi2_selector = SelectKBest(chi2, k=K_FEATURES)
chi2_selector.fit(X_train_nonneg, y_train)

X_train_chi = chi2_selector.transform(X_train_nonneg)
X_test_chi = chi2_selector.transform(X_test_nonneg)

selected_features_chi = [feature_cols[i] for i in chi2_selector.get_support(indices=True)]
print(f"   ✅ Selected {len(selected_features_chi)} features")
print(f"   Features: {selected_features_chi}")

chi_results = evaluate_models(X_train_chi, X_test_chi, y_train, y_test, 
                              f"   After Chi-Square ({K_FEATURES} features)")

# ===== METHOD 3: PEARSON CORRELATION =====
print("\n3️⃣  Pearson Correlation")
print("   Mengukur korelasi linear fitur-target...")

# Calculate correlation with target
correlations = []
for i, col in enumerate(feature_cols):
    corr = np.corrcoef(X_train_scaled[:, i], y_train)[0, 1]
    correlations.append((i, abs(corr)))

# Sort and select top K
correlations.sort(key=lambda x: x[1], reverse=True)
top_k_indices = [idx for idx, _ in correlations[:K_FEATURES]]

X_train_corr = X_train_scaled[:, top_k_indices]
X_test_corr = X_test_scaled[:, top_k_indices]

selected_features_corr = [feature_cols[i] for i in top_k_indices]
print(f"   ✅ Selected {len(selected_features_corr)} features")
print(f"   Features: {selected_features_corr}")

corr_results = evaluate_models(X_train_corr, X_test_corr, y_train, y_test, 
                               f"   After Pearson Corr ({K_FEATURES} features)")

# ===== METHOD 4: RFECV =====
print("\n4️⃣  RFECV (Recursive Feature Elimination with CV)")
print("   Eliminasi fitur secara rekursif dengan cross-validation...")

rfecv = RFECV(
    estimator=DecisionTreeClassifier(max_depth=10, min_samples_split=50, random_state=42),
    step=3,
    cv=5,
    scoring='f1_weighted',
    min_features_to_select=5,
    n_jobs=-1
)

import time
start = time.time()
rfecv.fit(X_train_scaled, y_train)
rfecv_time = time.time() - start

X_train_rfecv = rfecv.transform(X_train_scaled)
X_test_rfecv = rfecv.transform(X_test_scaled)

selected_features_rfecv = [feature_cols[i] for i in range(len(feature_cols)) if rfecv.support_[i]]
print(f"   ✅ Selected {len(selected_features_rfecv)} features (optimal dari CV)")
print(f"   Features: {selected_features_rfecv}")
print(f"   Time: {rfecv_time:.2f}s")

rfecv_results = evaluate_models(X_train_rfecv, X_test_rfecv, y_train, y_test, 
                                f"   After RFECV ({len(selected_features_rfecv)} features)")

# ============================================================================
# 6. COMPARISON & VISUALIZATION
# ============================================================================
print("\n[6/6] 📊 Creating Comparison Report...")
print("=" * 80)

# Compile results
all_results = []

# Baseline
for model_name, metrics in baseline_results.items():
    all_results.append({
        'Method': 'Baseline (No Selection)',
        'Features': len(feature_cols),
        'Model': model_name,
        'Accuracy': metrics['accuracy'],
        'F1-Score': metrics['f1']
    })

# Information Gain
for model_name, metrics in mi_results.items():
    all_results.append({
        'Method': 'Information Gain',
        'Features': K_FEATURES,
        'Model': model_name,
        'Accuracy': metrics['accuracy'],
        'F1-Score': metrics['f1']
    })

# Chi-Square
for model_name, metrics in chi_results.items():
    all_results.append({
        'Method': 'Chi-Square',
        'Features': K_FEATURES,
        'Model': model_name,
        'Accuracy': metrics['accuracy'],
        'F1-Score': metrics['f1']
    })

# Pearson Correlation
for model_name, metrics in corr_results.items():
    all_results.append({
        'Method': 'Pearson Correlation',
        'Features': K_FEATURES,
        'Model': model_name,
        'Accuracy': metrics['accuracy'],
        'F1-Score': metrics['f1']
    })

# RFECV
for model_name, metrics in rfecv_results.items():
    all_results.append({
        'Method': 'RFECV',
        'Features': len(selected_features_rfecv),
        'Model': model_name,
        'Accuracy': metrics['accuracy'],
        'F1-Score': metrics['f1']
    })

df_results = pd.DataFrame(all_results)

# Save to CSV
output_dir = Path(__file__).parent / 'outputs-comparison'
output_dir.mkdir(exist_ok=True)
df_results.to_csv(output_dir / 'dataset2_feature_selection_comparison.csv', index=False)
print(f"\n✅ Saved: {output_dir / 'dataset2_feature_selection_comparison.csv'}")

# Print summary table
print("\n📋 COMPARISON SUMMARY:")
print("=" * 80)
pivot = df_results.pivot_table(index='Method', columns='Model', values='F1-Score', aggfunc='mean')
print(pivot.to_string())

# Calculate average improvement
print("\n📈 AVERAGE F1-SCORE BY METHOD:")
avg_by_method = df_results.groupby('Method')['F1-Score'].mean().sort_values(ascending=False)
for method, score in avg_by_method.items():
    baseline_avg = df_results[df_results['Method'] == 'Baseline (No Selection)']['F1-Score'].mean()
    improvement = ((score - baseline_avg) / baseline_avg) * 100
    print(f"  {method:25s}: {score:.4f} ({improvement:+.2f}%)")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n📊 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Dataset 2: Perbandingan Metode Feature Selection', fontsize=16, fontweight='bold')

methods = ['Baseline (No Selection)', 'Information Gain', 'Chi-Square', 'Pearson Correlation', 'RFECV']
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

# Plot 1: F1-Score by Method for each Model
ax1 = axes[0, 0]
for i, model in enumerate(['Decision Tree', 'Naive Bayes', 'Logistic Regression']):
    model_data = df_results[df_results['Model'] == model]
    scores = [model_data[model_data['Method'] == m]['F1-Score'].values[0] for m in methods]
    x = np.arange(len(methods))
    ax1.plot(x, scores, marker='o', label=model, linewidth=2, markersize=8)

ax1.set_xlabel('Feature Selection Method', fontweight='bold')
ax1.set_ylabel('F1-Score', fontweight='bold')
ax1.set_title('F1-Score Comparison Across Methods', fontweight='bold')
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods, rotation=45, ha='right')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Average F1-Score by Method
ax2 = axes[0, 1]
avg_scores = [df_results[df_results['Method'] == m]['F1-Score'].mean() for m in methods]
bars = ax2.bar(range(len(methods)), avg_scores, color=colors, alpha=0.8)
ax2.set_xlabel('Feature Selection Method', fontweight='bold')
ax2.set_ylabel('Average F1-Score', fontweight='bold')
ax2.set_title('Average Performance Across All Models', fontweight='bold')
ax2.set_xticks(range(len(methods)))
ax2.set_xticklabels(methods, rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

for bar, score in zip(bars, avg_scores):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

# Plot 3: Number of Features
ax3 = axes[1, 0]
n_features = [len(feature_cols), K_FEATURES, K_FEATURES, K_FEATURES, len(selected_features_rfecv)]
bars = ax3.bar(range(len(methods)), n_features, color=colors, alpha=0.8)
ax3.set_xlabel('Feature Selection Method', fontweight='bold')
ax3.set_ylabel('Number of Selected Features', fontweight='bold')
ax3.set_title('Feature Reduction', fontweight='bold')
ax3.set_xticks(range(len(methods)))
ax3.set_xticklabels(methods, rotation=45, ha='right')
ax3.grid(axis='y', alpha=0.3)

for bar, n_feat in zip(bars, n_features):
    height = bar.get_height()
    reduction = ((len(feature_cols) - n_feat) / len(feature_cols)) * 100 if n_feat < len(feature_cols) else 0
    label = f'{n_feat}\n({reduction:.1f}%)' if reduction > 0 else f'{n_feat}'
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            label, ha='center', va='bottom', fontweight='bold')

# Plot 4: Heatmap of F1-Scores
ax4 = axes[1, 1]
pivot_data = df_results.pivot_table(index='Method', columns='Model', values='F1-Score')
pivot_data = pivot_data.reindex(methods)
sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn', cbar_kws={'label': 'F1-Score'},
            linewidths=0.5, ax=ax4, vmin=0.90, vmax=1.0)
ax4.set_title('F1-Score Heatmap', fontweight='bold')
ax4.set_xlabel('Model', fontweight='bold')
ax4.set_ylabel('Method', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'dataset2_feature_selection_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'dataset2_feature_selection_comparison.png'}")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)
