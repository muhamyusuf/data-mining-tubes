"""
================================================================================
BEFORE vs AFTER FEATURE SELECTION COMPARISON
================================================================================
Membuat visualisasi perbandingan performa BEFORE (semua fitur) vs AFTER
(dengan feature selection) untuk 4 metode feature selection.

Output: Grafik terpisah yang lebih fokus dan tidak penuh
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("=" * 80)
print("CREATING BEFORE vs AFTER FEATURE SELECTION COMPARISON")
print("=" * 80)

# Load data
output_dir = Path(__file__).parent / 'outputs-comparison'
df1 = pd.read_csv(output_dir / 'dataset1_feature_selection_comparison.csv')
df2 = pd.read_csv(output_dir / 'dataset2_feature_selection_comparison.csv')

print("\n✅ Loaded comparison results")

# ============================================================================
# DATASET 1: BEFORE vs AFTER ANALYSIS
# ============================================================================
print("\n[1/2] 📊 Analyzing Dataset 1...")

# Extract baseline and feature selection results
baseline_d1 = df1[df1['Method'] == 'Baseline (No Selection)']
methods_d1 = ['Information Gain', 'Chi-Square', 'Pearson Correlation', 'RFECV']

# Create comparison data
comparison_d1 = []
for method in methods_d1:
    method_data = df1[df1['Method'] == method]
    for model in ['Decision Tree', 'Naive Bayes', 'Logistic Regression']:
        baseline_f1 = baseline_d1[baseline_d1['Model'] == model]['F1-Score'].values[0]
        after_f1 = method_data[method_data['Model'] == model]['F1-Score'].values[0]
        n_features = method_data['Features'].values[0]
        
        comparison_d1.append({
            'Method': method,
            'Model': model,
            'Before (All Features)': baseline_f1,
            'After (Selected Features)': after_f1,
            'Improvement (%)': ((after_f1 - baseline_f1) / baseline_f1) * 100,
            'Features Used': n_features,
            'Feature Reduction (%)': ((46 - n_features) / 46) * 100
        })

df_comp_d1 = pd.DataFrame(comparison_d1)

# Save detailed comparison
df_comp_d1.to_csv(output_dir / 'dataset1_before_after_detailed.csv', index=False)
print(f"✅ Saved: dataset1_before_after_detailed.csv")

# ============================================================================
# DATASET 2: BEFORE vs AFTER ANALYSIS
# ============================================================================
print("\n[2/2] 📊 Analyzing Dataset 2...")

baseline_d2 = df2[df2['Method'] == 'Baseline (No Selection)']
methods_d2 = ['Information Gain', 'Chi-Square', 'Pearson Correlation', 'RFECV']

comparison_d2 = []
for method in methods_d2:
    method_data = df2[df2['Method'] == method]
    for model in ['Decision Tree', 'Naive Bayes', 'Logistic Regression']:
        baseline_f1 = baseline_d2[baseline_d2['Model'] == model]['F1-Score'].values[0]
        after_f1 = method_data[method_data['Model'] == model]['F1-Score'].values[0]
        n_features = method_data['Features'].values[0]
        
        comparison_d2.append({
            'Method': method,
            'Model': model,
            'Before (All Features)': baseline_f1,
            'After (Selected Features)': after_f1,
            'Improvement (%)': ((after_f1 - baseline_f1) / baseline_f1) * 100,
            'Features Used': n_features,
            'Feature Reduction (%)': ((13 - n_features) / 13) * 100
        })

df_comp_d2 = pd.DataFrame(comparison_d2)

# Save detailed comparison
df_comp_d2.to_csv(output_dir / 'dataset2_before_after_detailed.csv', index=False)
print(f"✅ Saved: dataset2_before_after_detailed.csv")

# ============================================================================
# VISUALIZATION 1: BEFORE vs AFTER BAR CHARTS (IMPROVED DESIGN)
# ============================================================================
print("\n📊 Creating Visualization 1: Before vs After Bar Charts...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Before vs After Feature Selection: F1-Score Comparison', 
             fontsize=16, fontweight='bold', y=0.995)

models = ['Decision Tree', 'Naive Bayes', 'Logistic Regression']
model_colors = ['#3498db', '#e74c3c', '#2ecc71']

# Dataset 1: Group by Method
for idx, method in enumerate(methods_d1):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    method_data = df_comp_d1[df_comp_d1['Method'] == method]
    
    x = np.arange(len(models))
    width = 0.35
    
    before_scores = [method_data[method_data['Model'] == m]['Before (All Features)'].values[0] for m in models]
    after_scores = [method_data[method_data['Model'] == m]['After (Selected Features)'].values[0] for m in models]
    n_features = method_data['Features Used'].values[0]
    
    # Bars with model-specific colors
    bars1 = ax.bar(x - width/2, before_scores, width, label='Before (46 feat)', 
                   color='lightgray', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Color after bars by model
    for i, (bar_x, score) in enumerate(zip(x, after_scores)):
        ax.bar(bar_x + width/2, score, width, color=model_colors[i], 
               alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (b, a) in enumerate(zip(before_scores, after_scores)):
        # Before label
        ax.text(i - width/2, b/2, f'{b:.3f}', ha='center', va='center', 
               fontsize=8, fontweight='bold', color='black')
        # After label
        ax.text(i + width/2, a/2, f'{a:.3f}', ha='center', va='center', 
               fontsize=8, fontweight='bold', color='white')
        
        # Improvement arrow and percentage
        improvement = ((a - b) / b) * 100
        arrow_color = '#27ae60' if improvement > 0.5 else '#e74c3c' if improvement < -0.5 else '#95a5a6'
        arrow_symbol = '↑' if improvement > 0.5 else '↓' if improvement < -0.5 else '→'
        
        y_pos = max(b, a) + 0.03
        ax.text(i, y_pos, f'{arrow_symbol}{abs(improvement):.1f}%', 
               ha='center', va='bottom', fontsize=10, color=arrow_color, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=arrow_color, linewidth=2))
    
    ax.set_ylabel('F1-Score', fontweight='bold', fontsize=11)
    ax.set_title(f'{method}\n46 → {n_features} features', fontweight='bold', fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight='bold')
    ax.set_ylim([0, 1.15])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Legend only on first subplot
    if idx == 0:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgray', edgecolor='black', label='Before'),
            Patch(facecolor=model_colors[0], edgecolor='black', label='After (DT)'),
            Patch(facecolor=model_colors[1], edgecolor='black', label='After (NB)'),
            Patch(facecolor=model_colors[2], edgecolor='black', label='After (LR)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig(output_dir / 'dataset1_before_after_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: dataset1_before_after_comparison.png")
plt.close()

# Dataset 2: Separate visualization (simpler because differences are small)
print("\n📊 Creating Visualization 1b: Dataset 2 Before vs After...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Dataset 2: Before vs After Feature Selection (Wave Height)', 
             fontsize=16, fontweight='bold', y=0.995)

for idx, method in enumerate(methods_d2):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    method_data = df_comp_d2[df_comp_d2['Method'] == method]
    
    x = np.arange(len(models))
    width = 0.35
    
    before_scores = [method_data[method_data['Model'] == m]['Before (All Features)'].values[0] for m in models]
    after_scores = [method_data[method_data['Model'] == m]['After (Selected Features)'].values[0] for m in models]
    n_features = method_data['Features Used'].values[0]
    
    # Bars with model-specific colors
    bars1 = ax.bar(x - width/2, before_scores, width, label='Before (13 feat)', 
                   color='lightgray', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for i, (bar_x, score) in enumerate(zip(x, after_scores)):
        ax.bar(bar_x + width/2, score, width, color=model_colors[i], 
               alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (b, a) in enumerate(zip(before_scores, after_scores)):
        ax.text(i - width/2, b + 0.003, f'{b:.4f}', ha='center', va='bottom', 
               fontsize=9, fontweight='bold')
        ax.text(i + width/2, a + 0.003, f'{a:.4f}', ha='center', va='bottom', 
               fontsize=9, fontweight='bold', color=model_colors[i])
        
        # Improvement indicator
        improvement = ((a - b) / b) * 100
        arrow_color = '#27ae60' if improvement > 0.1 else '#e74c3c' if improvement < -0.1 else '#95a5a6'
        arrow_symbol = '▲' if improvement > 0.1 else '▼' if improvement < -0.1 else '●'
        
        ax.annotate(f'{arrow_symbol}{improvement:+.2f}%', 
                   xy=(i, max(b, a)), xytext=(i, max(b,a) + 0.015),
                   ha='center', fontsize=9, color=arrow_color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=arrow_color, linewidth=1.5))
    
    ax.set_ylabel('F1-Score', fontweight='bold', fontsize=11)
    ax.set_title(f'{method}\n13 → {n_features} features', fontweight='bold', fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight='bold')
    ax.set_ylim([0.92, 1.02])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    if idx == 0:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgray', edgecolor='black', label='Before'),
            Patch(facecolor=model_colors[0], edgecolor='black', label='After (DT)'),
            Patch(facecolor=model_colors[1], edgecolor='black', label='After (NB)'),
            Patch(facecolor=model_colors[2], edgecolor='black', label='After (LR)')
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig(output_dir / 'dataset2_before_after_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: dataset2_before_after_comparison.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: IMPROVEMENT HEATMAP
# ============================================================================
print("\n📊 Creating Visualization 2: Improvement Heatmap...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Performance Improvement After Feature Selection (%)', 
             fontsize=16, fontweight='bold')

# Dataset 1 Heatmap
pivot_d1 = df_comp_d1.pivot_table(index='Method', columns='Model', 
                                   values='Improvement (%)', aggfunc='mean')
pivot_d1 = pivot_d1.reindex(methods_d1)

sns.heatmap(pivot_d1, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            cbar_kws={'label': 'Improvement (%)'}, linewidths=0.5, ax=axes[0],
            vmin=-30, vmax=15)
axes[0].set_title('Dataset 1: Pharmacy Demand\n(46 → Selected Features)', 
                  fontweight='bold', fontsize=12)
axes[0].set_xlabel('Model', fontweight='bold')
axes[0].set_ylabel('Feature Selection Method', fontweight='bold')

# Dataset 2 Heatmap
pivot_d2 = df_comp_d2.pivot_table(index='Method', columns='Model', 
                                   values='Improvement (%)', aggfunc='mean')
pivot_d2 = pivot_d2.reindex(methods_d2)

sns.heatmap(pivot_d2, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            cbar_kws={'label': 'Improvement (%)'}, linewidths=0.5, ax=axes[1],
            vmin=-1, vmax=2)
axes[1].set_title('Dataset 2: Wave Height\n(13 → Selected Features)', 
                  fontweight='bold', fontsize=12)
axes[1].set_xlabel('Model', fontweight='bold')
axes[1].set_ylabel('Feature Selection Method', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'before_after_improvement_heatmap.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: before_after_improvement_heatmap.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: AVERAGE IMPROVEMENT SUMMARY (IMPROVED)
# ============================================================================
print("\n📊 Creating Visualization 3: Average Improvement Summary...")

fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], hspace=0.3)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

fig.suptitle('Average Performance Improvement: Before vs After Feature Selection', 
             fontsize=16, fontweight='bold', y=0.98)

# Dataset 1
avg_improvement_d1 = df_comp_d1.groupby('Method')['Improvement (%)'].mean().reindex(methods_d1)
avg_features_d1 = df_comp_d1.groupby('Method')['Features Used'].mean().reindex(methods_d1)

# Sort by improvement
sorted_idx = avg_improvement_d1.argsort()
sorted_methods = [methods_d1[i] for i in sorted_idx]
sorted_improvements = avg_improvement_d1.iloc[sorted_idx]
sorted_features = avg_features_d1.iloc[sorted_idx]

# Color by performance
bar_colors = ['#e74c3c' if x < -5 else '#f39c12' if x < 0 else '#95a5a6' if x < 1 else '#27ae60' 
              for x in sorted_improvements]

bars = ax1.barh(sorted_methods, sorted_improvements, color=bar_colors, alpha=0.85, 
                edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Average Improvement (%)', fontweight='bold', fontsize=12)
ax1.set_title('Dataset 1: Pharmacy Demand (46 features baseline)', 
              fontweight='bold', fontsize=13, pad=15)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.7)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.set_xlim([-30, 8])

# Add labels
for i, (val, feat) in enumerate(zip(sorted_improvements, sorted_features)):
    # Value label
    x_pos = val + (1.5 if val > 0 else -1.5)
    align = 'left' if val > 0 else 'right'
    ax1.text(x_pos, i, f'{val:+.1f}%', ha=align, va='center', 
            fontweight='bold', fontsize=11, color='black')
    
    # Feature count label (on the bar)
    bar_x = val / 2
    ax1.text(bar_x, i, f'{int(feat)} feat', ha='center', va='center', 
            fontweight='bold', fontsize=9, color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

# Dataset 2
avg_improvement_d2 = df_comp_d2.groupby('Method')['Improvement (%)'].mean().reindex(methods_d2)
avg_features_d2 = df_comp_d2.groupby('Method')['Features Used'].mean().reindex(methods_d2)

sorted_idx2 = avg_improvement_d2.argsort()
sorted_methods2 = [methods_d2[i] for i in sorted_idx2]
sorted_improvements2 = avg_improvement_d2.iloc[sorted_idx2]
sorted_features2 = avg_features_d2.iloc[sorted_idx2]

bar_colors2 = ['#e74c3c' if x < -0.2 else '#f39c12' if x < 0 else '#95a5a6' if x < 0.2 else '#27ae60' 
               for x in sorted_improvements2]

bars = ax2.barh(sorted_methods2, sorted_improvements2, color=bar_colors2, alpha=0.85,
                edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Average Improvement (%)', fontweight='bold', fontsize=12)
ax2.set_title('Dataset 2: Wave Height (13 features baseline)', 
              fontweight='bold', fontsize=13, pad=15)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.7)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_xlim([-0.6, 0.7])

for i, (val, feat) in enumerate(zip(sorted_improvements2, sorted_features2)):
    x_pos = val + (0.08 if val > 0 else -0.08)
    align = 'left' if val > 0 else 'right'
    ax2.text(x_pos, i, f'{val:+.2f}%', ha=align, va='center', 
            fontweight='bold', fontsize=11, color='black')
    
    bar_x = val / 2
    ax2.text(bar_x, i, f'{int(feat)} feat', ha='center', va='center', 
            fontweight='bold', fontsize=9, color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

plt.tight_layout()
plt.savefig(output_dir / 'before_after_avg_improvement.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: before_after_avg_improvement.png")
plt.close()

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n📋 SUMMARY TABLE: Before vs After Comparison")
print("=" * 80)

print("\n📊 Dataset 1: Pharmacy Demand (46 features)")
print("-" * 80)
summary_d1 = df_comp_d1.groupby('Method').agg({
    'Features Used': 'first',
    'Feature Reduction (%)': 'first',
    'Improvement (%)': 'mean'
}).round(2)
summary_d1.columns = ['Features', 'Reduction %', 'Avg Improvement %']
print(summary_d1.to_string())

print("\n\n📊 Dataset 2: Wave Height (13 features)")
print("-" * 80)
summary_d2 = df_comp_d2.groupby('Method').agg({
    'Features Used': 'first',
    'Feature Reduction (%)': 'first',
    'Improvement (%)': 'mean'
}).round(2)
summary_d2.columns = ['Features', 'Reduction %', 'Avg Improvement %']
print(summary_d2.to_string())

# ============================================================================
# BEST/WORST ANALYSIS
# ============================================================================
print("\n\n🏆 BEST IMPROVEMENTS:")
print("-" * 80)

# Dataset 1
best_d1 = df_comp_d1.nlargest(3, 'Improvement (%)')
print("\nDataset 1 (Top 3):")
for idx, row in best_d1.iterrows():
    print(f"  {row['Method']:20s} + {row['Model']:20s}: {row['Improvement (%)']:+.2f}% "
          f"({int(row['Features Used'])} features)")

# Dataset 2
best_d2 = df_comp_d2.nlargest(3, 'Improvement (%)')
print("\nDataset 2 (Top 3):")
for idx, row in best_d2.iterrows():
    print(f"  {row['Method']:20s} + {row['Model']:20s}: {row['Improvement (%)']:+.2f}% "
          f"({int(row['Features Used'])} features)")

print("\n\n⚠️  WORST DEGRADATIONS:")
print("-" * 80)

# Dataset 1
worst_d1 = df_comp_d1.nsmallest(3, 'Improvement (%)')
print("\nDataset 1 (Bottom 3):")
for idx, row in worst_d1.iterrows():
    print(f"  {row['Method']:20s} + {row['Model']:20s}: {row['Improvement (%)']:+.2f}% "
          f"({int(row['Features Used'])} features)")

# Dataset 2
worst_d2 = df_comp_d2.nsmallest(3, 'Improvement (%)')
print("\nDataset 2 (Bottom 3):")
for idx, row in worst_d2.iterrows():
    print(f"  {row['Method']:20s} + {row['Model']:20s}: {row['Improvement (%)']:+.2f}% "
          f"({int(row['Features Used'])} features)")

print("\n" + "=" * 80)
print("✅ BEFORE vs AFTER COMPARISON COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  📄 CSV:")
print("     1. dataset1_before_after_detailed.csv")
print("     2. dataset2_before_after_detailed.csv")
print("\n  📊 VISUALIZATIONS:")
print("     3. dataset1_before_after_comparison.png  (4 methods detail)")
print("     4. dataset2_before_after_comparison.png  (4 methods detail)")
print("     5. before_after_improvement_heatmap.png  (comprehensive heatmap)")
print("     6. before_after_avg_improvement.png      (summary ranking)")
