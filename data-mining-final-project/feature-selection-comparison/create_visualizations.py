"""
Create Comparison Visualizations for README
Generates comparison charts from RFECV results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Setup
output_dir = Path(__file__).parent / 'outputs'
output_dir.mkdir(exist_ok=True)

# Load data
df1 = pd.read_csv(output_dir / 'dataset1_comparison.csv')
df2 = pd.read_csv(output_dir / 'dataset2_comparison.csv')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'before': '#e74c3c',
    'after': '#27ae60',
    'neutral': '#3498db'
}

# ===== FIGURE 1: OVERALL COMPARISON =====
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('RFECV Preprocessing: Comparative Analysis', 
             fontsize=20, fontweight='bold', y=0.98)

# --- ROW 1: DATASET 1 (PHARMACY) ---
ax1 = fig.add_subplot(gs[0, 0])
# Performance comparison
d1_before = df1[df1['Preprocessing'] == 'BEFORE (No RFECV)']
d1_after = df1[df1['Preprocessing'] == 'AFTER (RFECV)']

models = ['Decision Tree', 'Naive Bayes', 'Logistic Reg']
before_f1 = [
    d1_before[d1_before['Model'] == 'Decision Tree (Clf)']['Metric_Value'].values[0],
    d1_before[d1_before['Model'] == 'Naive Bayes']['Metric_Value'].values[0],
    d1_before[d1_before['Model'] == 'Logistic Regression']['Metric_Value'].values[0]
]
after_f1 = [
    d1_after[d1_after['Model'] == 'Decision Tree (Clf)']['Metric_Value'].values[0],
    d1_after[d1_after['Model'] == 'Naive Bayes']['Metric_Value'].values[0],
    d1_after[d1_after['Model'] == 'Logistic Regression']['Metric_Value'].values[0]
]

x = np.arange(len(models))
width = 0.35
ax1.bar(x - width/2, before_f1, width, label='BEFORE (53 feat)', color=colors['before'], alpha=0.8)
ax1.bar(x + width/2, after_f1, width, label='AFTER (10 feat)', color=colors['after'], alpha=0.8)
ax1.set_ylabel('F1-Score', fontweight='bold')
ax1.set_title('Dataset 1: Pharmacy Transaction', fontweight='bold', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15, ha='right')
ax1.legend(fontsize=8)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for i, (b, a) in enumerate(zip(before_f1, after_f1)):
    ax1.text(i - width/2, b, f'{b:.3f}', ha='center', va='bottom', fontsize=8)
    ax1.text(i + width/2, a, f'{a:.3f}', ha='center', va='bottom', fontsize=8)

ax2 = fig.add_subplot(gs[0, 1])
# Feature reduction
feature_data = [53, 10]
colors_pie = [colors['before'], colors['after']]
wedges, texts, autotexts = ax2.pie(feature_data, labels=['BEFORE\n53 features', 'AFTER\n10 features'],
                                     autopct='%d', colors=colors_pie, startangle=90,
                                     textprops={'fontsize': 10, 'fontweight': 'bold'})
ax2.set_title('Feature Reduction: 81.1%', fontweight='bold', fontsize=12)

ax3 = fig.add_subplot(gs[0, 2])
# Improvement
improvements_d1 = [
    ((after_f1[0] - before_f1[0]) / before_f1[0] * 100),
    ((after_f1[1] - before_f1[1]) / before_f1[1] * 100),
    ((after_f1[2] - before_f1[2]) / before_f1[2] * 100)
]
colors_bar = [colors['after'] if imp > 0 else colors['before'] for imp in improvements_d1]
bars = ax3.barh(models, improvements_d1, color=colors_bar, alpha=0.8)
ax3.set_xlabel('Improvement (%)', fontweight='bold')
ax3.set_title('Performance Change', fontweight='bold', fontsize=12)
ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax3.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, improvements_d1)):
    ax3.text(val, i, f' {val:+.1f}%', va='center', fontsize=9, fontweight='bold')

# --- ROW 2: DATASET 2 (WAVE) ---
ax4 = fig.add_subplot(gs[1, 0])
# Performance comparison
d2_before = df2[df2['Preprocessing'] == 'BEFORE (No RFECV)']
d2_after = df2[df2['Preprocessing'] == 'AFTER (RFECV)']

before_f1_d2 = [
    d2_before[d2_before['Model'] == 'Decision Tree (Clf)']['Metric_Value'].values[0],
    d2_before[d2_before['Model'] == 'Naive Bayes']['Metric_Value'].values[0],
    d2_before[d2_before['Model'] == 'Logistic Regression']['Metric_Value'].values[0]
]
after_f1_d2 = [
    d2_after[d2_after['Model'] == 'Decision Tree (Clf)']['Metric_Value'].values[0],
    d2_after[d2_after['Model'] == 'Naive Bayes']['Metric_Value'].values[0],
    d2_after[d2_after['Model'] == 'Logistic Regression']['Metric_Value'].values[0]
]

ax4.bar(x - width/2, before_f1_d2, width, label='BEFORE (86 feat)', color=colors['before'], alpha=0.8)
ax4.bar(x + width/2, after_f1_d2, width, label='AFTER (11 feat)', color=colors['after'], alpha=0.8)
ax4.set_ylabel('F1-Score', fontweight='bold')
ax4.set_title('Dataset 2: Wave Measurement', fontweight='bold', fontsize=12)
ax4.set_xticks(x)
ax4.set_xticklabels(models, rotation=15, ha='right')
ax4.legend(fontsize=8)
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for i, (b, a) in enumerate(zip(before_f1_d2, after_f1_d2)):
    ax4.text(i - width/2, b, f'{b:.3f}', ha='center', va='bottom', fontsize=8)
    ax4.text(i + width/2, a, f'{a:.3f}', ha='center', va='bottom', fontsize=8)

ax5 = fig.add_subplot(gs[1, 1])
# Feature reduction
feature_data_d2 = [86, 11]
wedges, texts, autotexts = ax5.pie(feature_data_d2, labels=['BEFORE\n86 features', 'AFTER\n11 features'],
                                     autopct='%d', colors=colors_pie, startangle=90,
                                     textprops={'fontsize': 10, 'fontweight': 'bold'})
ax5.set_title('Feature Reduction: 87.2%', fontweight='bold', fontsize=12)

ax6 = fig.add_subplot(gs[1, 2])
# Improvement
improvements_d2 = [
    ((after_f1_d2[0] - before_f1_d2[0]) / before_f1_d2[0] * 100),
    ((after_f1_d2[1] - before_f1_d2[1]) / before_f1_d2[1] * 100),
    ((after_f1_d2[2] - before_f1_d2[2]) / before_f1_d2[2] * 100)
]
colors_bar_d2 = [colors['after'] if imp > 0 else colors['before'] for imp in improvements_d2]
bars = ax6.barh(models, improvements_d2, color=colors_bar_d2, alpha=0.8)
ax6.set_xlabel('Improvement (%)', fontweight='bold')
ax6.set_title('Performance Change', fontweight='bold', fontsize=12)
ax6.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax6.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, improvements_d2)):
    ax6.text(val, i, f' {val:+.1f}%', va='center', fontsize=9, fontweight='bold')

# --- ROW 3: SUMMARY COMPARISON ---
ax7 = fig.add_subplot(gs[2, :2])
# Side-by-side comparison
datasets = ['Dataset 1\n(Pharmacy)', 'Dataset 2\n(Wave)']
avg_before = [np.mean(before_f1), np.mean(before_f1_d2)]
avg_after = [np.mean(after_f1), np.mean(after_f1_d2)]

x_ds = np.arange(len(datasets))
width_ds = 0.35
ax7.bar(x_ds - width_ds/2, avg_before, width_ds, label='BEFORE RFECV', color=colors['before'], alpha=0.8)
ax7.bar(x_ds + width_ds/2, avg_after, width_ds, label='AFTER RFECV', color=colors['after'], alpha=0.8)
ax7.set_ylabel('Average F1-Score', fontweight='bold')
ax7.set_title('Average Performance Across All Models', fontweight='bold', fontsize=14)
ax7.set_xticks(x_ds)
ax7.set_xticklabels(datasets)
ax7.legend(fontsize=10)
ax7.grid(axis='y', alpha=0.3)

# Add value labels
for i, (b, a) in enumerate(zip(avg_before, avg_after)):
    improvement = ((a - b) / b * 100)
    ax7.text(i - width_ds/2, b, f'{b:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax7.text(i + width_ds/2, a, f'{a:.4f}\n({improvement:+.1f}%)', ha='center', va='bottom', 
             fontsize=10, fontweight='bold', color=colors['after'] if improvement > 0 else colors['before'])

ax8 = fig.add_subplot(gs[2, 2])
# Verdict summary
verdicts = ['Dataset 1\n✅ USE RFECV', 'Dataset 2\n❌ SKIP RFECV']
verdict_scores = [np.mean(improvements_d1), np.mean(improvements_d2)]
verdict_colors = [colors['after'], colors['before']]

bars = ax8.bar(range(len(verdicts)), verdict_scores, color=verdict_colors, alpha=0.8)
ax8.set_ylabel('Avg Improvement (%)', fontweight='bold')
ax8.set_title('Recommendation', fontweight='bold', fontsize=14)
ax8.set_xticks(range(len(verdicts)))
ax8.set_xticklabels(['Dataset 1', 'Dataset 2'])
ax8.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax8.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, verdict_scores):
    ax8.text(bar.get_x() + bar.get_width()/2, val, f'{val:+.1f}%', 
             ha='center', va='bottom' if val > 0 else 'top', fontsize=11, fontweight='bold')

plt.savefig(output_dir / 'comparison_summary.png', dpi=300, bbox_inches='tight')
print("✅ Saved: outputs/comparison_summary.png")

# ===== FIGURE 2: SIMPLE SUMMARY CARD =====
fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.axis('off')

# Title
fig2.text(0.5, 0.95, 'RFECV Preprocessing Validation - Summary', 
         ha='center', fontsize=18, fontweight='bold')

# Dataset 1 box
box1_y = 0.75
fig2.text(0.25, box1_y, 'Dataset 1: Pharmacy Transaction', 
         ha='center', fontsize=14, fontweight='bold', 
         bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['after'], alpha=0.2))
fig2.text(0.25, box1_y - 0.08, f'Feature Reduction: 53 → 10 (81.1%)', ha='center', fontsize=11)
fig2.text(0.25, box1_y - 0.14, f'Avg Improvement: {np.mean(improvements_d1):+.1f}%', ha='center', fontsize=11)
fig2.text(0.25, box1_y - 0.20, f'Best Model: Decision Tree (+377%)', ha='center', fontsize=11)
fig2.text(0.25, box1_y - 0.26, '✅ Recommendation: USE RFECV', 
         ha='center', fontsize=12, fontweight='bold', color='green')

# Dataset 2 box
fig2.text(0.75, box1_y, 'Dataset 2: Wave Measurement', 
         ha='center', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['before'], alpha=0.2))
fig2.text(0.75, box1_y - 0.08, f'Feature Reduction: 86 → 11 (87.2%)', ha='center', fontsize=11)
fig2.text(0.75, box1_y - 0.14, f'Avg Improvement: {np.mean(improvements_d2):+.1f}%', ha='center', fontsize=11)
fig2.text(0.75, box1_y - 0.20, f'Best Model: Logistic Reg (+0.6%)', ha='center', fontsize=11)
fig2.text(0.75, box1_y - 0.26, '❌ Recommendation: SKIP RFECV', 
         ha='center', fontsize=12, fontweight='bold', color='red')

# Key insights
insights_y = 0.35
fig2.text(0.5, insights_y, 'Key Insights', ha='center', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))

insights = [
    '✓ Dataset 1: Noisy features → RFECV eliminates noise → Performance improves',
    '✗ Dataset 2: Clean features → RFECV removes useful info → Performance degrades',
    '→ RFECV effectiveness depends on data characteristics',
    '→ Always validate preprocessing on your specific dataset'
]

for i, insight in enumerate(insights):
    fig2.text(0.5, insights_y - 0.08 - (i * 0.06), insight, ha='center', fontsize=11)

plt.savefig(output_dir / 'summary_card.png', dpi=300, bbox_inches='tight')
print("✅ Saved: outputs/summary_card.png")

print("\n✅ All visualizations created successfully!")
print("   - comparison_summary.png (detailed comparison)")
print("   - summary_card.png (executive summary)")
