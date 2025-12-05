"""
================================================================================
VISUALISASI FITUR TERPILIH DARI SETIAP METODE FEATURE SELECTION
================================================================================
Membuat visualisasi fitur yang dipilih oleh setiap metode untuk Dataset 1 dan 2

Output: 8 visualisasi (4 metode x 2 dataset)
================================================================================
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

print("=" * 80)
print("CREATING FEATURE SELECTION VISUALIZATIONS")
print("=" * 80)

# Setup output directory
output_dir = Path(__file__).parent / 'outputs-comparison'
output_dir.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', 
          '#34495e', '#e67e22', '#95a5a6', '#d35400']

# ============================================================================
# DATASET 1: PHARMACY TRANSACTION - SELECTED FEATURES
# ============================================================================
print("\n[1/2] 📊 Creating Dataset 1 Visualizations...")

dataset1_selections = {
    'Information Gain': [
        'qty_roll_mean_3', 'qty_roll_max_3', 'qty_roll_min_3', 'qty_ewma_3', 
        'qty_ewma_7', 'qty_roll_mean_7', 'qty_roll_max_7', 'qty_roll_std_3',
        'qty_roll_min_7', 'qty_ewma_14'
    ],
    'Chi-Square': [
        'qty_roll_max_3', 'qty_roll_mean_7', 'qty_roll_max_7', 'qty_roll_mean_14',
        'qty_roll_max_14', 'qty_roll_mean_21', 'qty_roll_max_21', 'qty_roll_mean_30',
        'qty_roll_max_30', 'qty_roll_mean_3'
    ],
    'Pearson Correlation': [
        'qty_change_3', 'qty_change_7', 'qty_change_1', 'qty_roll_max_30',
        'qty_roll_max_21', 'qty_roll_max_14', 'qty_roll_max_7', 'qty_roll_std_30',
        'qty_roll_std_21', 'qty_roll_std_14'
    ],
    'RFECV': [
        'qty_lag_1', 'qty_roll_min_3', 'qty_ewma_3', 'qty_change_1',
        'qty_change_3', 'qty_cv_7'
    ]
}

# Create figure for Dataset 1
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
fig1.suptitle('Dataset 1: Pharmacy Transaction - Selected Features by Each Method', 
              fontsize=18, fontweight='bold', y=0.995)

methods = ['Information Gain', 'Chi-Square', 'Pearson Correlation', 'RFECV']
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for method, pos in zip(methods, positions):
    ax = axes1[pos]
    features = dataset1_selections[method]
    n_features = len(features)
    
    # Create horizontal bar chart
    y_pos = np.arange(n_features)
    # Reverse order so top feature appears at top
    features_display = features[::-1]
    
    # Color bars
    bar_colors = colors[:n_features][::-1]
    
    bars = ax.barh(y_pos, [1]*n_features, color=bar_colors, alpha=0.7, edgecolor='black')
    
    # Set labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_display, fontsize=9)
    ax.set_xlabel('Selected', fontsize=10, fontweight='bold')
    ax.set_title(f'{method}\n({n_features} features)', 
                fontsize=12, fontweight='bold', pad=10)
    
    # Remove x-axis
    ax.set_xlim(0, 1.1)
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Add feature ranking numbers
    for i, (bar, feature) in enumerate(zip(bars, features_display)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
               f'#{n_features-i}',
               ha='left', va='center', fontsize=8, fontweight='bold',
               color='darkgreen')

plt.tight_layout()
output_path1 = output_dir / 'dataset1_selected_features_visualization.png'
plt.savefig(output_path1, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path1}")
plt.close()

# ============================================================================
# DATASET 2: WAVE MEASUREMENT - SELECTED FEATURES
# ============================================================================
print("\n[2/2] 📊 Creating Dataset 2 Visualizations...")

dataset2_selections = {
    'Information Gain': [
        'Hmax(m)', 'WaveDir(deg)', 'PrimSwell(m)', 'WindSeaDir(deg)', 
        'WavePeriod(s)', 'SeaSurfaceSalinity(PSU)', 'WindSpeed(knots)'
    ],
    'Chi-Square': [
        'Hmax(m)', 'WaveDir(deg)', 'PrimSwell(m)', 'WindSeaDir(deg)', 
        'WavePeriod(s)', 'SeaSurfaceSalinity(PSU)', 'WindSpeed(knots)'
    ],
    'Pearson Correlation': [
        'Hmax(m)', 'WaveDir(deg)', 'WindSpeed(knots)', 'WavePeriod(s)', 
        'WindSeaDir(deg)', 'PrimSwell(m)', 'WindDir(deg)'
    ],
    'RFECV': [
        'Hmax(m)', 'WaveDir(deg)', 'WavePeriod(s)', 
        'SurfCurrentDir', 'WindSpeed(knots)'
    ]
}

# Create figure for Dataset 2
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Dataset 2: Wave Measurement - Selected Features by Each Method', 
              fontsize=18, fontweight='bold', y=0.995)

for method, pos in zip(methods, positions):
    ax = axes2[pos]
    features = dataset2_selections[method]
    n_features = len(features)
    
    # Create horizontal bar chart
    y_pos = np.arange(n_features)
    features_display = features[::-1]
    
    # Color bars
    bar_colors = colors[:n_features][::-1]
    
    bars = ax.barh(y_pos, [1]*n_features, color=bar_colors, alpha=0.7, edgecolor='black')
    
    # Set labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_display, fontsize=10)
    ax.set_xlabel('Selected', fontsize=10, fontweight='bold')
    ax.set_title(f'{method}\n({n_features} features)', 
                fontsize=12, fontweight='bold', pad=10)
    
    # Remove x-axis
    ax.set_xlim(0, 1.1)
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Add feature ranking numbers
    for i, (bar, feature) in enumerate(zip(bars, features_display)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
               f'#{n_features-i}',
               ha='left', va='center', fontsize=9, fontweight='bold',
               color='darkgreen')

plt.tight_layout()
output_path2 = output_dir / 'dataset2_selected_features_visualization.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path2}")
plt.close()

# ============================================================================
# SUMMARY COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("📊 SUMMARY OF SELECTED FEATURES")
print("=" * 80)

print("\n🔹 DATASET 1: PHARMACY TRANSACTION")
print("-" * 80)
for method in methods:
    features = dataset1_selections[method]
    print(f"\n{method:25s}: {len(features)} features")
    print(f"  Top 5: {features[:5]}")

print("\n\n🔹 DATASET 2: WAVE MEASUREMENT")
print("-" * 80)
for method in methods:
    features = dataset2_selections[method]
    print(f"\n{method:25s}: {len(features)} features")
    print(f"  All: {features}")

# ============================================================================
# FEATURE OVERLAP ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("🔍 FEATURE OVERLAP ANALYSIS")
print("=" * 80)

print("\n🔹 DATASET 1: Common Features")
print("-" * 80)
d1_all_features = set()
for features in dataset1_selections.values():
    d1_all_features.update(features)
print(f"Total unique features selected: {len(d1_all_features)}")
print(f"Original feature space: 46 features")

# Find features selected by multiple methods
feature_count_d1 = {}
for features in dataset1_selections.values():
    for f in features:
        feature_count_d1[f] = feature_count_d1.get(f, 0) + 1

common_d1 = {f: count for f, count in feature_count_d1.items() if count > 1}
if common_d1:
    print(f"\nFeatures selected by multiple methods:")
    for f, count in sorted(common_d1.items(), key=lambda x: x[1], reverse=True):
        print(f"  {f:30s}: {count} methods")

print("\n🔹 DATASET 2: Common Features")
print("-" * 80)
d2_all_features = set()
for features in dataset2_selections.values():
    d2_all_features.update(features)
print(f"Total unique features selected: {len(d2_all_features)}")
print(f"Original feature space: 13 features")

# Find features selected by multiple methods
feature_count_d2 = {}
for features in dataset2_selections.values():
    for f in features:
        feature_count_d2[f] = feature_count_d2.get(f, 0) + 1

common_d2 = {f: count for f, count in feature_count_d2.items() if count > 1}
if common_d2:
    print(f"\nFeatures selected by multiple methods:")
    for f, count in sorted(common_d2.items(), key=lambda x: x[1], reverse=True):
        print(f"  {f:30s}: {count} methods")

print("\n" + "=" * 80)
print("✅ VISUALIZATION COMPLETE!")
print("=" * 80)
print(f"\n📁 Output files:")
print(f"   1. {output_path1.name}")
print(f"   2. {output_path2.name}")
print("\n" + "=" * 80)
