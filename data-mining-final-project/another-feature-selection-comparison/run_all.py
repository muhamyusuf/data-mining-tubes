"""
Feature Selection Comparison - Run All Datasets
Executes both dataset1 and dataset2 comparisons and generates comprehensive report
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import time

print("="*80)
print("🚀 FEATURE SELECTION COMPARISON - COMPREHENSIVE ANALYSIS")
print("="*80)
print("\nObjective: Compare RFECV vs Mutual Information across two different datasets")
print("Model: Ridge Regression (anti-overfitting via L2 regularization)")
print("\n" + "="*80)

# ===== DATASET 1: PHARMACY =====
print("\n\n" + "🏥 RUNNING DATASET 1: PHARMACY TRANSACTION ANALYSIS")
print("="*80)

start_time = time.time()
result1 = subprocess.run([sys.executable, 'dataset1_comparison.py'], 
                        cwd=Path(__file__).parent,
                        capture_output=False)
dataset1_time = time.time() - start_time

if result1.returncode != 0:
    print(f"\n❌ Dataset 1 failed with exit code {result1.returncode}")
    dataset1_success = False
else:
    print(f"\n✅ Dataset 1 completed in {dataset1_time:.2f}s")
    dataset1_success = True

# ===== DATASET 2: WAVE =====
print("\n\n" + "🌊 RUNNING DATASET 2: WAVE DATA ANALYSIS")
print("="*80)

start_time = time.time()
result2 = subprocess.run([sys.executable, 'dataset2_comparison.py'], 
                        cwd=Path(__file__).parent,
                        capture_output=False)
dataset2_time = time.time() - start_time

if result2.returncode != 0:
    print(f"\n❌ Dataset 2 failed with exit code {result2.returncode}")
    dataset2_success = False
else:
    print(f"\n✅ Dataset 2 completed in {dataset2_time:.2f}s")
    dataset2_success = True

# ===== GENERATE CONSOLIDATED REPORT =====
print("\n\n" + "="*80)
print("📊 CONSOLIDATED REPORT")
print("="*80)

output_dir = Path(__file__).parent / 'outputs'

# Load results if available
if dataset1_success and (output_dir / 'dataset1_comparison.csv').exists():
    df1 = pd.read_csv(output_dir / 'dataset1_comparison.csv')
    print("\n📋 DATASET 1 (Pharmacy Transaction) - Summary:")
    print("-" * 80)
    print(df1.to_string(index=False))
    
if dataset2_success and (output_dir / 'dataset2_comparison.csv').exists():
    df2 = pd.read_csv(output_dir / 'dataset2_comparison.csv')
    print("\n📋 DATASET 2 (Wave Data) - Summary:")
    print("-" * 80)
    print(df2.to_string(index=False))

# ===== FINAL SUMMARY =====
print("\n\n" + "="*80)
print("🎯 EXECUTION SUMMARY")
print("="*80)
print(f"\nDataset 1 (Pharmacy): {'✅ Success' if dataset1_success else '❌ Failed'} ({dataset1_time:.2f}s)")
print(f"Dataset 2 (Wave):     {'✅ Success' if dataset2_success else '❌ Failed'} ({dataset2_time:.2f}s)")
print(f"\nTotal execution time: {dataset1_time + dataset2_time:.2f}s")

if dataset1_success and dataset2_success:
    print("\n✅ ALL ANALYSES COMPLETED SUCCESSFULLY!")
    print(f"\n📁 Results saved to: {output_dir.absolute()}")
elif dataset1_success or dataset2_success:
    print("\n⚠️ PARTIAL SUCCESS - Some datasets completed")
else:
    print("\n❌ ALL ANALYSES FAILED")
    sys.exit(1)
