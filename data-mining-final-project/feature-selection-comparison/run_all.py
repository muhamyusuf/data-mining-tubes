"""
Run All RFECV Analyses
Tugas Besar Penambangan Data 2025

Executes both dataset analyses sequentially
"""

import subprocess
import time
from pathlib import Path

print("="*80)
print("RFECV PREPROCESSING VALIDATION - BATCH EXECUTION")
print("Tugas Besar Penambangan Data 2025")
print("="*80)

script_dir = Path(__file__).parent

scripts = [
    ('Dataset 1: Pharmacy Transaction', 'dataset1_rfecv.py'),
    ('Dataset 2: Wave Measurement', 'dataset2_rfecv.py')
]

results = []

for name, script in scripts:
    print(f"\n{'='*80}")
    print(f"🚀 Starting: {name}")
    print(f"   Script: {script}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python', str(script_dir / script)],
            capture_output=False,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        status = "✅ SUCCESS"
        results.append((name, status, elapsed))
        
        print(f"\n✅ {name} completed in {elapsed:.2f}s")
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        status = "❌ FAILED"
        results.append((name, status, elapsed))
        print(f"\n❌ {name} failed after {elapsed:.2f}s")
        print(f"Error: {e}")

print("\n" + "="*80)
print("📊 BATCH EXECUTION SUMMARY")
print("="*80)

total_time = sum(r[2] for r in results)

for name, status, elapsed in results:
    print(f"\n{status} {name}")
    print(f"   Time: {elapsed:.2f}s")

print(f"\n{'='*80}")
print(f"Total execution time: {total_time:.2f}s")
print(f"{'='*80}")

print("\n💾 Results saved to: rfecv-only/outputs/")
print("   - dataset1_comparison.csv")
print("   - dataset1_selected_features.csv")
print("   - dataset2_comparison.csv")
print("   - dataset2_selected_features.csv")

print("\n✅ ALL ANALYSES COMPLETE!")
