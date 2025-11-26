"""
Run both dataset analyses sequentially
Usage: python scripts/run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

# Get project root directory
project_root = Path(__file__).parent.parent

print("="*70)
print("🚀 RUNNING ALL ANALYSES")
print("="*70)

datasets = [
    ("Dataset 1: Pharmacy", "scripts/dataset1_analysis.py"),
    ("Dataset 2: Wave", "scripts/dataset2_analysis.py")
]

results = []

for name, script in datasets:
    print(f"\n{'='*70}")
    print(f"▶️  Starting: {name}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, project_root / script],
            capture_output=False,
            text=True,
            check=True
        )
        elapsed = time.time() - start_time
        results.append((name, "✅ SUCCESS", elapsed))
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        results.append((name, "❌ FAILED", elapsed))
        print(f"\n❌ Error in {name}")

print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)
for name, status, elapsed in results:
    print(f"{status:<12} {name:<30} ({elapsed:.1f}s)")
print("="*70)
