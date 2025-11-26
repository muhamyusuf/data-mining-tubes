# 🚀 HOW TO RUN - Step by Step Instructions

## 📋 Prerequisites

Sebelum mulai, pastikan Anda punya:
- ✅ Windows PC
- ✅ UV installed ([Install UV](https://github.com/astral-sh/uv))
- ✅ VS Code atau Jupyter Notebook
- ✅ Dataset files di folder yang benar

---

## 🎯 Method 1: Using VS Code (Recommended)

### Step 1: Open Project
```powershell
# Buka folder project di VS Code
code "c:\Users\muham\OneDrive\Desktop\data-mining-uas\data-mining-final-project"
```

### Step 2: Select Python Interpreter
1. Tekan `Ctrl + Shift + P`
2. Ketik: "Python: Select Interpreter"
3. Pilih interpreter dari UV environment (biasanya ada di `.venv`)

### Step 3: Open Notebook
1. Klik file `dataset1-pharmacy-analysis.ipynb`
2. Wait for kernel to load
3. Click "Run All" atau run cell by cell

### Step 4: Repeat for Dataset 2
1. Klik file `dataset2-wave-analysis.ipynb`
2. Click "Run All"

### Step 5: View Results
- Check generated CSV files
- View PNG visualizations in folder

---

## 🎯 Method 2: Using Jupyter Notebook

### Step 1: Navigate to Project
```powershell
cd "c:\Users\muham\OneDrive\Desktop\data-mining-uas\data-mining-final-project"
```

### Step 2: Launch Jupyter
```powershell
uv run jupyter notebook
```

### Step 3: Run Dataset 1 Notebook
1. Browser akan terbuka
2. Click `dataset1-pharmacy-analysis.ipynb`
3. Menu: Cell → Run All
4. Wait for completion (~5-10 minutes)

### Step 4: Run Dataset 2 Notebook
1. Back to Jupyter home
2. Click `dataset2-wave-analysis.ipynb`
3. Menu: Cell → Run All
4. Wait for completion (~3-5 minutes)

### Step 5: Check Results
Files akan muncul di folder:
- `results_dataset1_pharmacy.csv`
- `results_dataset2_wave.csv`
- `comparison_dataset1_pharmacy.png`
- `comparison_dataset2_wave.png`
- `predictions_dataset2_wave.png`

---

## 🎯 Method 3: Command Line Execution

### Convert Notebook to Python Script
```powershell
# Install nbconvert if needed
uv add nbconvert

# Convert to Python
uv run jupyter nbconvert --to python dataset1-pharmacy-analysis.ipynb

# Run the script
uv run python dataset1-pharmacy-analysis.py
```

---

## 📊 What Happens During Execution

### Dataset 1 Notebook:

```
┌─────────────────────────────────────────┐
│ 1. Load & Merge CSV files (2021-2023)  │ ~30 sec
├─────────────────────────────────────────┤
│ 2. Preprocess & Filter Top Products    │ ~1 min
├─────────────────────────────────────────┤
│ 3. Feature Engineering (15+ features)  │ ~1 min
├─────────────────────────────────────────┤
│ 4. RFECV Feature Selection             │ ~3-5 min ⏳
├─────────────────────────────────────────┤
│ 5. Train LightGBM with RFECV features  │ ~30 sec
├─────────────────────────────────────────┤
│ 6. MI Feature Selection                │ ~30 sec ⚡
├─────────────────────────────────────────┤
│ 7. Train LightGBM with MI features     │ ~30 sec
├─────────────────────────────────────────┤
│ 8. Compare Results & Visualize         │ ~30 sec
└─────────────────────────────────────────┘
Total: ~8-10 minutes
```

### Dataset 2 Notebook:

```
┌─────────────────────────────────────────┐
│ 1. Auto-load Excel files               │ ~10 sec
├─────────────────────────────────────────┤
│ 2. Auto-detect features & target       │ ~5 sec
├─────────────────────────────────────────┤
│ 3. Standardize features                │ ~5 sec
├─────────────────────────────────────────┤
│ 4. RFECV Feature Selection             │ ~2-3 min ⏳
├─────────────────────────────────────────┤
│ 5. Train LightGBM with RFECV features  │ ~20 sec
├─────────────────────────────────────────┤
│ 6. MI Feature Selection                │ ~10 sec ⚡
├─────────────────────────────────────────┤
│ 7. Train LightGBM with MI features     │ ~20 sec
├─────────────────────────────────────────┤
│ 8. Compare & Visualize                 │ ~20 sec
└─────────────────────────────────────────┘
Total: ~4-6 minutes
```

---

## ✅ Expected Outputs

### Console Output:

```
✅ Libraries loaded successfully!
✅ Loaded dataset-type-1/2021.csv: 138354 rows
✅ Loaded dataset-type-1/2022.csv: 142876 rows
...
📊 Total data: 825,432 rows
🔝 Top 30 products by volume
✅ Daily aggregations: 12,450 rows
✅ Feature engineering completed!
📊 Total features: 19

===========================================
🔬 EXPERIMENT 1: RFECV + LightGBM
===========================================
⏳ Running RFECV... (this may take a few minutes)
✅ RFECV completed in 245.32 seconds
🎯 Optimal number of features: 12
📊 Selected features: [...]

📊 RFECV Results:
  ⏱️  Total time: 267.45s
  🎯 Features used: 12
  📉 RMSE: 15.2341
  📉 MAE: 10.8765
  📈 R²: 0.7234

===========================================
🔬 EXPERIMENT 2: Mutual Information + LightGBM
===========================================
⏳ Calculating Mutual Information scores...
✅ MI calculation completed in 12.45 seconds

📊 Mutual Information Results:
  ⏱️  Total time: 28.76s
  🎯 Features used: 12
  📉 RMSE: 15.8932
  📉 MAE: 11.2341
  📈 R²: 0.7056

🏆 WINNER ANALYSIS:
  🥇 RFECV wins on RMSE (15.2341 < 15.8932)
  🥇 RFECV wins on R² (0.7234 > 0.7056)
  ⚡ Mutual Information is 9.3x FASTER

💾 Results saved to: results_dataset1_pharmacy.csv
📊 Visualization saved to: comparison_dataset1_pharmacy.png
✅ Analysis Complete!
```

### Files Created:

```
📁 data-mining-final-project/
│
├── 📄 results_dataset1_pharmacy.csv
│   Method,RMSE,MAE,R2,Time (s),Num Features
│   RFECV,15.2341,10.8765,0.7234,267.45,12
│   Mutual Information,15.8932,11.2341,0.7056,28.76,12
│
├── 📄 results_dataset2_wave.csv
│   (Similar structure)
│
├── 🖼️ comparison_dataset1_pharmacy.png
│   [4-panel bar chart visualization]
│
├── 🖼️ comparison_dataset2_wave.png
│   [4-panel bar chart visualization]
│
└── 🖼️ predictions_dataset2_wave.png
    [Actual vs Predicted scatter plots]
```

---

## 🔧 Troubleshooting

### Issue 1: Kernel Not Found
**Error:** "No kernel found"

**Solution:**
```powershell
# Install ipykernel
uv add ipykernel

# Create kernel
uv run python -m ipykernel install --user --name=data-mining
```

### Issue 2: Module Not Found
**Error:** "ModuleNotFoundError: No module named 'lightgbm'"

**Solution:**
```powershell
# Reinstall dependencies
uv add pandas numpy scikit-learn lightgbm matplotlib seaborn openpyxl jupyter ipykernel
```

### Issue 3: Notebook Runs Forever
**Problem:** RFECV taking too long

**Solution:**
Edit notebook cell:
```python
# Change this:
rfecv = RFECV(estimator=base_estimator, step=1, cv=5)

# To this (less CV folds):
rfecv = RFECV(estimator=base_estimator, step=1, cv=3)
```

### Issue 4: Memory Error
**Error:** "MemoryError" or system freezes

**Solution:**
Edit notebook cell:
```python
# Reduce top products
top_n = 20  # Instead of 30

# Or sample data
df = df.sample(n=50000)  # Instead of all data
```

### Issue 5: Excel File Not Found (Dataset 2)
**Error:** "FileNotFoundError"

**Solution:**
```powershell
# Check files exist
ls dataset-type-2/*.xlsx

# If empty, files might be in different location
# Update path in notebook cell
```

### Issue 6: RMSE = 0.0000
**Problem:** Results too perfect (data leakage!)

**Solution:**
1. Restart kernel
2. Run All Cells again from scratch
3. Do NOT modify feature engineering code
4. Check you're not accidentally using future data

---

## 📊 Interpreting Your Results

### Good Results ✅

**Dataset 1 (Pharmacy):**
- RMSE: 10-50 (reasonable)
- R²: 0.3-0.8 (good fit)
- Methods show difference (RFECV usually better accuracy, MI faster)

**Dataset 2 (Wave):**
- RMSE: Depends on data scale
- R²: 0.4-0.9 (varies by data)
- Clear winner or informed tie

### Bad Results ❌

- RMSE = 0.0000 → Data leakage! Restart and rerun
- R² = 1.0000 → Too perfect, data leakage
- Both methods identical → Something wrong
- Negative R² → Model worse than baseline

---

## 🎯 Next Steps After Running

### 1. Analyze CSV Results
```python
import pandas as pd

# Load results
df1 = pd.read_csv('results_dataset1_pharmacy.csv')
df2 = pd.read_csv('results_dataset2_wave.csv')

# Compare
print(df1)
print(df2)
```

### 2. View Visualizations
- Open PNG files in folder
- Include in report/presentation
- Compare metrics visually

### 3. Determine Winner
- Check which method has lower RMSE
- Check which method has higher R²
- Consider time trade-off
- Make informed decision

### 4. Use Selected Features
```python
# Get selected features from notebook variables
print("RFECV features:", rfecv_features)
print("MI features:", mi_features)

# Save for later use
import pickle
with open('selected_features.pkl', 'wb') as f:
    pickle.dump(rfecv_features, f)
```

---

## 📝 Checklist Before Running

- [ ] UV installed and working
- [ ] Project folder opened
- [ ] Dataset files in correct locations:
  - [ ] `dataset-type-1/*.csv` (6 files)
  - [ ] `dataset-type-2/*.xlsx` (6 files)
- [ ] Python interpreter selected (if using VS Code)
- [ ] Enough disk space (~500MB for outputs)
- [ ] Time available (~15 minutes total)

---

## 💡 Tips for Success

### 1. Run Sequentially
- Don't skip cells
- Run in order from top to bottom
- Wait for each cell to complete

### 2. Monitor Progress
- Watch console output
- Check for errors after each cell
- Progress indicators will show

### 3. Save Intermediate Results
- Notebooks auto-save
- Don't close browser/VS Code during execution
- Can resume if interrupted

### 4. Compare Results
- Run both notebooks completely
- Compare Dataset 1 vs Dataset 2
- Different datasets may favor different methods

### 5. Document Findings
- Take screenshots of visualizations
- Note which method wins for each dataset
- Record execution times

---

## 🎓 Learning While Running

### Watch For:

1. **Data Loading:** How many rows loaded
2. **Feature Engineering:** How features created
3. **RFECV Progress:** See iterative elimination
4. **MI Speed:** Notice how fast it is
5. **Model Training:** LightGBM iterations
6. **Metrics Comparison:** Which method wins

### Questions to Answer:

1. Which method has better RMSE?
2. Which method is faster?
3. What's the speed vs accuracy trade-off?
4. Do methods agree on important features?
5. Which method would you use in production?

---

## ✅ Success Indicators

You know it's working when:

- ✅ No red error messages
- ✅ Progress bars/indicators appear
- ✅ Metrics look reasonable (not 0.0000 or 1.0000)
- ✅ CSV files created
- ✅ PNG visualizations generated
- ✅ Clear winner or informed decision possible

---

## 📞 Need Help?

### Quick Checks:
1. Restart kernel and run again
2. Check dataset files exist
3. Verify UV environment active
4. Read error messages carefully

### Common Fixes:
- Kernel issues → Reinstall ipykernel
- Module errors → Reinstall dependencies
- Slow execution → Reduce data size
- Memory errors → Sample data or reduce top_n

---

**Ready to run? Let's go! 🚀**

**Estimated Total Time:** 15-20 minutes for both notebooks  
**Expected Output:** 5 files (2 CSV + 3 PNG)  
**Success Rate:** 99% if following instructions ✅
