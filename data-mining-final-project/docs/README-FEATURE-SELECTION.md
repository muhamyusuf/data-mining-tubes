# 🎯 Feature Selection Comparison Project

## 📋 Overview

Project ini membandingkan **2 metode feature selection** pada **2 jenis dataset** berbeda:

### Feature Selection Methods:
1. **RFECV** (Recursive Feature Elimination with Cross-Validation)
   - ✅ Iterative, model-based selection
   - ✅ Automatic optimal feature number detection
   - ✅ Cross-validation untuk robustness
   - ⏱️ Slower but thorough

2. **Mutual Information** (MI)
   - ✅ Statistical, information-theory based
   - ✅ Fast computation
   - ✅ Captures non-linear relationships
   - ⚡ Much faster than RFECV

### Model: LightGBM
- Optimal untuk tabular data
- Fast training
- Handle missing values
- Built-in regularization

---

## 📂 Project Structure

```
data-mining-final-project/
│
├── dataset-type-1/              # Pharmacy transaction data (CSV)
│   ├── 2021.csv
│   ├── 2022.csv
│   ├── 2023.csv
│   ├── A2021.csv
│   ├── A2022.csv
│   └── A2023.csv
│
├── dataset-type-2/              # Wave data (Excel)
│   ├── Gelombang (1).xlsx
│   ├── Gelombang (2).xlsx
│   ├── ... (6 files total)
│
├── dataset1-pharmacy-analysis.ipynb    # 📊 Notebook for Dataset 1
├── dataset2-wave-analysis.ipynb        # 🌊 Notebook for Dataset 2
│
├── requirements.txt             # Python dependencies
├── pyproject.toml              # UV project config
├── uv.lock                     # UV lockfile
│
├── DATASET-1-GUIDE.md          # Guide for pharmacy data
├── DATASET-2-GUIDE.md          # Guide for wave data
└── README-FEATURE-SELECTION.md # This file
```

---

## 🚀 Quick Start

### 1. Setup Environment (Using UV)

```powershell
# Navigate to project directory
cd "c:\Users\muham\OneDrive\Desktop\data-mining-uas\data-mining-final-project"

# Install dependencies (if not already done)
uv add pandas numpy scikit-learn lightgbm matplotlib seaborn openpyxl jupyter ipykernel
```

### 2. Run Notebooks

#### Option A: Using Jupyter
```powershell
uv run jupyter notebook
```

Then open:
- `dataset1-pharmacy-analysis.ipynb`
- `dataset2-wave-analysis.ipynb`

#### Option B: Using VS Code
1. Open folder in VS Code
2. Select Python kernel from UV environment
3. Run cells sequentially

---

## 📊 Dataset 1: Pharmacy Demand Prediction

### Objective
Prediksi **demand produk apotek** (`qty_out`) berdasarkan historical transactions.

### Features Created
- **Temporal**: day, month, day_of_week, week_of_year
- **Lag features**: qty_in_lag1, qty_out_lag1, ... lag7
- **Rolling stats**: qty_in_roll7, qty_out_roll7
- **Aggregations**: sum, mean untuk input/output values

### Expected Results
| Metric | Good Range | Warning Sign |
|--------|-----------|--------------|
| RMSE   | 10-50     | 0.0000 (data leakage!) |
| R²     | 0.3-0.8   | 1.0000 (too perfect) |
| MAE    | Similar to RMSE | All metrics zero |

### Key Insights
- Lag features sangat penting untuk time series
- RFECV vs MI: Compare speed vs accuracy trade-off
- Top products filtering untuk efisiensi

---

## 🌊 Dataset 2: Wave Parameters Prediction

### Objective
Prediksi parameter gelombang dengan **auto-detection** target & features.

### Auto-Detection Features
- ✅ Automatically finds all Excel files
- ✅ Detects numeric columns
- ✅ Assigns target (last column) & features (others)
- ✅ No need to know column names beforehand!

### Expected Results
| Metric | Interpretation |
|--------|---------------|
| RMSE   | Lower is better, depends on data scale |
| R²     | 0.4-0.9 typical for wave data |
| MAE    | More robust to outliers than RMSE |

### Key Insights
- Standardization important for wave data
- Feature selection more critical with many features
- Compare RFECV vs MI for feature agreement

---

## 📈 Output Files

Each notebook generates:

### Dataset 1 (Pharmacy)
- `results_dataset1_pharmacy.csv` - Detailed metrics comparison
- `comparison_dataset1_pharmacy.png` - 4-panel visualization (RMSE, R², Time, MAE)

### Dataset 2 (Wave)
- `results_dataset2_wave.csv` - Detailed metrics comparison
- `comparison_dataset2_wave.png` - 4-panel visualization
- `predictions_dataset2_wave.png` - Actual vs Predicted scatter plots

---

## 🔬 Experiment Workflow

### For Each Dataset:

1. **Data Loading & Preprocessing**
   - Load CSV/Excel files
   - Clean missing values
   - Feature engineering (Dataset 1)

2. **RFECV Experiment**
   ```python
   RFECV → Select optimal features → Train LightGBM → Evaluate
   ```

3. **Mutual Information Experiment**
   ```python
   MI Scores → Select top K features → Train LightGBM → Evaluate
   ```

4. **Comparison & Analysis**
   - Side-by-side metrics
   - Visualization
   - Feature agreement analysis
   - Winner determination

---

## 🏆 Interpreting Results

### When RFECV Wins:
- ✅ Better for complex feature interactions
- ✅ More thorough validation
- ⚠️ Takes longer time
- 📌 Use when: Accuracy is critical, time is available

### When MI Wins:
- ✅ Fast feature screening
- ✅ Good for independent features
- ⚡ Much faster execution
- 📌 Use when: Speed matters, initial exploration

### When Results are Similar:
- 🤝 Features are relatively independent
- 💡 Either method works well
- 💰 Choose MI for efficiency

---

## 📊 Comparison Metrics Explained

| Metric | What It Measures | Better When |
|--------|-----------------|-------------|
| **RMSE** | Prediction error magnitude | Lower ⬇️ |
| **MAE** | Average absolute error | Lower ⬇️ |
| **R²** | Variance explained | Higher ⬆️ (max 1.0) |
| **Time** | Processing duration | Lower ⬇️ |
| **Features** | Number selected | Fewer = simpler model |

---

## ⚠️ Common Issues & Solutions

### Issue: "No numeric columns found"
**Solution:** Check Excel/CSV files have numeric data, not all text

### Issue: RMSE = 0.0000
**Solution:** Data leakage detected! Check:
- Not using future data (no `.shift(-1)`)
- Proper train/test split
- No target in features

### Issue: Memory error
**Solution:**
- Reduce top products (Dataset 1): `top_n = 20` instead of 50
- Sample data: `df = df.sample(n=10000)`
- Use fewer lag features

### Issue: Model too slow
**Solution:**
- Reduce CV folds: `cv=3` instead of `cv=5`
- Reduce estimators: `n_estimators=100`
- Use MI only (skip RFECV)

---

## 🎯 Best Practices

### For Dataset 1 (Time Series):
1. ✅ Always check for data leakage
2. ✅ Use lag features for temporal patterns
3. ✅ Filter top products untuk efisiensi
4. ✅ Don't shuffle train/test split

### For Dataset 2 (Wave Data):
1. ✅ Standardize features (already included)
2. ✅ Verify auto-detected target is correct
3. ✅ Check feature correlation before selection
4. ✅ Shuffle is OK (not time series)

### General:
1. ✅ Run all cells sequentially (don't skip)
2. ✅ Check for warnings/errors after each step
3. ✅ Validate results make sense (R² < 1.0, RMSE > 0)
4. ✅ Compare both methods before deciding

---

## 📝 Next Steps After Analysis

### If RFECV Performs Better:
1. Save selected features
2. Retrain on full dataset
3. Use for production model
4. Monitor performance over time

### If MI Performs Better:
1. Document MI scores for features
2. Consider using MI for feature screening
3. Combine with domain knowledge
4. Faster iteration cycles possible

### If Both Perform Similarly:
1. Use MI for speed advantage
2. Validate with RFECV periodically
3. Focus on model tuning instead
4. Ensemble both feature sets

---

## 🔍 Advanced Analysis (Optional)

### Feature Agreement Analysis
```python
# See which features both methods agree on
common_features = set(rfecv_features) & set(mi_features)
```

### Feature Importance Visualization
```python
# Already included in notebooks
# Check MI scores and RFECV rankings
```

### Cross-Dataset Insights
- Compare which method works better for each dataset type
- Understand when to use RFECV vs MI
- Build intuition for feature selection

---

## 💡 Key Takeaways

1. **RFECV**: Thorough but slow, best for final model
2. **MI**: Fast and effective, great for exploration
3. **LightGBM**: Excellent for both datasets
4. **Time Series**: Need careful feature engineering (Dataset 1)
5. **Auto-Detection**: Works well for unknown datasets (Dataset 2)

---

## 📞 Support & Documentation

- **Dataset 1 Guide**: See `DATASET-1-GUIDE.md`
- **Dataset 2 Guide**: See `DATASET-2-GUIDE.md`
- **Main README**: See `readme.md`

---

## ✅ Checklist Before Running

- [ ] UV environment setup complete
- [ ] All dependencies installed
- [ ] Dataset files in correct folders
- [ ] Jupyter kernel selected
- [ ] Enough disk space for outputs
- [ ] Ready to run all cells sequentially

---

**Happy Feature Selecting! 🎯📊**
