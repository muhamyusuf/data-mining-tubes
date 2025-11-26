# 🌊 Dataset 2 - Wave Analysis Guide

## Dataset Overview
**Tipe:** Data gelombang laut (ocean wave data)
**Format:** Excel files (*.xlsx)
**Lokasi:** `dataset-type-2/`
**Files:** Gelombang 1.xlsx, Gelombang 2.xlsx, ..., Gelombang 6.xlsx

## Objective
Prediksi **tinggi gelombang** atau parameter gelombang lainnya

## Auto-Detection System

Notebook ini **otomatis mendeteksi**:
1. ✅ Semua file Excel di folder `dataset-type-2/`
2. ✅ Kolom numeric di setiap file
3. ✅ Target variable (kolom numeric terakhir)
4. ✅ Feature columns (semua kolom numeric lainnya)

**Tidak perlu tahu nama kolom sebelumnya!**

## How Auto-Detection Works

### Step 1: Find All Excel Files
```python
files = glob.glob('dataset-type-2/*.xlsx')
# Finds: Gelombang 1.xlsx, Gelombang 2.xlsx, etc.
```

### Step 2: Load & Combine
```python
dfs = [pd.read_excel(f) for f in files]
df = pd.concat(dfs)
```

### Step 3: Detect Numeric Columns
```python
numeric_cols = df.select_dtypes(include=[np.number]).columns
```

### Step 4: Auto-Assign Target
```python
target_col = numeric_cols[-1]  # Last numeric column
feature_cols = numeric_cols[:-1]  # All others
```

## Example Data Structure

Misalnya file Excel berisi:
```
| Date       | Wind_Speed | Wave_Height | Wave_Period | Temperature |
|------------|-----------|-------------|-------------|-------------|
| 2023-01-01 | 15.2      | 2.5         | 8.3         | 28.1        |
| 2023-01-02 | 18.4      | 3.1         | 9.2         | 27.8        |
```

Auto-detection akan:
- **Target:** `Temperature` (last numeric column)
- **Features:** `Wind_Speed`, `Wave_Height`, `Wave_Period`

## 6 Experiments (Same as Dataset 1)

| #  | Feature Selection | Model     | Best For |
|----|------------------|-----------|----------|
| 1  | RFECV            | LightGBM  | Finding optimal feature subset iteratively |
| 2  | RFECV            | GRU       | Deep learning with RFE features |
| 3  | MI               | LightGBM  | Statistical feature importance |
| 4  | MI               | GRU       | Deep learning with MI features |
| 5  | SHAP             | LightGBM  | Interpretable feature importance |
| 6  | SHAP             | GRU       | Deep learning with SHAP features |

## Expected Results

### ✅ Good Results:
- RMSE: **Varies by data scale**
- R²: **0.4-0.9** (depends on data complexity)
- Different methods show different performance
- Training time: RFECV > SHAP > MI

### ❌ Warning Signs:
- All experiments have identical results → Check data leakage
- R² = 1.0 → Too perfect, likely leakage
- RMSE = 0 → Definite leakage

## Feature Selection Methods Comparison

### RFECV (Recursive Feature Elimination)
**Pros:**
- ✅ Iteratively finds best feature subset
- ✅ Uses cross-validation (more robust)
- ✅ Good for tree-based models

**Cons:**
- ❌ Slowest method
- ❌ May overfit with small datasets

### Mutual Information (MI)
**Pros:**
- ✅ Fast computation
- ✅ Captures non-linear relationships
- ✅ Statistical foundation

**Cons:**
- ❌ Assumes feature independence
- ❌ Less interpretable than SHAP

### SHAP Values
**Pros:**
- ✅ Most interpretable
- ✅ Shows feature contribution direction
- ✅ Game-theory based

**Cons:**
- ❌ Moderate computation time
- ❌ Requires tree-based model first

## Customization Options

### Change Target Column
Jika auto-detection salah, edit cell:
```python
# Original (auto-detect):
target_col = numeric_cols[-1]

# Manual override:
target_col = 'Wave_Height'  # Specify exact column name
```

### Change Number of Top Features
```python
# In FeatureSelector class
top_k = 10  # Default, change to 5 or 15
```

### Adjust Train/Test Split
```python
# In prepare_data function
test_size = 0.2  # Change to 0.3 for more test data
```

## Output Files

1. **results_dataset2.csv**
   - Columns: FS Method, Model, RMSE, MAE, R², Train Time, Features Selected
   - 6 rows (one per experiment)

2. **comparison_dataset2.png**
   - 4 subplots showing performance comparison
   - Visual comparison across all experiments

## Troubleshooting

### Problem: "No numeric columns found"
**Cause:** Excel files may have wrong format
**Solution:** 
- Check if Excel files have numeric data
- Open Excel manually to verify
- Ensure columns are formatted as numbers, not text

### Problem: Wrong target detected
**Cause:** Auto-detection uses last column
**Solution:**
```python
# Override auto-detection
target_col = 'YOUR_TARGET_COLUMN_NAME'
feature_cols = [col for col in numeric_cols if col != target_col]
```

### Problem: Too many features selected
**Cause:** RFECV or MI selecting too many
**Solution:**
```python
# Reduce top_k parameter
top_k = 5  # Instead of 10
```

### Problem: Model training too slow
**Cause:** Large dataset or many features
**Solution:**
- Reduce sample size: `df = df.sample(n=10000)`
- Reduce features: Lower `top_k` value
- Use only MI method (fastest)

## Interpretation Guide

### RMSE (Lower = Better)
- Main metric for regression
- Same unit as target variable
- Compare across experiments

### R² (Higher = Better, max 1.0)
- 0.9-1.0: Excellent fit (be careful of overfitting!)
- 0.7-0.9: Good fit
- 0.4-0.7: Moderate fit
- <0.4: Poor fit

### MAE (Lower = Better)
- Mean Absolute Error
- More robust to outliers than RMSE
- Easier to interpret

### Training Time
- Important for production systems
- RFECV usually slowest
- GRU slower than LightGBM

## Next Steps After Results

1. **Identify Best Performer**
   ```python
   # Check results_dataset2.csv
   # Sort by RMSE (ascending) or R² (descending)
   ```

2. **Analyze Feature Importance**
   - Look at which features were selected
   - Compare across methods

3. **Validate on New Data**
   - Use best model on completely new Excel file
   - Check if performance holds

4. **Production Deployment**
   - Save best model
   - Create prediction pipeline
   - Monitor performance over time

## Quick Comparison with Dataset 1

| Aspect | Dataset 1 (Pharmacy) | Dataset 2 (Wave) |
|--------|---------------------|------------------|
| **Data Type** | Transaction logs | Time series measurements |
| **Target** | Product demand | Wave/ocean parameter |
| **Features** | Engineered (lags, rolling) | Raw measurements |
| **Files** | 6 CSVs | 6 Excel files |
| **Challenge** | Temporal dependency | Physical relationships |
| **Best Method** | Likely SHAP/LightGBM | TBD (depends on data) |
