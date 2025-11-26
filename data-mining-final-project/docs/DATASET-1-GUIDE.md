# 📊 Dataset 1 - Pharmacy Analysis Guide

## Dataset Overview
**Tipe:** Transaksi apotek (pharmacy transactions)
**Format:** CSV files
**Lokasi:** `dataset-type-1/`
**Files:** 2021.csv, 2022.csv, 2023.csv, A2021.csv, A2022.csv, A2023.csv

## Objective
Prediksi **demand produk apotek** berdasarkan historical transactions

## Target Variable
`qty_out` (current day) - Jumlah produk yang keluar hari ini

**PENTING:** Kita predict current day, BUKAN future day (untuk avoid data leakage!)

## Features Created

### 1. Temporal Features
- `day`: Tanggal (1-31)
- `month`: Bulan (1-12)
- `day_of_week`: Hari dalam minggu (0=Senin, 6=Minggu)
- `week_of_year`: Minggu ke-n dalam tahun

### 2. Lag Features (Historical Data)
- `qty_in_lag1`: Input 1 hari lalu
- `qty_in_lag2`: Input 2 hari lalu
- `qty_in_lag3`: Input 3 hari lalu
- `qty_in_lag7`: Input 7 hari lalu
- `qty_out_lag1`: Output 1 hari lalu
- `qty_out_lag2`: Output 2 hari lalu
- `qty_out_lag3`: Output 3 hari lalu
- `qty_out_lag7`: Output 7 hari lalu

### 3. Rolling Statistics (7-day window)
- `qty_in_roll7`: Average input dalam 7 hari
- `qty_out_roll7`: Average output dalam 7 hari

### 4. Daily Aggregations
- `qty_in_sum`: Total input hari ini
- `qty_in_mean`: Rata-rata input hari ini
- `value_in_sum`: Total nilai input
- `value_in_mean`: Rata-rata nilai input

## Data Processing Steps

### Step 1: Load & Merge
```python
# Combine all CSV files (2021-2023, A2021-A2023)
df = pd.concat([load_csv(file) for file in csv_files])
```

### Step 2: Filter Top Products
```python
# Only analyze top 50 products by volume
top_products = df.groupby('KODE_BARANG')['QTY_KLR'].sum().nlargest(50)
```

### Step 3: Create Daily Aggregations
```python
# Group by product + date
daily_df = df.groupby(['KODE_BARANG', 'TANGGAL']).agg({...})
```

### Step 4: Add Lag Features
```python
# Create historical features with proper grouping
for lag in [1,2,3,7]:
    df[f'qty_in_lag{lag}'] = df.groupby('product')['qty_in'].shift(lag)
```

### Step 5: Drop NaN & Split Train/Test
```python
# Remove rows with missing lag features
# 80% train, 20% test
```

## Expected Results

### ✅ Good Results (No Leakage):
- RMSE: **10-50** (tergantung scale data)
- R²: **0.3-0.8** (realistic prediction)
- MAE: Similar to RMSE

### ❌ Bad Results (Data Leakage):
- RMSE: **0.0000**
- R²: **1.0000**
- All predictions perfect (too good to be true!)

## 6 Experiments

| #  | Feature Selection | Model     | Description |
|----|------------------|-----------|-------------|
| 1  | RFECV            | LightGBM  | Recursive elimination + gradient boosting |
| 2  | RFECV            | GRU       | Recursive elimination + deep learning |
| 3  | MI               | LightGBM  | Mutual information + gradient boosting |
| 4  | MI               | GRU       | Mutual information + deep learning |
| 5  | SHAP             | LightGBM  | SHAP values + gradient boosting |
| 6  | SHAP             | GRU       | SHAP values + deep learning |

## Troubleshooting

### Problem: Target all zeros
**Solution:** Pastikan tidak pakai `.shift(-1)` untuk target

### Problem: RMSE 0.0000
**Solution:** Restart kernel + run ulang dari awal

### Problem: Model terlalu lambat
**Solution:** Kurangi jumlah top products (dari 50 ke 20)

### Problem: Memory error
**Solution:** Filter data berdasarkan tahun tertentu saja

## Output Files

1. `results_dataset1.csv` - Detailed metrics for all experiments
2. `comparison_dataset1.png` - 4-panel visualization:
   - RMSE comparison
   - R² comparison  
   - Training time
   - MAE comparison

## Next Steps

1. ✅ Run notebook completely
2. 📊 Analyze visualization
3. 📈 Compare which combination works best
4. 🎯 Use best model for production forecasting
