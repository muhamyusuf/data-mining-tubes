# RFECV Feature Selection Validation
## Tugas Besar Penambangan Data 2025

[![Python](https://img.shields.io/badge/Python-3.10.18-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

---

## 📋 Overview

Proyek ini mengimplementasikan dan memvalidasi **RFECV (Recursive Feature Elimination with Cross-Validation)** sebagai metode preprocessing untuk feature selection. Fokus utama adalah **mengevaluasi efektivitas preprocessing**, bukan mencari model terbaik.

### 🎯 Objektif

1. ✅ Implementasi RFECV sebagai preprocessing method
2. ✅ Validasi menggunakan model ML sederhana (Decision Tree, Naive Bayes, Logistic Regression)
3. ✅ Perbandingan BEFORE vs AFTER feature selection
4. ✅ Analisis komparatif pada 2 dataset berbeda karakteristik
5. ✅ Critical thinking: kapan RFECV efektif vs tidak efektif

---

## 📊 Dataset

### Dataset 1: Pharmacy Transaction
- **Sumber:** 6 CSV files (2021-2023, A2021-A2023)
- **Raw data:** 479,951 transaksi
- **Setelah preprocessing:** 21,224 sampel
- **Features:** 53 (temporal, lag, rolling, noise)
- **Target:** Demand classification (High/Low)

### Dataset 2: Wave Measurement
- **Sumber:** 6 Excel files (sensor gelombang)
- **Raw data:** 61,225 rows
- **Setelah cleaning:** 8,736 sampel
- **Features:** 86 (64 original + 22 noise)
- **Target:** Wave parameter (column 63)

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+
# Virtual environment sudah setup: .venv

# Aktifkan virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### Installation

```bash
# Install dependencies (sudah terinstall di .venv)
pip install pandas numpy scikit-learn scipy openpyxl
```

### Running Analysis

#### Option 1: Run All (Recommended)
```bash
cd rfecv-only
python run_all.py
```

#### Option 2: Run Individual Dataset
```bash
# Dataset 1 only
python dataset1_rfecv.py

# Dataset 2 only
python dataset2_rfecv.py
```

### Execution Time
- **Dataset 1:** ~16 detik
- **Dataset 2:** ~14 detik
- **Total:** ~30 detik

---

## 📁 Project Structure

```
rfecv-only/
├── dataset1_rfecv.py          # RFECV validation for pharmacy data
├── dataset2_rfecv.py          # RFECV validation for wave data
├── run_all.py                 # Batch execution script
├── LAPORAN.md                 # Laporan akademis lengkap
├── README.md                  # This file
└── outputs/
    ├── dataset1_comparison.csv          # Results comparison Dataset 1
    ├── dataset1_selected_features.csv   # Selected features Dataset 1
    ├── dataset2_comparison.csv          # Results comparison Dataset 2
    └── dataset2_selected_features.csv   # Selected features Dataset 2
```

---

## 📈 Key Results

### Dataset 1: Pharmacy Transaction ✅

| Metric | Value |
|--------|-------|
| **Feature Reduction** | 81.1% (53 → 10) |
| **Best Improvement** | +376.99% (Decision Tree) |
| **Average Improvement** | +95.21% |
| **Significant Models** | 2/3 (Decision Tree, Naive Bayes) |
| **Recommendation** | ✅ **USE RFECV** |

**Key Insight:** RFECV sangat efektif karena mengeliminasi noise dan mengurangi overfitting pada Decision Tree.

### Dataset 2: Wave Measurement ⚠️

| Metric | Value |
|--------|-------|
| **Feature Reduction** | 87.2% (86 → 11) |
| **Best Improvement** | +0.62% (Logistic Regression) |
| **Average Improvement** | -0.19% (DEGRADASI) |
| **Significant Models** | 1/3 (Naive Bayes - degraded) |
| **Recommendation** | ❌ **SKIP RFECV** |

**Key Insight:** RFECV tidak efektif karena original features sudah highly informative, baseline performance sudah tinggi.

---

## 🔬 Methodology

### RFECV Parameters

```python
RFECV(
    estimator=DecisionTreeClassifier(max_depth=8, min_samples_split=50),
    step=1,              # Dataset 1 (smaller step for precision)
    step=3,              # Dataset 2 (larger step for speed)
    cv=5,                # 5-fold cross-validation
    scoring='f1',        # Optimize F1-Score
    min_features_to_select=5,
    n_jobs=-1            # Parallel processing
)
```

### Validation Models

| Model | Purpose | Parameters |
|-------|---------|------------|
| Decision Tree Classifier | Classification | `max_depth=8-10, min_samples_split=20-50` |
| Decision Tree Regressor | Regression | `max_depth=8-10, min_samples_split=20-50` |
| Naive Bayes | Classification | Default `GaussianNB()` |
| Logistic Regression | Classification | `max_iter=1000` |

### Evaluation Metrics

- **F1-Score:** Harmonic mean of Precision & Recall
- **R²:** Coefficient of Determination
- **RMSE:** Root Mean Squared Error
- **p-value:** Paired t-test for statistical significance (α=0.05)

---

## 💡 Critical Insights

### When is RFECV Effective?

✅ **USE RFECV when:**
- Dataset has many noisy/redundant features
- Baseline model is overfitting
- Sample size is large relative to features (>200:1 ratio)
- Features are relatively independent

⚠️ **SKIP RFECV when:**
- Baseline performance is already high
- Features are highly interdependent
- Small sample size (<100:1 ratio)
- Computational cost is prohibitive

### Model-Specific Findings

| Model | Dataset 1 | Dataset 2 | Observation |
|-------|-----------|-----------|-------------|
| **Decision Tree** | +376.99% | -0.20% | Most benefited from noise removal |
| **Naive Bayes** | -0.29% | -1.14% | Degraded in both (violates independence assumption?) |
| **Logistic Regression** | +0.47% | +0.62% | Most stable across datasets |

---

## 📚 Technical Details

### Dataset 1 Feature Engineering

**Strategy:** Create many features including intentional noise to demonstrate RFECV effectiveness

- **Temporal (9):** day, month, day_of_week, week, quarter, is_weekend, is_month_start/end, day_of_year
- **Lag (7):** qty_lag for 1, 2, 3, 7, 14, 21, 28 days
- **Rolling (20):** mean, std, max, min for windows 3, 7, 14, 21, 30
- **EWMA (3):** Exponential weighted moving average for 3, 7, 14
- **Changes (3):** qty_change for lag 1, 3, 7, 14
- **CV (3):** Coefficient of variation for windows 7, 14, 30
- **Ratios (2):** in_out_ratio, out_in_ratio
- **NOISE (6):** random_noise 1/2/3, constant, day²/month² (intentional)

### Dataset 2 Augmentation

- **Original (64):** Physical sensor measurements
- **Added Noise (22):**
  - Random noise (10)
  - Uniform noise (5)
  - Constants (3)
  - Redundant squares/cubes (3)
  - Correlated noise (1)

**Rationale:** Clean datasets don't show dramatic preprocessing effects. Noise addition demonstrates RFECV capability to eliminate uninformative features.

---

## 📊 Output Files

### Comparison CSV Format

| Column | Description |
|--------|-------------|
| `Model` | Model name |
| `Preprocessing` | BEFORE (No RFECV) / AFTER (RFECV) |
| `Features` | Number of features used |
| `Metric_Value` | F1-Score or R² |
| `Metric_Name` | F1-Score or R² |

### Selected Features CSV Format

| Column | Description |
|--------|-------------|
| `Feature` | Feature name |
| `Rank` | Selection rank (1 = most important) |

---

## 🎓 Academic Report

Lihat **LAPORAN.md** untuk analisis lengkap:
- Metodologi detail
- Hasil dan pembahasan
- Analisis komparatif
- Critical insights
- Kesimpulan dan rekomendasi

---

## 🛠️ Troubleshooting

### Issue: ModuleNotFoundError
```bash
# Pastikan virtual environment aktif
.venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn scipy openpyxl
```

### Issue: FileNotFoundError
```bash
# Pastikan run dari directory yang benar
cd rfecv-only
python dataset1_rfecv.py
```

### Issue: Excel file not found (Dataset 2)
```bash
# Pastikan file Excel ada di dataset-type-2/
# Check: Gelombang (1).xlsx, ..., Gelombang (6).xlsx
```

---

## 📌 Notes

### Reproducibility
- **Random seed:** 42 (all random operations)
- **Python version:** 3.10.18
- **Libraries:** See requirements above
- **OS:** Windows (PowerShell)

### Performance Tips
- Use `n_jobs=-1` for parallel processing (faster RFECV)
- Increase `step` parameter for faster execution (lower precision)
- Reduce `cv` folds if dataset is very large

---

## 🏆 Conclusion

**Main Takeaway:**  
Feature selection effectiveness is **highly dataset-dependent**. RFECV is not a silver bullet - it works excellently on noisy datasets (Dataset 1: +95% improvement) but can degrade performance on clean datasets (Dataset 2: -0.19%).

**Recommendation:**  
Always perform empirical validation on your specific dataset. Don't blindly apply preprocessing methods without measuring their impact.

**Critical Thinking:**  
- ✅ Dataset 1: RFECV eliminates noise, reduces overfitting → **USE**
- ❌ Dataset 2: Features already optimal, RFECV removes useful info → **SKIP**

**Trade-off:**  
Feature reduction (80%+) is attractive for model simplicity and deployment efficiency, but only justified if performance doesn't degrade.

---

## 📧 Contact

**Tugas Besar Penambangan Data 2025**  
Institut Teknologi Sumatera  
Teknik Informatika

---

## 📄 License

Academic use only - Tugas Besar Penambangan Data IF25-32025

---

**🎉 Happy Feature Selection!**

*Remember: The best preprocessing is the one that works for YOUR data.*
