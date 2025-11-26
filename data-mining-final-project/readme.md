# Feature Selection Comparison Project

## 📊 Project Overview

Perbandingan **RFECV vs SelectKBest (F-regression)** untuk feature selection menggunakan **LightGBM** pada 2 dataset berbeda.

### Two Datasets:
1. **Dataset-1 (Pharmacy)**: Prediksi transaction volume (qty_total)
2. **Dataset-2 (Wave)**: Prediksi wind speed dari parameter gelombang

## 🚀 Quick Start

```bash
# Run kedua dataset sekaligus
python scripts/run_all.py

# Atau run satu-satu:
python scripts/dataset1_analysis.py
python scripts/dataset2_analysis.py
```

**Hasil:**
- Dataset 1: R² = 0.8965 (RFECV wins)
- Dataset 2: R² = 0.8821 (SelectKBest wins)
- Total runtime: **< 2 menit**

## 📂 Project Structure

```
project/
├── dataset-type-1/          # Pharmacy transaction CSVs
├── dataset-type-2/          # Wave parameter Excel files
├── scripts/                 # Python analysis scripts
│   ├── dataset1_analysis.py
│   ├── dataset2_analysis.py
│   ├── run_all.py
│   └── main.py
├── notebooks/               # Jupyter notebooks (original)
│   ├── dataset1-pharmacy-analysis.ipynb
│   ├── dataset2-wave-analysis.ipynb
│   └── feature-selection-comparison.ipynb
├── outputs/                 # Analysis results
│   ├── dataset1-output/
│   │   ├── comparison_summary.csv
│   │   ├── selected_features.csv
│   │   ├── test_predictions.csv
│   │   ├── all_feature_scores.csv
│   │   ├── results_visualization.png
│   │   └── analysis_summary.txt
│   └── dataset2-output/     # Same structure
├── docs/                    # Documentation
│   ├── EXECUTIVE-SUMMARY.md
│   ├── HOW-TO-RUN.md
│   ├── QUICK-REFERENCE.md
│   ├── README-FEATURE-SELECTION.md
│   ├── VISUAL-WORKFLOW.md
│   ├── DATASET-1-GUIDE.md
│   └── DATASET-2-GUIDE.md
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 🎯 Feature Selection Methods

1. **RFECV** (Recursive Feature Elimination with CV)
   - Model-based, iterative elimination
   - 5-fold cross-validation
   - Minimum 5 features (Dataset 1) / 3 features (Dataset 2)
   - Best for complex non-linear patterns

2. **SelectKBest** (F-regression)
   - Statistical F-test
   - Fast, efficient
   - Best for linear relationships
   - Selects same k as RFECV for fair comparison

## 📊 Results Summary

| Dataset | Winner | RMSE Test | R² Test | Overfitting Gap |
|---------|--------|-----------|---------|-----------------|
| Pharmacy | RFECV | 1.83 | 0.8965 | 0.0928 |
| Wave | SelectKBest | 0.9486 | 0.8821 | 0.0006 |

## ⚙️ Technical Details

✅ **LightGBM params**: n_estimators=150, max_depth=4, reg_alpha/lambda=0.3
✅ **Cross-validation**: 5-fold for RFECV
✅ **Time series handling**: shuffle=False for Dataset 1
✅ **Standardization**: Applied to Dataset 2 (wave data)
✅ **Overfitting control**: Strong regularization + min_features constraint

## 🔧 Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## 📖 Documentation

Detailed documentation available in `docs/`:
- **EXECUTIVE-SUMMARY.md**: High-level overview and key findings
- **HOW-TO-RUN.md**: Step-by-step execution guide
- **QUICK-REFERENCE.md**: Quick commands and tips
- **README-FEATURE-SELECTION.md**: Feature selection methodology
- **VISUAL-WORKFLOW.md**: Workflow diagrams
- **DATASET-1-GUIDE.md**: Pharmacy dataset details
- **DATASET-2-GUIDE.md**: Wave dataset details

## 🎓 Key Learnings

1. **R² Interpretation**: 
   - R² → 1.0 is good (high predictive power)
   - R² = 1.0000 perfect is suspicious (overfitting)
   - Current results (0.88-0.89) are excellent

2. **Method Selection**:
   - RFECV: Best for complex non-linear patterns
   - SelectKBest: Best for linear relationships

3. **Overfitting Control**:
   - Gap < 0.05: Excellent generalization
   - Gap 0.05-0.15: Acceptable
   - Gap > 0.15: High overfitting

## 👨‍💻 Author

Data Mining Final Project - Feature Selection Comparison

## 📄 License

Educational project for academic purposes.


```python
@dataclass
class Config:
    random_state: int = 42           # Random seed
    test_size: float = 0.2           # Test set size
    max_features: int = 20           # Max features to select
    gru_epochs: int = 50             # GRU training epochs
    gru_batch_size: int = 32         # GRU batch size
    # ... and more
```

## 📝 Notes

- The project uses **transaction data** as a proxy for wave height prediction
- Feature engineering creates temporal and aggregated features
- GRU model uses reshaped data for time series processing
- SHAP calculation uses a sample (1000 rows) for efficiency

## 🔗 Dependencies

- Python 3.10+
- pandas 2.3+
- numpy 2.2+
- scikit-learn 1.7+
- lightgbm 4.6+
- tensorflow 2.20+
- shap 0.49+
- matplotlib 3.10+
- seaborn 0.13+

## 📄 License

This project is for educational purposes (UAS Data Mining).

## 👤 Author

Muhammad - Data Mining Final Project

---

**Last Updated**: November 2025