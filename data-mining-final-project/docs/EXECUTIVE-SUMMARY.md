# 📊 Executive Summary: Feature Selection Comparison

## 🎯 Project Goal

Membandingkan **RFECV** vs **Mutual Information** untuk feature selection pada 2 dataset berbeda, menggunakan **LightGBM** sebagai model prediksi.

---

## 🔬 Methodology

### Feature Selection Methods Compared:

| Method | Type | Speed | Strength |
|--------|------|-------|----------|
| **RFECV** | Model-based, Iterative | Slow ⏳ | Optimal subset, Cross-validated |
| **Mutual Information** | Statistical | Fast ⚡ | Quick screening, Non-linear |

### Model Used:
- **LightGBM Regressor** - Optimal untuk tabular data, fast training

### Metrics Evaluated:
1. **RMSE** (Root Mean Squared Error) - Lower is better
2. **MAE** (Mean Absolute Error) - Lower is better
3. **R²** (R-squared) - Higher is better (max 1.0)
4. **Time** (Processing time) - Lower is better
5. **Features Selected** - Fewer = simpler model

---

## 📂 Datasets Analyzed

### Dataset 1: Pharmacy Demand Prediction 💊

**Description:**
- Transaction logs dari apotek (2021-2023)
- Prediksi demand produk (`qty_out`)

**Features Created:**
- 15+ engineered features
- Temporal: day, month, day_of_week
- Lags: 1, 2, 3, 7 days historical data
- Rolling: 7-day averages
- Aggregations: sum, mean values

**Key Characteristics:**
- Time series data
- ~300K+ transactions
- Top 30 products analyzed
- High temporal dependency

**Expected Winner:** RFECV (handles feature interactions)

---

### Dataset 2: Wave Parameters Prediction 🌊

**Description:**
- Ocean wave measurement data
- Auto-detected target & features from Excel files

**Features:**
- Auto-detected numeric columns
- Physical measurements (wave height, speed, etc.)
- Standardized for better performance

**Key Characteristics:**
- 6 Excel files combined
- Unknown structure (auto-detection used)
- Physical relationships between features

**Expected Winner:** Could go either way (depends on feature independence)

---

## 📊 Comparison Framework

### Evaluation Criteria:

```
┌─────────────────────────────────────────────────────┐
│  RFECV WINS if:                                     │
│  • RMSE significantly lower (>5% better)            │
│  • R² significantly higher (>2% better)             │
│  • Better captures feature interactions             │
│  • Time difference acceptable                       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  MI WINS if:                                        │
│  • Similar or better accuracy                       │
│  • Much faster (2x+ speedup)                        │
│  • Good enough for use case                         │
└─────────────────────────────────────────────────────┘
```

---

## 🎯 Expected Outcomes

### Scenario 1: RFECV Dominates
- **Metrics:** Better RMSE, R², MAE
- **Trade-off:** Takes longer time
- **Recommendation:** Use RFECV for final model
- **Use Case:** Production deployment where accuracy is critical

### Scenario 2: MI Dominates
- **Metrics:** Similar accuracy, much faster
- **Trade-off:** May miss feature interactions
- **Recommendation:** Use MI for iterations
- **Use Case:** Fast exploration, frequent retraining

### Scenario 3: Mixed Results
- **Metrics:** Each wins on different aspects
- **Trade-off:** Need to prioritize
- **Recommendation:** Use both (MI for exploration, RFECV for final)
- **Use Case:** Research projects, comprehensive analysis

---

## 📈 Interpretation Guide

### What Good Results Look Like:

| Metric | Dataset 1 (Pharmacy) | Dataset 2 (Wave) |
|--------|---------------------|------------------|
| **RMSE** | 10-50 (depends on scale) | Varies by data scale |
| **R²** | 0.3-0.8 | 0.4-0.9 |
| **MAE** | Similar to RMSE | Similar to RMSE |
| **Time** | RFECV > MI | RFECV > MI |

### Red Flags 🚩:

- **RMSE = 0.0000** → Data leakage detected!
- **R² = 1.0000** → Too perfect, check for leakage
- **Negative R²** → Model worse than baseline
- **All methods identical** → Something's wrong

---

## 💡 Key Insights

### RFECV Strengths:
1. ✅ Automatically finds optimal number of features
2. ✅ Cross-validation for robustness
3. ✅ Captures feature interactions
4. ✅ Model-aware selection

### RFECV Weaknesses:
1. ❌ Computationally expensive
2. ❌ Slow for large feature sets
3. ❌ Results depend on base model

### MI Strengths:
1. ✅ Very fast computation
2. ✅ Model-agnostic
3. ✅ Captures non-linear relationships
4. ✅ Based on solid theory (information theory)

### MI Weaknesses:
1. ❌ Assumes feature independence
2. ❌ May miss feature interactions
3. ❌ Need to choose K (number of features)

---

## 🏆 Decision Matrix

### Choose RFECV When:
- ✅ Accuracy is top priority
- ✅ Computational time is available
- ✅ Production deployment planned
- ✅ Feature interactions important
- ✅ Need automatic optimal K selection

### Choose MI When:
- ✅ Speed is critical
- ✅ Initial exploration phase
- ✅ Large feature space (>100 features)
- ✅ Frequent retraining needed
- ✅ Features relatively independent

### Use Both When:
- ✅ Comprehensive analysis required
- ✅ Building ensemble models
- ✅ Research/academic purposes
- ✅ Want multiple perspectives
- ✅ Validating feature importance

---

## 📋 Deliverables

### Notebooks:
1. **dataset1-pharmacy-analysis.ipynb**
   - Full pipeline for pharmacy data
   - RFECV vs MI comparison
   - Detailed results & visualizations

2. **dataset2-wave-analysis.ipynb**
   - Auto-detection implementation
   - RFECV vs MI comparison
   - Prediction analysis

### Output Files:
1. **results_dataset1_pharmacy.csv** - Detailed metrics (Dataset 1)
2. **results_dataset2_wave.csv** - Detailed metrics (Dataset 2)
3. **comparison_dataset1_pharmacy.png** - 4-panel visualization
4. **comparison_dataset2_wave.png** - 4-panel visualization
5. **predictions_dataset2_wave.png** - Actual vs Predicted plots

### Documentation:
1. **README-FEATURE-SELECTION.md** - Complete guide
2. **QUICK-REFERENCE.md** - Quick lookup
3. **EXECUTIVE-SUMMARY.md** - This file
4. **DATASET-1-GUIDE.md** - Dataset 1 specifics
5. **DATASET-2-GUIDE.md** - Dataset 2 specifics

---

## 🚀 How to Use Results

### Step 1: Run Notebooks
```powershell
uv run jupyter notebook
# Open and run both .ipynb files
```

### Step 2: Analyze Results
- Check CSV files for metrics
- View PNG visualizations
- Compare RFECV vs MI performance

### Step 3: Make Decision
- Identify winner based on priorities
- Consider trade-offs (accuracy vs speed)
- Check feature agreement percentage

### Step 4: Deploy
- Save selected features
- Retrain on full dataset (if needed)
- Monitor performance in production

---

## 📊 Presentation Ready Charts

### Chart 1: Method Comparison
```
         RFECV              vs              MI
    ┌──────────────┐              ┌──────────────┐
    │ Accuracy: ⭐⭐⭐⭐⭐ │              │ Speed:    ⭐⭐⭐⭐⭐ │
    │ Speed:    ⭐⭐       │              │ Accuracy: ⭐⭐⭐⭐   │
    │ Robust:   ⭐⭐⭐⭐⭐ │              │ Simple:   ⭐⭐⭐⭐⭐ │
    └──────────────┘              └──────────────┘
```

### Chart 2: Use Case Matrix
```
High Accuracy Need
        ↑
        │  RFECV          Both
        │  (Production)   (Research)
        │
        │  Baseline       MI
        │  (Quick check)  (Exploration)
        │
        └──────────────────────→ High Speed Need
```

---

## 🎓 Learning Outcomes

After this project, you'll know:

1. ✅ **How RFECV works** - Recursive elimination with CV
2. ✅ **How MI works** - Information-theoretic feature scoring
3. ✅ **When to use each** - Trade-offs and decision criteria
4. ✅ **LightGBM for regression** - Fast gradient boosting
5. ✅ **Feature engineering** - Lag, rolling, temporal features
6. ✅ **Data leakage prevention** - Critical for time series
7. ✅ **Auto-detection techniques** - Unknown dataset handling
8. ✅ **Comparison methodology** - Fair evaluation framework

---

## 🎯 Success Criteria

Project achieves success when:

- [x] ✅ Both notebooks execute completely
- [x] ✅ Results are valid (no data leakage)
- [x] ✅ Clear winner identified (or informed tie)
- [x] ✅ Visualizations generated
- [x] ✅ CSV results exported
- [x] ✅ Documentation complete
- [x] ✅ Reproducible workflow

---

## 📞 Quick Support

### Common Issues:

**Q: RMSE is 0.0000, what's wrong?**  
A: Data leakage! Check you're not using future data in features.

**Q: Which method should I use?**  
A: Run both! If time is critical, use MI. If accuracy is critical, use RFECV.

**Q: Features don't make sense**  
A: For Dataset 2, check auto-detection selected correct target column.

**Q: Code is too slow**  
A: Reduce `top_n` products (Dataset 1) or use smaller sample.

---

## 🔗 References

### Documentation Files:
- Full guide: `README-FEATURE-SELECTION.md`
- Quick lookup: `QUICK-REFERENCE.md`
- Dataset 1: `DATASET-1-GUIDE.md`
- Dataset 2: `DATASET-2-GUIDE.md`

### Notebooks:
- Pharmacy: `dataset1-pharmacy-analysis.ipynb`
- Wave: `dataset2-wave-analysis.ipynb`

### Libraries:
- [scikit-learn](https://scikit-learn.org/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [pandas](https://pandas.pydata.org/)

---

## 📝 Conclusion

This project provides a **comprehensive comparison** of two popular feature selection methods:

- **RFECV**: Best for final model optimization
- **MI**: Best for fast exploration

Both methods have their place in the ML pipeline. Use this framework to:
1. Understand trade-offs
2. Make informed decisions
3. Build better models
4. Save time and resources

**Remember:** The "best" method depends on your specific use case, constraints, and priorities! 🎯

---

**Project:** Data Mining Final Project  
**Topic:** Feature Selection Method Comparison  
**Methods:** RFECV vs Mutual Information  
**Model:** LightGBM  
**Year:** 2025
