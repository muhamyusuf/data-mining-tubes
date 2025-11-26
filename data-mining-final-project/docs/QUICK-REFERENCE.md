# 🎯 Feature Selection Comparison - Quick Reference

## 📊 Project Summary

Comparison antara **RFECV** vs **Mutual Information** untuk 2 dataset berbeda menggunakan **LightGBM**.

---

## 🔬 Metode Comparison

### RFECV (Recursive Feature Elimination with CV)

**How It Works:**
1. Start with all features
2. Train model and rank features by importance
3. Remove least important feature
4. Repeat until optimal set found
5. Use cross-validation for robustness

**Pros:**
- ✅ Finds optimal feature subset automatically
- ✅ Uses model feedback (iterative)
- ✅ Cross-validation reduces overfitting
- ✅ Good for feature interactions

**Cons:**
- ❌ Computationally expensive
- ❌ Slow for large feature sets
- ❌ Model-dependent results

**Best For:**
- Final model optimization
- When accuracy is critical
- Complex feature relationships
- Production deployments

---

### Mutual Information (MI)

**How It Works:**
1. Calculate statistical dependency between each feature and target
2. Rank features by MI score
3. Select top K features
4. Done!

**Pros:**
- ✅ Very fast computation
- ✅ Captures non-linear relationships
- ✅ Model-agnostic
- ✅ Based on information theory

**Cons:**
- ❌ Assumes feature independence
- ❌ Less interpretable than SHAP
- ❌ No feature interactions considered

**Best For:**
- Initial feature exploration
- Fast iterations
- Large feature spaces
- Baseline comparisons

---

## 📂 Dataset Characteristics

### Dataset 1: Pharmacy Transactions
- **Type:** Time series (transaction logs)
- **Format:** 6 CSV files (2021-2023, A2021-A2023)
- **Target:** `qty_out_sum` (product demand)
- **Features:** 15+ engineered features
  - Temporal: day, month, day_of_week, week_of_year
  - Lags: qty_in/out_lag1, lag2, lag3, lag7
  - Rolling: qty_in/out_roll7
  - Aggregations: sum, mean values
- **Challenge:** Temporal dependencies, data leakage risk
- **Size:** ~300K+ rows (filtered to top 30 products)

### Dataset 2: Wave Parameters
- **Type:** Physical measurements
- **Format:** 6 Excel files
- **Target:** Auto-detected (last numeric column)
- **Features:** Auto-detected (all numeric columns)
- **Challenge:** Unknown structure, need auto-detection
- **Size:** Varies by files

---

## 🎯 Expected Outcomes

### Performance Metrics

| Scenario | RMSE | R² | Time | Winner |
|----------|------|-----|------|--------|
| RFECV better accuracy | Lower | Higher | Slower | RFECV for production |
| MI faster | Similar | Similar | Much faster | MI for iterations |
| Similar results | ~Same | ~Same | MI faster | MI wins on efficiency |

### Decision Matrix

| Priority | Accuracy > Speed | Speed > Accuracy | Need Interpretability |
|----------|-----------------|------------------|---------------------|
| **Method** | RFECV | MI | RFECV or SHAP |
| **Use Case** | Production model | Exploration | Research/Reporting |

---

## 📈 Interpreting Results

### Good Results ✅
- RMSE: Reasonable for data scale (not 0.0000)
- R²: 0.3-0.9 (not exactly 1.0)
- Methods show different strengths
- Feature agreement 40-80%

### Warning Signs ⚠️
- RMSE = 0.0000 → Data leakage!
- R² = 1.0000 → Too perfect, check leakage
- All methods identical → Something wrong
- Negative R² → Model worse than baseline

### Data Leakage Checklist
- [ ] No future data in features (`shift(-1)`)
- [ ] No target in feature list
- [ ] Proper train/test split
- [ ] Lag features use past only
- [ ] No look-ahead bias

---

## 🏆 Winner Criteria

### RFECV Wins If:
1. **Lower RMSE** than MI (by >5%)
2. **Higher R²** than MI (by >2%)
3. **Better feature interactions** captured
4. Time difference is acceptable

**Action:** Use RFECV features for final model

### MI Wins If:
1. **Similar or better RMSE** to RFECV
2. **Similar or better R²** to RFECV
3. **Much faster** (2x+ speedup)
4. Good enough accuracy

**Action:** Use MI for fast iterations, validate with RFECV occasionally

### Mixed Results:
- Compare on your priority metric (RMSE vs Speed)
- Consider ensemble of both feature sets
- Use MI for exploration, RFECV for final

---

## 🔍 Feature Selection Insights

### Common Agreement (High %)
- Both methods see same important features
- These features are robustly important
- Safe to use for modeling

### Low Agreement (< 40%)
- Methods have different perspectives
- RFECV captures interactions
- MI captures independence
- Consider using union of both

### Feature Types

| Feature Type | RFECV Preference | MI Preference |
|-------------|-----------------|---------------|
| Highly correlated | May drop redundant | Ranks all high |
| Interactions | Captures well | May miss |
| Non-linear | Good if model supports | Captures well |
| Independent | Ranks normally | Ranks well |

---

## 💻 Quick Start Commands

### Setup
```powershell
cd "c:\Users\muham\OneDrive\Desktop\data-mining-uas\data-mining-final-project"
uv add pandas numpy scikit-learn lightgbm matplotlib seaborn openpyxl jupyter ipykernel
```

### Run Notebooks
```powershell
uv run jupyter notebook
```

### Or Use VS Code
1. Open folder in VS Code
2. Select Python from UV environment
3. Open `.ipynb` files
4. Run All Cells

---

## 📊 Output Files Guide

### CSV Results
- **columns:** Method, RMSE, MAE, R2, Time, Num Features
- **rows:** RFECV, Mutual Information
- **usage:** Import to Excel/Python for analysis

### PNG Visualizations
- **4-panel comparison:** RMSE, R², Time, MAE
- **Scatter plots:** Actual vs Predicted (Dataset 2)
- **usage:** Include in reports/presentations

---

## 🎓 Learning Objectives

After completing this project, you'll understand:

1. ✅ How RFECV works (iterative elimination)
2. ✅ How Mutual Information works (statistical dependency)
3. ✅ When to use each method
4. ✅ How to interpret comparison results
5. ✅ Feature engineering for time series
6. ✅ Auto-detection techniques
7. ✅ Avoiding data leakage
8. ✅ LightGBM for regression

---

## 🚀 Next Steps

### After Running Notebooks:

1. **Analyze Results**
   - Check CSV files for metrics
   - View PNG visualizations
   - Compare methods

2. **Determine Winner**
   - Based on your priority (accuracy vs speed)
   - Check feature agreement
   - Validate results make sense

3. **Production Deployment**
   - Save selected features
   - Retrain on full data
   - Monitor performance

4. **Further Experiments**
   - Try SHAP feature selection
   - Compare with other models (XGBoost, Random Forest)
   - Hyperparameter tuning

---

## 📚 References

### Notebooks
- `dataset1-pharmacy-analysis.ipynb` - Pharmacy demand prediction
- `dataset2-wave-analysis.ipynb` - Wave parameter prediction

### Documentation
- `README-FEATURE-SELECTION.md` - Full documentation
- `DATASET-1-GUIDE.md` - Dataset 1 specifics
- `DATASET-2-GUIDE.md` - Dataset 2 specifics

### Libraries
- [scikit-learn RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)
- [scikit-learn MI](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html)
- [LightGBM](https://lightgbm.readthedocs.io/)

---

## ✅ Success Criteria

Project is successful when:
- [x] Both notebooks run without errors
- [x] Results make sense (RMSE > 0, R² < 1)
- [x] Clear winner or informed decision
- [x] Outputs saved (CSV + PNG)
- [x] Understanding of trade-offs

---

**Created for:** Data Mining Final Project  
**Purpose:** Feature Selection Method Comparison  
**Tools:** Python, UV, Jupyter, scikit-learn, LightGBM  
**Author:** Student Project 2025
