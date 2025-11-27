# Feature Selection Comparison - Final Report

## Executive Summary

Comprehensive comparison of **RFECV** vs **Mutual Information** feature selection methods across two distinct datasets using Ridge Regression (α=10.0, L2 regularization for overfitting prevention).

---

## Dataset 1: Pharmacy Transaction Volume

**Context**: Time series pharmacy data (2021-2023), predicting daily product volume.

### Data Characteristics
- **Samples**: 22,975 daily transactions
- **Original Features**: 17 (temporal + lag + rolling statistics + EWMA)
- **Target Distribution**: Highly skewed (median=2, mean=89, max=12,073)
- **Challenge**: Long-tail distribution limits linear model effectiveness

### Results

| Method | Features | RMSE | R² | Training Time | p-value | Verdict |
|--------|----------|------|----|--------------:|---------|---------|
| **Baseline** | 17 | 535.06 | 0.6660 | 0.0065s | - | - |
| **RFECV** | 12 (29% ↓) | 534.91 | 0.6662 | 0.0032s | **<0.0001** ✅ | Significant |
| **Mutual Info** | 12 (29% ↓) | **534.11** | **0.6672** | 0.0046s | **<0.0001** ✅ | Significant |

### Key Findings

**Performance**:
- Mutual Information: **Best accuracy** (RMSE 534.11, +0.18% improvement)
- Both methods: Statistically significant (p<0.0001)
- Improvement: **Marginal** due to already-informative features + skewed distribution

**Efficiency**:
- RFECV: 2.07x faster training, but 3.4s selection overhead
- Mutual Information: 1.41x faster training, 1.8s selection cost
- **Verdict**: Mutual Info **53% faster** overall (1.8s vs 3.4s)

**Feature Reduction**: 29.4% (17→12) without meaningful accuracy loss

---

## Dataset 2: Wave Measurement Data

**Context**: Oceanographic wave measurements (6 Excel files), auto-detected features.

### Data Characteristics
- **Samples**: 8,736 wave measurements (after NaN cleaning)
- **Original Features**: 67 (auto-detected numeric columns)
- **Target Distribution**: Normally distributed (median=199, mean=190, std=69)
- **Challenge**: High dimensionality (67 features) with unknown domain knowledge

### Results

| Method | Features | RMSE | R² | Training Time | p-value | Verdict |
|--------|----------|------|----|--------------:|---------|---------|
| **Baseline** | 67 | 56.20 | 0.3516 | 0.0089s | - | - |
| **RFECV** | 13 (81% ↓) | 56.21 | 0.3514 | 0.0020s | 0.7167 | Not significant |
| **Mutual Info** | 13 (81% ↓) | **56.16** | **0.3524** | 0.0021s | 0.0828 | Not significant |

### Key Findings

**Performance**:
- Mutual Information: Best accuracy (marginal +0.06% RMSE improvement)
- **Neither method** achieves statistical significance (p>0.05)
- Improvement: **Minimal** (<0.1%), suggests baseline already optimal

**Efficiency**:
- RFECV: 4.46x faster training, 4.1s selection overhead
- Mutual Information: 4.29x faster training, 2.2s selection cost
- **Verdict**: Mutual Info **45% faster** overall (2.2s vs 4.1s)

**Feature Reduction**: **Dramatic** 80.6% (67→13) with NO accuracy loss

---

## Comparative Analysis: RFECV vs Mutual Information

### RFECV (Recursive Feature Elimination)

**Strengths**:
- Model-aware: Considers feature interactions within Ridge framework
- More accurate selection when features have complex dependencies
- Optimal for final model deployment

**Weaknesses**:
- **Computationally expensive**: 2-4 seconds vs 0.003-2 seconds
- Requires multiple model fits (5-fold CV × iterations)
- Not practical for iterative feature engineering

**Best use case**: Final optimization when computational budget allows

---

### Mutual Information

**Strengths**:
- **Extremely fast**: 45-53% faster than RFECV
- Detects non-linear dependencies (information-theoretic)
- Model-agnostic (works with any estimator)
- Suitable for high-dimensional data

**Weaknesses**:
- Evaluates features independently (may miss interactions)
- Slightly less accurate in Dataset 1 (0.18% vs 0.03% improvement)

**Best use case**: Rapid screening, exploratory analysis, large-scale features (>100)

---

## Cross-Dataset Insights

### Feature Selection Effectiveness

**Dataset 1 (Pharmacy)**: 
- ✅ Feature selection **works** (p<0.0001, statistically significant)
- Improvement **marginal** (0.18%) due to skewed data + already-curated features
- 29% feature reduction valuable for interpretability

**Dataset 2 (Wave)**:
- ⚠️ Feature selection **not significant** (p>0.05)
- Improvement **negligible** (0.06%)
- **Massive** 81% feature reduction without harm → most features redundant

### Model Simplification Benefit

**Key insight**: Even when accuracy improvement is minimal, feature reduction provides:
1. **Interpretability**: Fewer features easier to explain (12 vs 17, 13 vs 67)
2. **Training efficiency**: 1.4-4.5x faster
3. **Deployment simplicity**: Reduced feature engineering pipeline
4. **Overfitting prevention**: Ridge + feature selection = double protection

---

## Recommendations

### When to Use RFECV
- ✅ Final model optimization (production deployment)
- ✅ Feature interactions are critical
- ✅ Computational budget allows (offline training)
- ✅ Need maximum statistical rigor

### When to Use Mutual Information
- ✅ **Rapid feature screening** (recommended default)
- ✅ High-dimensional data (>50 features)
- ✅ Iterative feature engineering workflow
- ✅ Non-linear relationships suspected
- ✅ **Time-constrained environments**

### Hybrid Approach (Recommended)
1. **Start**: Mutual Information for fast exploration
2. **Validate**: RFECV for final model refinement
3. **Compare**: Consensus features = highest confidence

---

## Limitations

1. **Small Improvements**: <0.2% accuracy gain suggests features already well-optimized
2. **Model-Specific**: Results specific to Ridge Regression; tree-based models may differ
3. **Data Quality**: Dataset 2 has extensive NaN (51% missing values), limiting analysis
4. **Statistical Power**: Dataset 2 p-values near threshold (0.08) suggest borderline significance

---

## Conclusion

### Dataset 1 (Pharmacy): ✅ **Feature Selection Validated**
- Both methods statistically significant (p<0.0001)
- Mutual Information **wins** (faster + better accuracy)
- 29% feature reduction + marginal accuracy gain
- **Recommendation**: Use Mutual Information for production

### Dataset 2 (Wave): ⚠️ **Feature Selection Not Necessary**
- No statistical significance (p>0.05)
- **Massive** 81% feature reduction with no harm
- Most features redundant (67→13 with no accuracy loss)
- **Recommendation**: Use feature selection **only for simplicity**, not accuracy

### Overall Winner: 🏆 **Mutual Information**
- **45-53% faster** than RFECV
- Comparable or better accuracy across both datasets
- Practical for real-world iterative workflows
- **Best default choice** for feature screening

### Pragmatic Strategy
For production ML pipelines:
1. **Always start** with Mutual Information (fast screening)
2. **Optionally validate** with RFECV if time permits
3. **Monitor p-values**: Ignore feature selection if p>0.05
4. **Prioritize simplicity**: Even marginal gains justify feature reduction for interpretability

---

*Analysis completed with Ridge Regression (α=10.0), 80/20 train-test split, paired t-test significance testing (α=0.05)*
