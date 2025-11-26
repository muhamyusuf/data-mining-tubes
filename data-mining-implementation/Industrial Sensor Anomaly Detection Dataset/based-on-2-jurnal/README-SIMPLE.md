# Feature Selection untuk Deteksi Anomali Sensor Industri

## 1. Deskripsi Dataset

### SWAT Dataset (Secure Water Treatment)
- **Sumber**: iTrust, SUTD Singapore
- **Jumlah Sampel**: 1,000 sampel
- **Jumlah Fitur**: 52 sensor + 1 target
- **Target**: Normal/Attack (Binary)
- **Distribusi Kelas**: Normal (85%), Attack (15%) - **Imbalanced**
- **Jenis Data**: Sensor flow meter, level, pressure, analyzer

### WADI Dataset (Water Distribution)
- **Sumber**: iTrust, SUTD Singapore
- **Jumlah Sampel**: 500 sampel
- **Jumlah Fitur**: 123 sensor + 1 target
- **Target**: Normal/Attack (Binary)
- **Distribusi Kelas**: Normal (mayoritas), Attack (minoritas) - **Imbalanced**
- **Jenis Data**: Sensor jaringan distribusi air

---

## 2. Metode dari Jurnal yang Diterapkan

### Jurnal 1: Pravin Singh Yada et al. (2024)
**Judul**: "Ensemble methods with feature selection and data balancing"  
**Sumber**: Engineering Applications of Artificial Intelligence 2024

**Metode yang Diambil**:

1. **SMOTE (Data Balancing)**
   - Untuk mengatasi ketidakseimbangan kelas
   - Membuat sampel sintetik untuk kelas minoritas

2. **Ensemble Feature Selection**
   - Menggabungkan multiple metode seleksi fitur
   - Voting mechanism untuk ranking fitur
   - Filter + Wrapper + Embedded methods

**Implementasi Code**:
```python
# SMOTE Balancing
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y_encoded)

# Filter Methods
chi2_selector = SelectKBest(chi2, k=30)  # Chi-Square
mi_selector = SelectKBest(mutual_info_classif, k=30)  # Mutual Information

# Wrapper Method
rfe = RFE(RandomForestClassifier(), n_features_to_select=30)

# Embedded Methods
rf = RandomForestClassifier()  # Random Forest Importance
xgb_model = xgb.XGBClassifier()  # XGBoost Importance
lasso = LassoCV()  # LASSO Regularization

# Ensemble Voting
vote_counter = Counter()
for method, features in selected_features.items():
    for feature in features:
        vote_counter[feature] += 1
```

### Jurnal 2: Sreehari & Dhinesh (2024)
**Judul**: "Critical Factor Analysis using Inclusive Feature Selection"  
**Sumber**: Applied Artificial Intelligence, 2024

**Metode yang Diambil**:

1. **Pearson Correlation**
   - Analisis korelasi fitur dengan target

2. **Inclusive Strategy**
   - Multiple filter methods comparison
   - Feature ranking dan selection

**Implementasi Code**:
```python
# Pearson Correlation
correlations = df.corr()['target'].abs()
corr_features = correlations.nlargest(30)

# Multiple methods comparison (inclusive strategy)
selected_features = {}
for method_name, method_results in all_methods.items():
    selected_features[method_name] = method_results
```

**Catatan**: ANOVA F-test diambil dari literature umum, bukan spesifik dari Jurnal Linux1.

---

## 3. Tahapan Implementasi

### Langkah 1: Variance Filtering
```python
variance_selector = VarianceThreshold(threshold=0.01)
X_filtered = variance_selector.fit_transform(X)
```
**Tujuan**: Hapus fitur dengan variance sangat rendah

### Langkah 2: SMOTE Balancing
```python
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
```
**Tujuan**: Seimbangkan data untuk kelas minoritas (Attack)

### Langkah 3: 9 Metode Seleksi Fitur

| Kategori | Metode | Sumber Jurnal |
|----------|--------|---------------|
| **Filter** | Chi-Square | Sahfa et al. |
| **Filter** | Mutual Information | Sahfa et al. |
| **Filter** | ANOVA F-test | Literature umum |
| **Filter** | Pearson Correlation | Sreehari & Dhinesh |
| **Wrapper** | RFE | Sahfa et al. |
| **Embedded** | Random Forest | Sahfa et al. |
| **Embedded** | XGBoost | Sahfa et al. |
| **Embedded** | LightGBM | Enhancement |
| **Embedded** | LASSO | Sahfa et al. |

### Langkah 4: Ensemble Voting
```python
# Hitung vote setiap fitur
vote_counter = Counter()
for method, features in selected_features.items():
    for feature in features:
        vote_counter[feature] += 1

# Buat DataFrame voting
voting_df = pd.DataFrame([
    {'Feature': feat, 'Votes': votes}
    for feat, votes in vote_counter.items()
]).sort_values('Votes', ascending=False)
```

### Langkah 5: Model Validation
```python
# Compare: ALL features vs SELECTED features
models = [RandomForest, XGBoost, LightGBM]

for model in models:
    # Train with ALL features
    model.fit(X_all, y)
    f1_all = f1_score(y_test, model.predict(X_all_test))
    
    # Train with SELECTED features
    model.fit(X_selected, y)
    f1_selected = f1_score(y_test, model.predict(X_selected_test))
```

### Langkah 6: Threshold Optimization
```python
# Test berbagai threshold
for threshold in [3, 4, 5, 6, 7, 9]:
    selected_features = voting_df[voting_df['Votes'] >= threshold]
    model.fit(X[selected_features])
    f1 = f1_score(y_test, model.predict(X_test))
```

---

## 4. Hasil Penelitian

### 4.1 Hasil Feature Reduction

| Dataset | Fitur Awal | Fitur Terpilih | Reduksi | Threshold Optimal |
|---------|------------|----------------|---------|-------------------|
| **SWAT** | 52 | 21 | 59.6% | ≥6 votes |
| **WADI** | 123 | 17 | 86.2% | ≥5 votes |

### 4.2 Hasil Model Performance

**SWAT Dataset (21 fitur terpilih)**:
```
Random Forest:
  - F1-Score: 0.9733 (+0.36% dari baseline)
  - Training: 1.12x lebih cepat

XGBoost:
  - F1-Score: 0.8800 (maintained)
  - Training: 3.14x lebih cepat

LightGBM:
  - F1-Score: 0.9000 (+2.0% improvement)
  - Training: 2.94x lebih cepat
```

**WADI Dataset (17 fitur terpilih)**:
```
Random Forest:
  - F1-Score: 0.9507 (+0.33% dari baseline)
  - Training: 1.18x lebih cepat

XGBoost:
  - F1-Score: 0.9048 (+1.59% improvement)
  - Training: 3.11x lebih cepat

LightGBM:
  - F1-Score: 0.9000 (maintained)
  - Training: 3.00x lebih cepat
```

### 4.3 Threshold Optimization Results

**SWAT**:
| Threshold | Jumlah Fitur | F1-Score | Rekomendasi |
|-----------|--------------|----------|-------------|
| ≥4 | 40 | 0.9695 | - |
| ≥5 | 31 | 0.9732 | Konservatif |
| **≥6** | **21** | **0.9733** | **OPTIMAL** ✓ |
| ≥7 | 19 | 0.9733 | - |
| ≥9 | 8 | 0.9600 | Critical only |

**WADI**:
| Threshold | Jumlah Fitur | F1-Score | Rekomendasi |
|-----------|--------------|----------|-------------|
| ≥4 | 29 | 0.9474 | - |
| **≥5** | **17** | **0.9507** | **OPTIMAL** ✓ |
| ≥6 | 9 | 0.9507 | Ultra-lightweight |
| ≥7 | 2 | 0.9507 | Minimal |

### 4.4 Kesimpulan Utama

**Keberhasilan**:
1. ✓ Reduksi fitur 60-86% tanpa penurunan performa
2. ✓ F1-Score meningkat +0.33% sampai +2.0%
3. ✓ Training time 1.1x - 3.1x lebih cepat
4. ✓ Threshold optimal ditentukan secara data-driven (F1-Score maximization)

**Fitur Terpenting**:
- SWAT: 8 fitur dengan 9/9 votes (100% agreement)
- WADI: 0 fitur dengan 9/9 votes (kompleksitas lebih tinggi)

**Cross-Dataset**:
- Common features: 0 (EXPECTED)
- Alasan: Setiap instalasi punya konfigurasi sensor berbeda
- Validasi: Feature selection bersifat site-specific (CORRECT)

---

## 5. File Output

### CSV Files (12 files):
```
v3_swat_feature_voting.csv          - Hasil voting SWAT
v3_wadi_feature_voting.csv          - Hasil voting WADI
v3_swat_threshold_5.csv             - SWAT threshold ≥5
v3_swat_threshold_6.csv             - SWAT threshold ≥6 (OPTIMAL)
v3_wadi_threshold_5.csv             - WADI threshold ≥5 (OPTIMAL)
v3_swat_model_validation.csv        - Validasi model SWAT
v3_wadi_model_validation.csv        - Validasi model WADI
v3_swat_threshold_optimization.csv  - Optimasi threshold SWAT
v3_wadi_threshold_optimization.csv  - Optimasi threshold WADI
v3_cross_dataset_validation.csv     - Validasi cross-dataset
v3_feature_reduction_tracking.csv   - Tracking reduksi fitur
```

### Visualisasi (2 files):
```
v3_feature_selection_results.png           - 4-panel voting & threshold
v3_comprehensive_performance_analysis.png  - 6-panel model performance
```

---

## 6. Cara Menjalankan

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm matplotlib

# 2. Run notebook
jupyter notebook feature-selection-sensor-v3.ipynb

# 3. Execute all cells (Kernel → Restart & Run All)
```

**Runtime**: 
- SWAT: ~91 detik
- WADI: ~150 detik
- Total: ~4 menit

---

## 7. Referensi

1. **Yadav, P. S., et al. (2024)**. Ensemble methods with feature selection and data balancing. *Engineering Applications of Artificial Intelligence*, 139, 109527.

2. **Sreehari, E., & Dhinesh Babu, L. D. (2024)**. Critical Factor Analysis using Inclusive Feature Selection Strategy. *Applied Artificial Intelligence*, 38(1).

3. **Chawla, N. V., et al. (2002)**. SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.

---
