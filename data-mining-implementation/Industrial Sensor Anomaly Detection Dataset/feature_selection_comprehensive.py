"""
Comprehensive Feature Selection Pipeline for SWAT & WADI Datasets
Dual-dataset analysis with cross-validation and transparent tracking
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import (
    SelectKBest, chi2, mutual_info_classif, f_classif,
    VarianceThreshold, RFE
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
import xgboost as xgb
import lightgbm as lgb

def run_feature_selection_pipeline(df, dataset_name, target_col='Normal/Attack'):
    """
    Run complete feature selection pipeline on a dataset with transparent tracking.
    Returns selected features from all methods and detailed reduction stats.
    """
    
    print(f"\n" + "="*90)
    print(f"FEATURE SELECTION PIPELINE - {dataset_name} DATASET")
    print("="*90)
    
    # Track feature reduction
    reduction_log = {}
    
    # ============ PREPROCESSING ============
    print(f"\n[{dataset_name}] Preprocessing...")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode target
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    feature_names = X.columns.tolist()
    reduction_log['0_original'] = len(feature_names)
    print(f"  ✓ Original features: {len(feature_names)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  ✓ Train/Test split: {X_train.shape} / {X_test.shape}")
    print(f"  ✓ Attack ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")
    
    # ============ STEP 1: VARIANCE THRESHOLD ============
    print(f"\n[{dataset_name} - STEP 1] Variance Threshold Filtering...")
    
    var_threshold = 0.01
    selector_var = VarianceThreshold(threshold=var_threshold)
    selector_var.fit(X_train_scaled)
    
    variance_scores = pd.DataFrame({
        'feature': feature_names,
        'variance': selector_var.variances_
    }).sort_values('variance', ascending=False)
    
    high_var_features = variance_scores[variance_scores['variance'] > var_threshold]['feature'].tolist()
    low_var_removed = len(feature_names) - len(high_var_features)
    reduction_log['1_variance_filtered'] = len(high_var_features)
    
    print(f"  ✓ Kept: {len(high_var_features)} features (variance > {var_threshold})")
    print(f"  ✓ Removed: {low_var_removed} low-variance features")
    print(f"  ✓ Reduction: {len(feature_names)} → {len(high_var_features)} features")
    
    # Update feature list for subsequent methods
    active_features = high_var_features
    
    # ============ STEP 2: FILTER METHODS ============
    print(f"\n[{dataset_name} - STEP 2] Running Filter Methods...")
    
    k_best = min(30, len(active_features))
    selected_features = {}
    
    # Chi-Square
    X_train_nonneg = X_train_scaled - X_train_scaled.min() + 1e-6
    selector_chi2 = SelectKBest(chi2, k=k_best)
    selector_chi2.fit(X_train_nonneg, y_train)
    chi2_features = [feature_names[i] for i in selector_chi2.get_support(indices=True)]
    selected_features['chi2'] = chi2_features
    print(f"  ✓ Chi-Square: {len(chi2_features)} features")
    
    # Mutual Information
    selector_mi = SelectKBest(mutual_info_classif, k=k_best)
    selector_mi.fit(X_train_scaled, y_train)
    mi_features = [feature_names[i] for i in selector_mi.get_support(indices=True)]
    selected_features['mutual_info'] = mi_features
    print(f"  ✓ Mutual Information: {len(mi_features)} features")
    
    # ANOVA F-test
    selector_f = SelectKBest(f_classif, k=k_best)
    selector_f.fit(X_train_scaled, y_train)
    f_features = [feature_names[i] for i in selector_f.get_support(indices=True)]
    selected_features['anova_f'] = f_features
    print(f"  ✓ ANOVA F-test: {len(f_features)} features")
    
    # Correlation
    corr_scores = [np.abs(np.corrcoef(X_train[col], y_train)[0, 1]) for col in feature_names]
    corr_df = pd.DataFrame({'feature': feature_names, 'corr': corr_scores}).sort_values('corr', ascending=False)
    corr_features = corr_df.head(k_best)['feature'].tolist()
    selected_features['correlation'] = corr_features
    print(f"  ✓ Correlation: {len(corr_features)} features")
    
    # ============ STEP 3: WRAPPER METHOD (RFE - Optimized) ============
    print(f"\n[{dataset_name} - STEP 3] Running Wrapper Method (RFE)...")
    
    rf_estimator = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rf_estimator, n_features_to_select=k_best, step=10)
    rfe.fit(X_train_scaled, y_train)
    rfe_features = [feature_names[i] for i in np.where(rfe.support_)[0]]
    selected_features['rfe'] = rfe_features
    print(f"  ✓ RFE: {len(rfe_features)} features")
    
    # ============ STEP 4: EMBEDDED METHODS ============
    print(f"\n[{dataset_name} - STEP 4] Running Embedded Methods...")
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X_train_scaled, y_train)
    rf_importance = pd.DataFrame({'feature': feature_names, 'importance': rf.feature_importances_})
    rf_features = rf_importance.nlargest(k_best, 'importance')['feature'].tolist()
    selected_features['random_forest'] = rf_features
    print(f"  ✓ Random Forest: {len(rf_features)} features")
    
    # XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, 
                                   random_state=42, eval_metric='logloss', 
                                   scale_pos_weight=scale_pos_weight, n_jobs=-1)
    xgb_model.fit(X_train_scaled, y_train)
    xgb_importance = pd.DataFrame({'feature': feature_names, 'importance': xgb_model.feature_importances_})
    xgb_features = xgb_importance.nlargest(k_best, 'importance')['feature'].tolist()
    selected_features['xgboost'] = xgb_features
    print(f"  ✓ XGBoost: {len(xgb_features)} features")
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, 
                                    random_state=42, class_weight='balanced', n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train_scaled, y_train)
    lgb_importance = pd.DataFrame({'feature': feature_names, 'importance': lgb_model.feature_importances_})
    lgb_features = lgb_importance.nlargest(k_best, 'importance')['feature'].tolist()
    selected_features['lightgbm'] = lgb_features
    print(f"  ✓ LightGBM: {len(lgb_features)} features")
    
    # ============ STEP 5: LASSO ============
    print(f"\n[{dataset_name} - STEP 5] Running LASSO (L1 Regularization)...")
    
    lasso = LogisticRegressionCV(penalty='l1', solver='liblinear', cv=5, random_state=42, 
                                  class_weight='balanced', max_iter=1000)
    lasso.fit(X_train_scaled, y_train)
    lasso_importance = pd.DataFrame({'feature': feature_names, 'coef': np.abs(lasso.coef_[0])})
    lasso_features = lasso_importance[lasso_importance['coef'] > 0].nlargest(k_best, 'coef')['feature'].tolist()
    selected_features['lasso'] = lasso_features
    print(f"  ✓ LASSO: {len(lasso_features)} features (non-zero coefficients)")
    
    # ============ ENSEMBLE VOTING ============
    print(f"\n[{dataset_name} - STEP 6] Ensemble Voting...")
    
    all_selected = sum(selected_features.values(), [])
    feature_votes = Counter(all_selected)
    voting_df = pd.DataFrame({'feature': list(feature_votes.keys()), 
                               'votes': list(feature_votes.values())}).sort_values('votes', ascending=False)
    
    num_methods = len(selected_features)
    print(f"  ✓ Total methods: {num_methods}")
    print(f"  ✓ Features with max votes ({num_methods}/{num_methods}): {(voting_df['votes'] == num_methods).sum()}")
    print(f"  ✓ Features with ≥7 votes: {(voting_df['votes'] >= 7).sum()}")
    print(f"  ✓ Features with ≥6 votes: {(voting_df['votes'] >= 6).sum()}")
    print(f"  ✓ Features with ≥5 votes: {(voting_df['votes'] >= 5).sum()}")
    print(f"  ✓ Features with ≥4 votes: {(voting_df['votes'] >= 4).sum()}")
    
    reduction_log['2_voting_available'] = len(voting_df)
    
    # Summary
    print(f"\n" + "-"*90)
    print(f"[{dataset_name}] FEATURE REDUCTION SUMMARY:")
    print(f"  Step 0 - Original: {reduction_log['0_original']} features")
    print(f"  Step 1 - After Variance Filter: {reduction_log['1_variance_filtered']} features (removed {reduction_log['0_original'] - reduction_log['1_variance_filtered']})") 
    print(f"  Step 2-5 - Multi-method selection: {num_methods} methods × top-{k_best} = voting pool")
    print(f"  Step 6 - Ensemble voting: {reduction_log['2_voting_available']} unique features with votes")
    print("-"*90)
    
    return {
        'dataset_name': dataset_name,
        'selected_features': selected_features,
        'voting_df': voting_df,
        'reduction_log': reduction_log,
        'num_methods': num_methods,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names
    }
