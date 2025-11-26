# 📊 Visual Workflow Diagram

## 🔄 Complete Analysis Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE SELECTION COMPARISON                      │
│                   RFECV vs Mutual Information                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────┐
                    │   Select Dataset          │
                    └───────────────────────────┘
                                    │
                ┌───────────────────┴───────────────────┐
                │                                       │
                ▼                                       ▼
    ┌──────────────────────┐              ┌──────────────────────┐
    │   DATASET 1          │              │   DATASET 2          │
    │   Pharmacy Data      │              │   Wave Data          │
    │   (CSV Files)        │              │   (Excel Files)      │
    └──────────────────────┘              └──────────────────────┘
                │                                       │
                ▼                                       ▼
    ┌──────────────────────┐              ┌──────────────────────┐
    │ Feature Engineering  │              │ Auto-Detection       │
    │ - Temporal features  │              │ - Find numeric cols  │
    │ - Lag features       │              │ - Assign target      │
    │ - Rolling stats      │              │ - Standardize        │
    └──────────────────────┘              └──────────────────────┘
                │                                       │
                └───────────────────┬───────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │  Train/Test Split     │
                        │  (80% / 20%)          │
                        └───────────────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            │                                               │
            ▼                                               ▼
┌─────────────────────────┐                   ┌─────────────────────────┐
│   EXPERIMENT 1          │                   │   EXPERIMENT 2          │
│   RFECV                 │                   │   Mutual Information    │
└─────────────────────────┘                   └─────────────────────────┘
            │                                               │
            ▼                                               ▼
┌─────────────────────────┐                   ┌─────────────────────────┐
│ Step 1: RFECV Selection │                   │ Step 1: Calculate MI    │
│ - Cross-validation      │                   │ - Information theory    │
│ - Iterative elimination │                   │ - Statistical scores    │
│ - Auto optimal K        │                   │ - Fast computation      │
│ Time: ~3-5 min ⏳       │                   │ Time: ~10-30 sec ⚡     │
└─────────────────────────┘                   └─────────────────────────┘
            │                                               │
            ▼                                               ▼
┌─────────────────────────┐                   ┌─────────────────────────┐
│ Step 2: Train LightGBM  │                   │ Step 2: Select Top K    │
│ - Selected features     │                   │ - Highest MI scores     │
│ - Gradient boosting     │                   │ - Same K as RFECV       │
│ - 200 estimators        │                   └─────────────────────────┘
│ Time: ~20-30 sec        │                                 │
└─────────────────────────┘                                 ▼
            │                                   ┌─────────────────────────┐
            │                                   │ Step 3: Train LightGBM  │
            │                                   │ - Selected features     │
            │                                   │ - Gradient boosting     │
            │                                   │ - 200 estimators        │
            │                                   │ Time: ~20-30 sec        │
            │                                   └─────────────────────────┘
            │                                               │
            ▼                                               ▼
┌─────────────────────────┐                   ┌─────────────────────────┐
│ Predictions & Metrics   │                   │ Predictions & Metrics   │
│ - RMSE                  │                   │ - RMSE                  │
│ - MAE                   │                   │ - MAE                   │
│ - R²                    │                   │ - R²                    │
│ - Time                  │                   │ - Time                  │
└─────────────────────────┘                   └─────────────────────────┘
            │                                               │
            └───────────────────────┬───────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │   COMPARISON          │
                        │   Side-by-side        │
                        └───────────────────────┘
                                    │
                ┌───────────────────┴───────────────────┐
                │                                       │
                ▼                                       ▼
    ┌──────────────────────┐              ┌──────────────────────┐
    │   CSV Results        │              │   PNG Visualizations │
    │   - Metrics table    │              │   - 4-panel charts   │
    │   - Feature lists    │              │   - Scatter plots    │
    └──────────────────────┘              └──────────────────────┘
                │                                       │
                └───────────────────┬───────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │   DECISION            │
                        │   Which method wins?  │
                        └───────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │ RFECV Wins   │      │ MI Wins      │      │ Mixed/Tie    │
    │              │      │              │      │              │
    │ Use for:     │      │ Use for:     │      │ Use for:     │
    │ • Production │      │ • Exploration│      │ • Both!      │
    │ • Accuracy   │      │ • Speed      │      │ • Ensemble   │
    └──────────────┘      └──────────────┘      └──────────────┘
```

---

## 🔍 Method Comparison Matrix

```
┌────────────────────────────────────────────────────────────────────┐
│                    RFECV vs Mutual Information                     │
└────────────────────────────────────────────────────────────────────┘

╔═══════════════════╦═════════════════════╦═════════════════════════╗
║    Aspect         ║       RFECV         ║   Mutual Information    ║
╠═══════════════════╬═════════════════════╬═════════════════════════╣
║ Speed             ║ Slow ⏳ (3-5 min)   ║ Fast ⚡ (10-30 sec)     ║
║ Accuracy          ║ High ⭐⭐⭐⭐⭐        ║ Good ⭐⭐⭐⭐           ║
║ Interactions      ║ Captures ✅         ║ May miss ⚠️            ║
║ Model-agnostic    ║ No ❌               ║ Yes ✅                 ║
║ Auto optimal K    ║ Yes ✅              ║ No ❌ (manual)         ║
║ Cross-validation  ║ Built-in ✅         ║ No ❌                  ║
║ Interpretability  ║ Moderate 😐         ║ Statistical 📊         ║
║ Memory usage      ║ High 💾💾💾          ║ Low 💾                 ║
║ Best for          ║ Final model 🎯      ║ Exploration 🔍         ║
╚═══════════════════╩═════════════════════╩═════════════════════════╝
```

---

## 📊 Dataset Characteristics

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATASET 1: PHARMACY                         │
└─────────────────────────────────────────────────────────────────────┘

Type: Time Series (Transaction Logs)
Target: qty_out_sum (Product Demand)
Size: ~300K+ rows → Filtered to top 30 products

Features Created (15+):
┌─────────────────┬──────────────────────────────────────────┐
│ Category        │ Features                                 │
├─────────────────┼──────────────────────────────────────────┤
│ Temporal        │ day, month, day_of_week, week_of_year    │
│ Lag (Historical)│ qty_in_lag1, lag2, lag3, lag7            │
│                 │ qty_out_lag1, lag2, lag3, lag7           │
│ Rolling (7-day) │ qty_in_roll7, qty_out_roll7              │
│ Aggregations    │ qty_in_sum, qty_in_mean                  │
│                 │ value_in_sum, value_in_mean              │
└─────────────────┴──────────────────────────────────────────┘

Challenges:
• Data leakage risk ⚠️ (no future data!)
• Temporal dependencies
• Need proper train/test split (no shuffle)

Expected Winner: RFECV (captures lag interactions)


┌─────────────────────────────────────────────────────────────────────┐
│                          DATASET 2: WAVE                            │
└─────────────────────────────────────────────────────────────────────┘

Type: Physical Measurements (Excel)
Target: Auto-detected (last numeric column)
Size: Varies (6 Excel files)

Features:
┌─────────────────┬──────────────────────────────────────────┐
│ Category        │ Details                                  │
├─────────────────┼──────────────────────────────────────────┤
│ Auto-detected   │ All numeric columns                      │
│ Preprocessing   │ Standardized (mean=0, std=1)             │
│ Unknown names   │ Works without knowing columns!           │
└─────────────────┴──────────────────────────────────────────┘

Challenges:
• Unknown structure
• Need auto-detection
• Physical relationships between features

Expected Winner: Depends on feature independence
```

---

## 🎯 Decision Tree

```
                        Start
                          │
                          ▼
              ┌───────────────────────┐
              │ What's your priority? │
              └───────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
  ┌──────────┐      ┌──────────┐      ┌──────────┐
  │ Accuracy │      │  Speed   │      │   Both   │
  └──────────┘      └──────────┘      └──────────┘
        │                 │                 │
        ▼                 ▼                 ▼
  Use RFECV          Use MI           Run both,
  for final         for fast          then decide
  model           iterations         based on
                                      results
```

---

## 📈 Expected Performance Patterns

```
Dataset 1 (Pharmacy - Time Series):
════════════════════════════════════

RMSE:  RFECV ████████░░ 10-30
       MI    ██████████ 20-40

R²:    RFECV ████████░░ 0.6-0.8
       MI    ███████░░░ 0.5-0.7

Time:  RFECV ██████████ 200-300 sec
       MI    █░░░░░░░░░ 20-40 sec

Winner: RFECV (accuracy) vs MI (speed)


Dataset 2 (Wave - Physical):
═══════════════════════════

RMSE:  RFECV ████████░░ Varies
       MI    ████████░░ Similar

R²:    RFECV ████████░░ 0.5-0.9
       MI    ████████░░ 0.5-0.9

Time:  RFECV ██████████ 150-200 sec
       MI    █░░░░░░░░░ 10-20 sec

Winner: Often similar, MI faster
```

---

## 🏆 Success Metrics

```
✅ GOOD RESULTS:
┌─────────────────────────────────────┐
│ RMSE: 10-50 (Dataset 1)             │
│       Reasonable for scale (D2)     │
│ R²:   0.3-0.9 (NOT 1.0!)            │
│ MAE:  Similar to RMSE               │
│ Time: RFECV > MI (expected)         │
└─────────────────────────────────────┘

❌ WARNING SIGNS:
┌─────────────────────────────────────┐
│ RMSE: 0.0000 → Data leakage! 🚨     │
│ R²:   1.0000 → Too perfect! 🚨      │
│ R²:   <0 → Worse than baseline 🚨   │
│ All identical → Error in code 🚨    │
└─────────────────────────────────────┘
```

---

## 🔄 Iteration Workflow

```
First Run:
┌──────┐   ┌──────┐   ┌─────────┐   ┌──────────┐
│ Load │ → │ Prep │ → │ RFECV   │ → │ Evaluate │
│ Data │   │ Data │   │ + MI    │   │ Results  │
└──────┘   └──────┘   └─────────┘   └──────────┘
                                            │
                                            ▼
                                    ┌───────────────┐
                                    │ Good results? │
                                    └───────────────┘
                                            │
                            ┌───────────────┴───────────────┐
                            │                               │
                            ▼                               ▼
                       ┌─────────┐                   ┌────────────┐
                       │   YES   │                   │     NO     │
                       │ ✅ Done │                   │ Debug 🔧   │
                       └─────────┘                   └────────────┘
                                                            │
                                                            ▼
                                                    ┌────────────┐
                                                    │ Check for: │
                                                    │ - Leakage  │
                                                    │ - Errors   │
                                                    │ - NaN      │
                                                    └────────────┘
                                                            │
                                                            ▼
                                                    [Restart kernel]
                                                            │
                                                            ▼
                                                    [Run again]
```

---

**Use this diagram to understand the complete workflow! 🎯**
