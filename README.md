# Credit Card Fraud Detection Model

![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Model](https://img.shields.io/badge/Model-XGBoost-orange)

## Overview
End-to-end credit card fraud detection model handling extreme class imbalance (578:1) using SMOTE oversampling and XGBoost. Evaluated on Precision, Recall, and AUC-PR — not accuracy.

## Results

| Metric | Value | Notes |
|---|---|---|
| AUC-ROC | 0.9756 | Excellent discrimination |
| AUC-PR | 0.8205 | Strong (random = 0.17%) |
| Precision | 94.96% | 95% of alerts are real fraud |
| Recall | 76.35% | Catches 76% of all fraud |
| F1 Score | 0.8464 | At optimal threshold 0.976 |
| False alarms | 6 per 85,443 transactions | Operationally viable |

## The Core Challenge
**578:1 class imbalance** — 99.83% legitimate vs 0.17% fraud.
Accuracy is a useless metric here — predicting "legitimate" for everything scores 99.83% while catching zero fraud.

## Approach

```
Raw Data → EDA → Feature Engineering → Train/Test Split → SMOTE (578:1 → 10:1) → XGBoost → Threshold Tuning → Evaluation
```

**Imbalance handling — two layers:**
1. SMOTE oversampling — synthetic fraud samples (10:1 target ratio)
2. scale_pos_weight=10 — XGBoost penalises missed fraud 10x more

**Why not 50/50 balancing?**
Would require 578 synthetic samples per real fraud — too artificial. 10:1 provides sufficient examples while remaining statistically realistic.

## Key Findings
- **V14 dominates** — single PCA component accounts for 46.7% of model importance
- **Fraud is smaller, not larger** — median fraud €9 vs legitimate €22 (probe transactions to verify stolen cards)
- **Nighttime fraud 3.7x higher** — 0.518% vs 0.141% during day
- **Threshold tuning matters** — moving from 0.5 to 0.976 reduced false alarms from 211 to 6 while maintaining strong recall
- **Business impact** — €13,720 saving on test set vs no model

## Tech Stack
Python | XGBoost | scikit-learn | imbalanced-learn (SMOTE) | pandas | numpy | matplotlib | seaborn

## Project Structure
```
fraud-detection-credit-card/
├── data/                        # Download from Kaggle (see below)
├── notebooks/
│   ├── 01_eda.ipynb             # EDA, feature engineering
│   └── 02_model.ipynb           # SMOTE, XGBoost, evaluation
├── reports/
│   ├── model_documentation.md  # Full model document
│   └── *.png                   # Charts and visualisations
└── src/                        # Helper functions
```

## Dataset
Download from Kaggle and place in `data/` folder:
- Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- File: `creditcard.csv` (~144MB, excluded from repo)

## Full Documentation
[model_documentation.md](reports/model_documentation.md)

## Related Project
[Credit Risk PD Scorecard Model](https://github.com/amit-bansa1/credit-risk-pd-scorecard) 
— PD scorecard using WoE/IV and Logistic Regression (KS 49.2%, Gini 63.2%)

## Author
**Amit Bansal** — Manager, Decision Science at HSBC  
[GitHub](https://github.com/amit-bansa1) | 
[LinkedIn](https://www.linkedin.com/in/theamitbansal)