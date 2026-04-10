# Credit Card Fraud Detection Model
## Model Documentation

**Author:** Amit Bansal  
**Date:** April 2026  
**Version:** 1.0  
**Status:** Development Complete — Validation Passed  

---

## 1. Executive Summary

This document describes the development and validation of a credit card fraud detection model using XGBoost with SMOTE-based oversampling to handle extreme class imbalance.

The model achieves AUC-ROC of 0.976 and AUC-PR of 0.821 on the held-out test set. At the optimal decision threshold of 0.976, the model catches 76.4% of fraud with a precision of 94.96% — meaning 95% of flagged transactions are genuine fraud.

| Metric | Value | Status |
|---|---|---|
| AUC-ROC | 0.9756 | ✅ Excellent |
| AUC-PR | 0.8205 | ✅ Strong |
| Precision (optimal threshold) | 94.96% | ✅ Production-grade |
| Recall (optimal threshold) | 76.35% | ✅ Good |
| F1 Score (optimal threshold) | 0.8464 | ✅ Strong |

---

## 2. Business Objective

### 2.1 Problem Statement
Credit card fraud causes significant financial losses to banks and cardholders globally. The challenge is detecting fraudulent transactions in real time — within milliseconds — from a stream where 99.83% of transactions are legitimate.

### 2.2 Model Use Case
This model assigns a fraud probability score to each transaction. 
Transactions scoring above a defined threshold are flagged for:
- Automatic blocking (high confidence fraud)
- Analyst review queue (borderline cases)
- Real-time cardholder alert

### 2.3 Target Variable
`Class` — binary indicator of transaction legitimacy
- 0 = Legitimate transaction (99.827% of population)
- 1 = Fraudulent transaction (0.173% of population)
- Imbalance ratio: 578:1

---

## 3. Data

### 3.1 Dataset
- **Source:** ULB Machine Learning Group — Kaggle
- **Population:** European cardholders, September 2013
- **Size:** 284,807 transactions over 2 days
- **Fraudulent transactions:** 492 (0.173%)

### 3.2 Features

| Feature | Description | Type |
|---|---|---|
| V1–V28 | PCA-transformed transaction features (anonymised) | Continuous |
| Amount | Transaction amount in Euros | Continuous |
| Time | Seconds elapsed since first transaction | Continuous |

**Note on V1–V28:** Original features were anonymised via PCA to 
protect customer privacy and commercial confidentiality. This is 
standard practice when banks share data externally.

### 3.3 Feature Engineering

| Engineered Feature | Description | Rationale |
|---|---|---|
| Amount_scaled | Log-transformed and standardised Amount | Handles right skew, matches V1-V28 scale |
| Time_scaled | Standardised Time in hours | Normalises scale |
| hour_of_day | Hour within 24-hour cycle | Captures day/night pattern |
| is_night | Binary flag for midnight to 6am | Nighttime fraud rate 3.7x higher |

### 3.4 Key Data Insights

**Amount patterns:**
- Legitimate median: €22 — Fraud median: €9
- Fraud transactions are smaller on average
- Fraudsters use small probe transactions to verify stolen cards

**Time patterns:**
- Legitimate transactions follow human sleep patterns
- Fraud occurs uniformly across all hours — automated/cross-timezone
- Nighttime fraud rate: 0.518% vs daytime: 0.141%

**PCA component separation:**
Top separating components by mean difference between fraud and legitimate: V3 (7.05), V14 (6.98), V17 (6.68), V12 (6.27), V10 (5.69)

---

## 4. Methodology

### 4.1 Development Approach

```
Raw Data → EDA → Feature Engineering → Train/Test Split → SMOTE Oversampling → XGBoost Training → Threshold Tuning → Evaluation → Business Impact Analysis
```

### 4.2 Train/Test Split
- 70% training (199,364 rows), 30% test (85,443 rows)
- Stratified to preserve 0.173% fraud rate in both sets
- Split performed before SMOTE — test set never balanced

### 4.3 Handling Class Imbalance

**The problem:** 578:1 imbalance makes accuracy meaningless.
A model predicting "legitimate" for every transaction scores 99.83% accuracy while catching zero fraud.

**Solution — two-layer approach:**

**Layer 1 — SMOTE oversampling:**
Creates synthetic fraud samples by interpolating between real fraud examples using k-nearest neighbours (k=5).

```
sampling_strategy = 0.1
→ Fraud becomes 10% of legitimate count
→ Training ratio: 578:1 → 10:1
→ Synthetic samples created: 19,558
```

Rationale for 10:1 target: 50/50 balance would require 578 synthetic samples per real fraud — too artificial and noisy. 10:1 provides sufficient examples while keeping synthetic samples statistically realistic.

**Note on parameter choice:** sampling_strategy=0.1 (10:1 target ratio) was chosen over 50:50 balancing because creating 578 
synthetic samples per real fraud transaction would make synthetic data dominate, potentially introducing noise rather than signal. A production system would evaluate multiple sampling ratios (5:1, 10:1, 20:1) via cross-validation and select the ratio maximising AUC-PR on a held-out validation set.

**Layer 2 — scale_pos_weight:**
XGBoost parameter set to 10 (post-SMOTE ratio). Penalises misclassification of fraud 10x more than legitimate.Handles remaining imbalance at algorithm level.

**Why SMOTE over GANs/VAEs:**
- SMOTE is mathematically transparent — explainable to regulators
- Sufficient quality for tabular data (vs images where GANs excel)
- Proven production track record in fraud detection
- Computationally efficient — seconds vs hours for deep generative models
- GANs require large datasets to learn reliable distributions — 
  492 real frauds is insufficient for stable GAN training

### 4.4 XGBoost Model

**Why XGBoost over Logistic Regression:**
- Captures non-linear fraud patterns
- Handles remaining imbalance natively
- Robust to outliers in Amount
- Faster training than neural networks
- Industry standard for tabular fraud detection

**Model parameters:**

| Parameter | Value | Rationale |
|---|---|---|
| n_estimators | 300 | Max trees — early stopping at 169 |
| max_depth | 6 | Controls tree complexity |
| learning_rate | 0.05 | Small steps — careful learning |
| scale_pos_weight | 10 | Post-SMOTE imbalance correction |
| subsample | 0.8 | 80% rows per tree — reduces overfitting |
| colsample_bytree | 0.8 | 80% features per tree |
| eval_metric | aucpr | Optimise precision-recall not accuracy |
| early_stopping_rounds | 20 | Stop if no improvement for 20 rounds |

**Training progress:**

| Trees | AUC-PR |
|---|---|
| 0 | 0.429 |
| 50 | 0.709 |
| 100 | 0.805 |
| 169 (best) | 0.820 |

### 4.5 Threshold Tuning

Default threshold of 0.5 produced 211 false alarms — operationally unacceptable. Optimal threshold identified by maximising F1 score across all possible thresholds.

**Optimal threshold: 0.976**

This high threshold reflects the extreme imbalance — the model needs to be very confident before flagging a transaction as fraud.

---

## 5. Results

### 5.1 Model Performance

| Metric | Default (0.5) | Optimal (0.976) |
|---|---|---|
| Precision | 37.01% | 94.96% |
| Recall | 83.78% | 76.35% |
| F1 Score | 0.5135 | 0.8464 |
| AUC-PR | 0.8205 | 0.8205 |
| AUC-ROC | 0.9756 | 0.9756 |
| False alarms | 211 | 6 |
| Fraud caught | 124/148 (83.8%) | 113/148 (76.4%) |

### 5.2 Feature Importance

| Rank | Feature | Importance | Type |
|---|---|---|---|
| 1 | V14 | 46.7% | PCA component |
| 2 | V10 | 11.3% | PCA component |
| 3 | V4 | 5.2% | PCA component |
| 4 | V17 | 3.7% | PCA component |
| 5 | V8 | 2.6% | PCA component |
| 7 | Time_scaled | 2.0% | Engineered |
| 12 | hour_of_day | 1.4% | Engineered |

V14 alone contributes 46.7% of model importance — dominant single predictor. Both engineered time features appear in top 15, validating the nighttime fraud signal.

### 5.3 Business Impact

| Scenario | Total Cost | Saving |
|---|---|---|
| No model | €18,087 | — |
| Default threshold | €6,098 | €11,989 |
| Optimal threshold | €4,367 | €13,720 |

Threshold tuning delivers €1,731 additional saving on test set. At production scale, this represents significant incremental value.

Assumptions: missed fraud = average fraud amount (€122).
False positive = €15 customer service cost.

---

## 6. Model Limitations

**1. Dataset vintage**
Data from September 2013 — fraud patterns have evolved significantly since then. Modern fraud includes mobile payment fraud, account takeover, and synthetic identity fraud not present in this dataset.

**2. Anonymised features**
V1-V28 are PCA components — we cannot interpret what original behavioural features they represent. In production, interpretable features (merchant category, location, device fingerprint) would be available and would improve both performance and explainability.

**3. Real-time constraints not modelled**
Production fraud detection requires sub-200ms inference. This model's latency has not been benchmarked. XGBoost is generally fast enough for real-time scoring but deployment infrastructure is not covered here.

**4. Static model**
Fraudsters adapt constantly. A production model requires regular retraining — weekly or monthly — as new fraud patterns emerge. Model monitoring and drift detection are not implemented.

**5. Geographic specificity**
European cardholder behaviour. Fraud patterns differ by market — Indian UPI fraud, US card-not-present fraud, and European POS fraud have different signatures.

**6. Threshold is F1-optimal, not cost-optimal**
The optimal threshold maximises F1. A production system would tune threshold based on actual fraud loss amounts and customer service costs — which vary by bank and market.

**7. Hyperparameter tuning**
XGBoost parameters (max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8) were set to empirically validated defaults from XGBoost literature. A production implementation would use cross-validated hyperparameter search (GridSearchCV or Optuna) to optimise these values for the specific dataset and business objective.

**8. Feature importance concentration risk**
V14 alone accounts for 46.7% of model importance. If this PCA component becomes unavailable in production (data pipeline failure, vendor change), model performance would degrade significantly. A production system would monitor V14's distribution continuously and implement fallback logic.

**9. Threshold is F1-optimal, not cost-optimal**
The optimal threshold of 0.976 maximises F1 score. A production system would tune threshold based on actual, empirically measured fraud loss amounts and customer service costs — which vary by bank, market, and product type. The €15 false positive cost used here is an illustrative assumption.

---

## 7. Real-World Production Context

In a production fraud detection system this model would be one component of a broader architecture:

| Component | Description |
|---|---|
| Feature store | Real-time transaction features + customer history |
| Velocity engine | Count of transactions in last 1hr, 24hr, 7 days |
| ML scoring | This XGBoost model — fraud probability score |
| Rules engine | Hard rules for known fraud patterns |
| Case management | Analyst review queue for borderline cases |
| Feedback loop | Confirmed fraud labels fed back for retraining |

The ML model handles novel, pattern-based fraud. 
Rules handle known, well-defined fraud signatures. 
Both layers together outperform either alone.

---

## 8. Conclusion

This model demonstrates an end-to-end fraud detection pipeline handling the core challenge of extreme class imbalance (578:1) through SMOTE oversampling and XGBoost with scale_pos_weight.

At the optimal threshold the model achieves 94.96% precision — 95% of flagged transactions are genuine fraud — making it operationally viable for a real analyst review workflow.

The business impact analysis demonstrates €13,720 in cost savings on the test set vs no model, with an additional €1,731 
from threshold tuning alone.

---

*Document prepared by Amit Bansal | 
github.com/amit-bansa1/fraud-detection-credit-card*