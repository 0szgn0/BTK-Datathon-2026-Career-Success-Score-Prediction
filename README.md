<div align="center">

# BTK Datathon 2026
## Career Success Score Prediction

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-FFCC00?style=for-the-badge&logo=yandex&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-E74C3C?style=for-the-badge)
![TabNet](https://img.shields.io/badge/TabNet-Deep%20Learning-9B59B6?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-2ECC71?style=for-the-badge)

*End-to-end stacked ensemble pipeline for predicting student career success scores.*

</div>

---

## Overview

This repository contains the full solution pipeline developed for the **BTK Datathon 2026**. The objective was to predict the `career_success_score` of students based on their academic background, technical skills, soft skills, and online presence.

---

## 🏆 Performance & Robustness: The Anti-Overfit Architecture

In competitive data science, the true test of a model is how it performs on unseen private data. While many models suffer from "shake-down" due to overfitting the public leaderboard, this architecture demonstrated exceptional stability:

| Split | MSE |
|---|---|
| Public Leaderboard | 88.664360 |
| Private Leaderboard | 89.155655 |
| **Delta** | **~0.49 points** |

This minimal gap between public and private scores proves the highly robust, generalizable nature of the pipeline. It successfully avoids target leakage and relies on a strict validation strategy rather than public leaderboard probing.

---

## 🧠 Key Highlight: Advanced Feature Engineering

Instead of relying solely on raw data, this pipeline extracts deep, domain-specific, and statistical features to give the models a stronger learning representation.

### 1. Missingness as a Signal
Often, missing data isn't just "unknown" — it's a behavior. For instance, a missing GitHub score usually implies the student doesn't use GitHub at all.
* **Implementation:** Before any imputation took place, boolean `__isna` flags were generated for critical columns. This allowed the models to capture the "absence of activity" as an explicit predictive feature.

### 2. Domain-Driven Role Alignment
A manually curated mapping matrix was created to evaluate the semantic fit between a student's `department` and their `target_role`.
* **Implementation:** A binary `role_dept_match` feature was engineered (e.g., matching a "Statistics" student aiming for a "Data Scientist" role).

### 3. NLP & Latent Semantics on Mentor Feedback
Raw text data (`mentor_feedback_text`) was transformed into numerical representations using two complementary NLP approaches:
* **Sentiment Proxy:** Fast, keyword-based counting of positive and negative terms to capture the general tone of the feedback.
* **Latent Topic Modeling:** Applied **TF-IDF Vectorization** capped at 500 features, followed by **Truncated SVD** dimensionality reduction to compress the text into 5 dense latent dimensions, preserving semantic structure without exploding the feature space.

### 4. High-Order Interaction & Risk Flags
Hand-crafted metrics designed to capture higher-order constructs that raw columns cannot express individually:

| Feature | Description |
|---|---|
| `digital_footprint` | Composite score of LinkedIn, Portfolio, and weighted GitHub impact |
| `academic_risk` | Flags students with low attendance (<70%) and multiple failed courses |
| `tech_z_score_by_tier` | Normalizes technical performance relative to peers in the same university tier |
| `skill_chaos_score` | Cross-domain std across technical skills, soft skills, and GPA — identifies uneven development |
| `suspicious_profile_flag` | Top-20% tech score with zero digital presence — anomaly detector |
| `cv_conversion_rate` | Interview count / applications sent — measures job search efficiency |

---

## ⚙️ Model Architecture: Multi-Seed Stacked Ensemble

To achieve a highly stable private leaderboard score, a **Level-2 Stacking Regressor** was built using a multi-seed, 7-fold cross-validation approach.

```
┌─────────────────────────────────────────────────┐
│              Input Features (engineered)         │
└────────────────────┬────────────────────────────┘
                     │  7-Fold CV × 3 Seeds
       ┌─────────────┼──────────────┐
       ▼             ▼              ▼
  CatBoost       XGBoost        TabNet
  (raw cats)  (target-enc)    (scaled)
       │             │              │
       └─────────────┼──────────────┘
                     │  OOF predictions stacked
                     ▼
              Ridge Meta-Learner
                     │
                     ▼
            Final Prediction (avg across seeds)
```

### Zero-Leakage Validation Strategy
* Target encoding (`category_encoders`) and feature scaling (`StandardScaler`) are fitted **exclusively on the training fold** inside the CV loop — never on validation or test data.

### Level 1: Diverse Base Learners
| Model | Why |
|---|---|
| **CatBoost** | Natively handles raw categoricals; robust to outliers |
| **XGBoost** | Histogram-based trees on target-encoded features; fast and accurate |
| **TabNet** | Attention-based deep learning for tabular data; captures complex non-linearities |

### Level 2: Ridge Meta-Learner
* OOF predictions from all three base models are stacked and fed into a **Ridge Regression** meta-learner.
* L2 regularization prevents the meta-learner from over-relying on any single base model in cases of collinear OOF predictions.
* **Multi-Seed Averaging:** The entire pipeline runs across 3 random seeds; final predictions are averaged to minimize variance from fold randomness.

---

## 🛠️ Tech Stack

![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![numpy](https://img.shields.io/badge/numpy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)

| Category | Libraries |
|---|---|
| Data Processing | `pandas`, `numpy`, `scikit-learn`, `category_encoders` |
| NLP | `TfidfVectorizer`, `TruncatedSVD` |
| Modeling | `catboost`, `xgboost`, `pytorch-tabnet` |
| Hyperparameter Tuning | `optuna` (Bayesian optimization) |

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/0szgn0/BTK-Datathon-2026-Career-Success-Score-Prediction.git
   cd BTK-Datathon-2026-Career-Success-Score-Prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the notebook:
   ```bash
   jupyter notebook btk-datathon-26.ipynb
   ```

> **Note:** The dataset is downloaded automatically via `kagglehub`. Make sure your Kaggle API credentials are configured before running.

---

<div align="center">
  <sub>Built for BTK Datathon 2026 · <a href="https://github.com/0szgn0">@0szgn0</a></sub>
</div>
