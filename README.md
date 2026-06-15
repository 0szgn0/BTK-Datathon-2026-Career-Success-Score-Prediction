# 🚀 BTK Datathon 2026: Career Success Score Prediction

This repository contains the end-to-end pipeline developed for the **BTK Datathon 2026**. The goal of the competition was to predict the `career_success_score` of students based on their academic background, technical skills, soft skills, and online presence.

## 🏆 Performance & Robustness: The Anti-Overfit Architecture

In competitive data science, the true test of a model is how it performs on unseen private data. While many models suffer from "shake-down" due to overfitting the public leaderboard, this architecture demonstrated exceptional stability:

* **Public MSE:** 88.66
* **Private MSE:** 89.155
* **Delta:** ~0.49 points

This minimal gap between public and private scores proves the highly robust, generalizable nature of the pipeline. It successfully avoids target leakage and relies on a strict validation strategy rather than public leaderboard probing.

---

## 🧠 Key Highlight: Advanced Feature Engineering

Instead of relying solely on raw data, this pipeline extracts deep, domain-specific, and statistical features to give the models a stronger learning representation.

### 1. Missingness as a Signal
Often, missing data isn't just "unknown"—it's a behavior. For instance, a missing GitHub score usually implies the student doesn't use GitHub at all. 
* **Implementation:** Before any imputation took place, boolean `__isna` flags were generated for critical columns. This allowed the models to capture the "absence of activity" as an explicit predictive feature.

### 2. Domain-Driven Role Alignment
A manually curated mapping matrix was created to evaluate the semantic fit between a student's `department` and their `target_role`. 
* **Implementation:** A binary `role_dept_match` feature was engineered (e.g., matching a "Statistics" student aiming for a "Data Scientist" role).

### 3. NLP & Latent Semantics on Mentor Feedback
Raw text data (`mentor_feedback_text`) was transformed into numerical representations using two complementary NLP approaches:
* **Sentiment Proxy:** Fast, keyword-based counting of positive (e.g., *başarılı*, *potansiyel*) and negative terms to capture the general tone.
* **Latent Topic Modeling:** Applied **TF-IDF Vectorization** capped at 500 features, followed by **Truncated SVD** dimensionality reduction to compress the text into 5 dense latent dimensions, preserving deep semantic meanings without exploding the feature space.

### 4. High-Order Interaction & Risk Flags
Hand-crafted metrics designed to capture higher-order constructs that raw columns cannot express individually:
* `digital_footprint`: A composite score of LinkedIn, Portfolio, and GitHub impact.
* `academic_risk`: Flags students with low attendance (<70%) coupled with multiple failed courses.
* `tech_z_score_by_tier`: Normalizes a student's technical performance relative to peers in their specific university tier, isolating individual merit from institutional advantage.
* `skill_chaos_score`: Calculates the cross-domain standard deviation across technical skills, soft skills, and GPA to identify candidates with uneven development.

---

## ⚙️ Model Architecture: Multi-Seed Stacked Ensemble

To achieve the highly stable Private Leaderboard score, a **Level-2 Stacking Regressor** was built using a multi-seed, 7-fold Cross-Validation approach.

### The Zero-Leakage Validation Strategy
* **Strict Isolation:** Target encoding (`category_encoders`) and feature scaling (`StandardScaler`) were strictly fitted **inside** the CV loop on the training folds only. This guarantees zero data leakage into the validation or test sets.

### Level 1: Diverse Base Learners
Three fundamentally different algorithms were trained to capture diverse patterns in the data:
1. **CatBoost:** Natively handles raw categoricals, robust to outliers.
2. **XGBoost:** Trained on target-encoded features utilizing the highly efficient histogram-based tree method.
3. **TabNet (PyTorch):** An attentive deep learning architecture for tabular data, capturing complex non-linear relationships.

### Level 2: Ridge Meta-Learner (Stacking)
* Out-of-fold (OOF) predictions from the base models were stacked and fed into a **Ridge Regression** meta-learner. 
* L2 regularization (Ridge) was explicitly chosen to prevent the meta-learner from over-relying on any single base model in case of collinear OOF predictions.
* **Multi-Seed Averaging:** The entire pipeline is executed across 3 different random seeds, and the final predictions are averaged. This drastically minimizes the variance introduced by the randomness of fold splits.

---

## 🛠️ Tech Stack
* **Data Processing & Engineering:** `pandas`, `numpy`, `scikit-learn`, `category_encoders`
* **NLP:** `TfidfVectorizer`, `TruncatedSVD`
* **Modeling:** `catboost`, `xgboost`, `pytorch_tabnet`

## 🚀 How to Run

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
