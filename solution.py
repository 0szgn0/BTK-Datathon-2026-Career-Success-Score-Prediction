import kagglehub
import pandas as pd
import numpy as np
import os
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import category_encoders as ce

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor

import warnings
warnings.filterwarnings('ignore')

path = kagglehub.competition_download('datathon-2026')

print("🚀 Pipeline started: data ingestion, feature engineering, and model training.")

# =============================================================================
# 1. DATA INGESTION
# Train and test sets are loaded separately. Target variable is isolated from
# training features, and both splits are concatenated for consistent preprocessing.
# =============================================================================
train_df = pd.read_csv(os.path.join(path, "train.csv"))
test_df  = pd.read_csv(os.path.join(path, "test_x.csv"))

y_train       = train_df['career_success_score']
train_features = train_df.drop(['career_success_score'], axis=1)
all_data       = pd.concat([train_features, test_df], axis=0).reset_index(drop=True)

# =============================================================================
# 2. DOMAIN-DRIVEN FEATURE: ROLE–DEPARTMENT ALIGNMENT
# Encodes semantic fit between a student's academic background and their
# target career role. A binary indicator (1 = aligned, 0 = misaligned)
# derived from a manually curated mapping of departments to viable roles.
# =============================================================================
ALIGN = {
    "Computer Engineering":          {"Backend Developer", "Frontend Developer", "Software Developer",
                                      "DevOps Engineer", "Cloud Engineer", "AI Engineer", "MLOps Engineer"},
    "Software Engineering":          {"Backend Developer", "Frontend Developer", "Software Developer",
                                      "DevOps Engineer", "Cloud Engineer"},
    "Statistics":                    {"Data Analyst", "Data Scientist", "AI Engineer",
                                      "MLOps Engineer", "Product Analyst"},
    "Mathematics":                   {"Data Analyst", "Data Scientist", "AI Engineer", "MLOps Engineer"},
    "Management Information Systems":{"Product Analyst", "Data Analyst"},
    "Industrial Engineering":        {"Product Analyst", "Data Analyst"},
    "Electrical Electronics Engineering": {"DevOps Engineer", "Cloud Engineer", "AI Engineer"}
}
all_data["role_dept_match"] = all_data.apply(
    lambda r: int(r["target_role"] in ALIGN.get(r["department"], set())), axis=1
)

# =============================================================================
# 3. MISSINGNESS AS A SIGNAL (indicator flags before imputation)
# Missing values in certain features carry predictive information on their own
# (e.g., a missing GitHub score often implies no GitHub activity at all).
# Binary missingness flags are created prior to any imputation so this signal
# is preserved rather than silently discarded.
# =============================================================================
missing_watch_cols = [
    "english_exam_score", "internship_duration_months", "portfolio_score",
    "github_avg_stars", "open_source_contribution_count",
    "linkedin_profile_score", "hr_interview_score"
]
for c in missing_watch_cols:
    if c in all_data.columns:
        all_data[f"{c}__isna"] = all_data[c].isna().astype(int)

# Students with both GitHub metrics missing likely have no public coding presence.
all_data["no_github_profile"] = (
    all_data["github_avg_stars"].isna() & all_data["open_source_contribution_count"].isna()
).astype(int)

# =============================================================================
# 4. IMPUTATION STRATEGY
# GitHub activity columns are zero-filled: absence of activity is a meaningful
# value in itself, not an unknown. Subjective scoring columns (exams, interviews,
# profiles) are filled with train-set medians to avoid data leakage.
# =============================================================================
zero_fill_cols = ['internship_duration_months', 'github_avg_stars', 'open_source_contribution_count']
for col in zero_fill_cols:
    all_data[col] = all_data[col].fillna(0)

median_fill_cols = ['english_exam_score', 'hr_interview_score', 'linkedin_profile_score', 'portfolio_score']
for col in median_fill_cols:
    train_median = train_df[col].median()
    all_data[col] = all_data[col].fillna(train_median)

# =============================================================================
# 5. CATEGORICAL ENCODING PREPARATION
# Categorical columns are cast to string before target encoding is applied
# inside the cross-validation loop, ensuring no leakage across folds.
# =============================================================================
cat_features = ['university_tier', 'target_role']
for col in cat_features:
    all_data[col] = all_data[col].astype(str)

# =============================================================================
# 6. NLP FEATURE EXTRACTION FROM MENTOR FEEDBACK
# Mentor feedback text is processed in two complementary ways:
#   (a) Sentiment proxy: keyword-based positive/negative word counts capture
#       the overall tone of the feedback with minimal overhead.
#   (b) Latent semantics: TF-IDF vectorization followed by Truncated SVD
#       compresses 500 token-level features into 5 dense latent dimensions,
#       preserving topic-level structure while reducing dimensionality.
# =============================================================================
positive_words = ['başarılı', 'potansiyel', 'harika', 'iyi', 'istekli',
                  'kararlılıkla', 'dikkat çekici', 'etkileyici', 'yüksek', 'mükemmel', 'güçlü']
negative_words = ['eksik', 'geliştirmeli', 'zayıf', 'ihtiyaç',
                  'zorlaştırabilir', 'düşük', 'pratik yapmanız', 'yetersiz']

all_data['feedback_lower'] = all_data['mentor_feedback_text'].fillna('').str.lower()
all_data['positive_feedback_count'] = all_data['feedback_lower'].apply(
    lambda x: sum(1 for word in positive_words if word in x)
)
all_data['negative_feedback_count'] = all_data['feedback_lower'].apply(
    lambda x: sum(1 for word in negative_words if word in x)
)

tfidf       = TfidfVectorizer(max_features=500)
tfidf_matrix = tfidf.fit_transform(all_data['feedback_lower'])
svd          = TruncatedSVD(n_components=5, random_state=42)
svd_matrix   = svd.fit_transform(tfidf_matrix)

svd_df   = pd.DataFrame(svd_matrix, columns=[f'feedback_svd_{i+1}' for i in range(5)])
all_data = pd.concat([all_data, svd_df], axis=1)

# =============================================================================
# 7. FEATURE ENGINEERING
# Hand-crafted features designed to capture higher-order interactions and
# latent constructs that raw columns cannot express individually.
# =============================================================================

tech_cols = ['coding_score', 'problem_solving_score', 'data_structures_score', 'sql_score',
             'machine_learning_score', 'backend_score', 'frontend_score', 'cloud_score', 'devops_score']
soft_cols = ['communication_score', 'teamwork_score', 'leadership_score', 'presentation_score']

# --- Aggregate skill profiles ---
all_data['total_tech_score']    = all_data[tech_cols].sum(axis=1)
all_data['avg_tech_score']      = all_data[tech_cols].mean(axis=1)
all_data['avg_soft_skills']     = all_data[soft_cols].mean(axis=1)
all_data['avg_interview_score'] = all_data[['technical_interview_score', 'hr_interview_score']].mean(axis=1)

# --- Experience volume and quality ---
# Total internship volume weights count by duration, rewarding depth over breadth.
all_data['total_internship_volume'] = all_data['internship_count'] * all_data['internship_duration_months']
# GitHub impact weights repo count by average star rating as a quality multiplier.
all_data['github_impact']           = all_data['github_repo_count'] * all_data['github_avg_stars']
# Attendance adjusted for academic failures; penalizes poor attendance more for high-failure students.
all_data['attendance_vs_fail']      = all_data['attendance_rate'] / (all_data['failed_courses_count'] + 1)
all_data['total_projects']          = all_data['real_client_project_count'] + all_data['freelance_project_count']

# --- Profile consistency and balance ---
# High variance across tech sub-scores may indicate a specialist or an inconsistent candidate.
all_data['tech_variance']      = all_data[tech_cols].std(axis=1)
# A large gap between technical and HR interview scores signals a social-technical imbalance.
all_data['interview_gap']      = abs(all_data['technical_interview_score'] - all_data['hr_interview_score'])
# Composite online presence score; GitHub stars are weighted higher as they reflect external validation.
all_data['digital_footprint']  = (all_data['linkedin_profile_score'] + all_data['portfolio_score']
                                   + all_data['github_avg_stars'] * 10)
all_data['tech_vs_soft_ratio'] = all_data['avg_tech_score'] / (all_data['avg_soft_skills'] + 1)
# Combines project count and internship months into a single practical experience proxy.
all_data['experience_density'] = all_data['total_projects'] + all_data['internship_duration_months']

# --- Readiness and risk flags ---
train_tech_mean = all_data['avg_tech_score'].iloc[:len(train_df)].mean()
# A student is considered globally competitive if they meet the English proficiency
# threshold AND have above-average technical skills relative to the training population.
all_data['is_global_ready'] = (
    (all_data['english_exam_score'] >= 75) & (all_data['avg_tech_score'] >= train_tech_mean)
).astype(int)
# Flags students at academic risk: low attendance combined with multiple failed courses.
all_data['academic_risk'] = (
    (all_data['attendance_rate'] < 70) & (all_data['failed_courses_count'] > 2)
).astype(int)

# --- Quality-weighted interaction features ---
# Scales project count by quality score to distinguish prolific from impactful contributors.
all_data['quality_per_project']      = all_data['project_quality_score'] * all_data['real_client_project_count']
# Compares output quality against overall technical depth; high ratio implies efficiency.
all_data['quality_vs_tech_ratio']    = all_data['project_quality_score'] / (all_data['total_tech_score'] + 1)
# Captures disconnect between scored technical ability and live interview performance.
all_data['tech_vs_interview_gap']    = abs(all_data['total_tech_score'] - all_data['avg_interview_score'])
# Joint signal: high values indicate both strong technical interviews and strong project output.
all_data['quality_x_tech_interview'] = all_data['project_quality_score'] * all_data['technical_interview_score']
all_data['tech_impact_score']        = all_data['total_tech_score'] * all_data['project_quality_score']

# --- Job search efficiency ---
# Interview conversion rate: ratio of interviews secured to applications sent.
all_data['cv_conversion_rate']  = all_data['interviews_attended'] / (all_data['applications_sent'] + 1)
# Hackathon win rate normalizes awards against participation to reward precision over volume.
all_data['hackathon_win_rate']  = all_data['hackathon_awards'] / (all_data['hackathon_count'] + 1)
prof_networks = ['LinkedIn', 'Twitter', 'Github', 'Reddit']
# Binary flag for students whose primary social platform is career or tech-oriented.
all_data['is_professional_networker'] = all_data['preferred_social_media_platform'].isin(prof_networks).astype(int)

# Anomaly flag: top-20% technical performers with zero online presence are statistically
# inconsistent and may represent data quality issues or unusual profiles.
tech_top_20_threshold = all_data['total_tech_score'].quantile(0.80)
all_data['suspicious_profile_flag'] = (
    (all_data['total_tech_score'] > tech_top_20_threshold) &
    (all_data['digital_footprint'] == 0)
).astype(int)

# Within-tier z-score: normalizes technical performance relative to peers at the same
# university tier, isolating individual merit from institutional advantage.
all_data['tech_z_score_by_tier'] = all_data.groupby('university_tier')['total_tech_score'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-9)
)

# Skill chaos score: cross-domain standard deviation across technical skills, soft skills,
# and GPA. High values indicate a candidate with uneven development across key competency areas.
norm_tech = all_data['avg_tech_score'] / 100   # retained for interpretability
norm_soft = all_data['avg_soft_skills'] / 100  # retained for interpretability
norm_cgpa = all_data['cgpa'] / 4.0             # retained for interpretability
all_data['skill_chaos_score'] = all_data[['avg_tech_score', 'avg_soft_skills', 'cgpa']].std(axis=1)

# =============================================================================
# 8. FEATURE SELECTION (column pruning)
# Columns removed for one of three reasons:
#   (i)  Low predictive utility or redundancy with engineered features
#   (ii) Consumed entirely by a derived feature (e.g., raw counts aggregated)
#   (iii) Post-feature-importance ablation during experimentation
# =============================================================================
cols_to_drop = [
    'hobby', 'preferred_social_media_platform', 'department', 'age', 'avg_tech_score',
    'attendance_rate', 'freelance_project_count', 'feedback_lower', 'mentor_feedback_text',
    'student_id', 'internship_count', 'failed_courses_count', 'hackathon_count',
    'hackathon_awards', 'presentation_score', 'interview_gap', 'github_avg_stars',
    'total_projects', 'sql_score', 'data_structures_score', 'interviews_attended',
    'tech_vs_soft_ratio', 'total_internship_volume', 'machine_learning_score',
    'hr_interview_score', 'tech_variance', 'linkedin_profile_score',
    'english_exam_score', 'tech_vs_interview_gap', 'leadership_score'
]
cols_to_drop = [col for col in cols_to_drop if col in all_data.columns]
all_data = all_data.drop(cols_to_drop, axis=1)

student_ids_test = test_df['student_id']

X      = all_data.iloc[:len(train_df)].copy()
X_test = all_data.iloc[len(train_df):].copy()
y      = y_train.copy()

cat_features_indices = [X.columns.get_loc(col) for col in cat_features]

# =============================================================================
# 9. MULTI-SEED STACKED ENSEMBLE (7-Fold Cross-Validation)
#
# Architecture overview:
#   - Three base learners: CatBoost, XGBoost, TabNet
#   - 7-fold CV per seed generates out-of-fold (OOF) predictions
#   - A Ridge meta-learner is trained on OOF predictions (stacking)
#   - The full loop runs over 3 random seeds; final predictions are
#     averaged across seeds to reduce variance from fold randomness
#
# This approach balances predictive strength (diverse base learners),
# generalization (OOF stacking), and stability (multi-seed averaging).
# =============================================================================
SEEDS = [6, 16, 2004]

final_blend_oof  = np.zeros(len(X))
final_blend_test = np.zeros(len(X_test))

print(f"\n🌟 Launching multi-seed ensemble training ({len(SEEDS)} seeds × 7 folds × 3 models)...")

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"  Seed {seed} — initializing fold splits and base learners")
    print(f"{'='*50}")

    kf = KFold(n_splits=7, shuffle=True, random_state=seed)

    oof_cat    = np.zeros(len(X))
    oof_xgb    = np.zeros(len(X))
    oof_tabnet = np.zeros(len(X))

    test_preds_cat    = np.zeros(len(X_test))
    test_preds_xgb    = np.zeros(len(X_test))
    test_preds_tabnet = np.zeros(len(X_test))

    # Hyperparameters sourced from Optuna-based Bayesian optimization
    cat_params = {
        'iterations': 1876, 'learning_rate': 0.033879238243611935, 'depth': 7,
        'l2_leaf_reg': 6.788120472202029, 'bagging_temperature': 0.8075386228260518,
        'random_strength': 9.736873465962884, 'border_count': 197,
        'random_seed': seed, 'verbose': 0
    }

    xgb_params = {
        'n_estimators': 733, 'learning_rate': 0.021766257606774843, 'max_depth': 4,
        'subsample': 0.6004952071355936, 'colsample_bytree': 0.8353091973285699,
        'min_child_weight': 6, 'gamma': 0.3829487142125,
        'reg_alpha': 4.883641985410483, 'reg_lambda': 4.186260395775482,
        'random_state': seed, 'tree_method': 'hist'
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

        X_train_fold, X_val_fold = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        X_test_fold = X_test.copy()

        # Target encoding is fit exclusively on the training fold to prevent
        # target leakage into the validation or test sets.
        encoder = ce.TargetEncoder(cols=cat_features, smoothing=10)
        X_train_fold_enc = encoder.fit_transform(X_train_fold, y_train_fold)
        X_val_fold_enc   = encoder.transform(X_val_fold)
        X_test_fold_enc  = encoder.transform(X_test_fold)

        # TabNet requires normalized inputs; scaler is also fit on the training
        # fold only to maintain strict data isolation.
        scaler       = StandardScaler()
        X_train_tab  = scaler.fit_transform(X_train_fold_enc).astype(np.float32)
        X_val_tab    = scaler.transform(X_val_fold_enc).astype(np.float32)
        X_test_tab   = scaler.transform(X_test_fold_enc).astype(np.float32)

        y_train_tab = y_train_fold.values.reshape(-1, 1).astype(np.float32)
        y_val_tab   = y_val_fold.values.reshape(-1, 1).astype(np.float32)

        # --- Base Learner 1: CatBoost ---
        # Handles raw categoricals natively; early stopping on the validation fold.
        cat_model = CatBoostRegressor(**cat_params, cat_features=cat_features_indices)
        cat_model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=100)

        # --- Base Learner 2: XGBoost ---
        # Operates on target-encoded features; GPU-accelerated histogram method.
        xgb_model = XGBRegressor(**xgb_params, early_stopping_rounds=100)
        xgb_model.fit(X_train_fold_enc, y_train_fold, eval_set=[(X_val_fold_enc, y_val_fold)], verbose=False)

        # --- Base Learner 3: TabNet ---
        # Attention-based deep learning model for tabular data; trained on scaled inputs.
        tabnet_model = TabNetRegressor(verbose=0, seed=seed)
        tabnet_model.fit(
            X_train=X_train_tab, y_train=y_train_tab,
            eval_set=[(X_val_tab, y_val_tab)],
            eval_name=['val'], eval_metric=['mse'],
            max_epochs=250, patience=30,
            batch_size=256, virtual_batch_size=128
        )

        # Collect OOF predictions for the stacking meta-learner
        oof_cat[val_idx]    = cat_model.predict(X_val_fold)
        oof_xgb[val_idx]    = xgb_model.predict(X_val_fold_enc)
        oof_tabnet[val_idx] = tabnet_model.predict(X_val_tab).flatten()

        # Accumulate test predictions (averaged across folds)
        test_preds_cat    += cat_model.predict(X_test_fold) / kf.n_splits
        test_preds_xgb    += xgb_model.predict(X_test_fold_enc) / kf.n_splits
        test_preds_tabnet += tabnet_model.predict(X_test_tab).flatten() / kf.n_splits

    # --- Level-2 Meta-Learner: Ridge Regression ---
    # Trained on the stacked OOF predictions of all three base models.
    # Ridge regularization prevents the meta-learner from over-relying on
    # any single base model in cases of collinear OOF predictions.
    X_meta      = np.column_stack((oof_cat, oof_xgb, oof_tabnet))
    X_test_meta = np.column_stack((test_preds_cat, test_preds_xgb, test_preds_tabnet))

    seed_ridge = Ridge(alpha=10.0, random_state=seed)
    seed_ridge.fit(X_meta, y)

    seed_oof_preds  = seed_ridge.predict(X_meta)
    seed_test_preds = seed_ridge.predict(X_test_meta)

    seed_mse = mean_squared_error(y, seed_oof_preds)
    print(f"  ✅ Seed {seed} stacked OOF MSE: {seed_mse:.4f}")

    final_blend_oof  += seed_oof_preds  / len(SEEDS)
    final_blend_test += seed_test_preds / len(SEEDS)


# =============================================================================
# 10. FINAL EVALUATION & SUBMISSION
# =============================================================================
print("\n" + "="*50)
print("🏆  MULTI-SEED ENSEMBLE — FINAL RESULTS")
print("="*50)

final_overall_mse = mean_squared_error(y, final_blend_oof)
print(f"🔥 Combined OOF MSE (all seeds averaged): {final_overall_mse:.4f}")

# Clip predictions to valid score range [0, 100] before submission
final_blend_test = np.clip(final_blend_test, 0, 100)
submission_df = pd.DataFrame({'student_id': student_ids_test, 'career_success_score': final_blend_test})
submission_df.to_csv('submission.csv', index=False)
print("✅ Submission file saved to 'submission.csv'.")
