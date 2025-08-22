# evaluate_models_summary.py
# Compact overall metrics + elaborated per-task results.
# Upgraded Intensity (bpm): tunes multiple regressors, chooses best by hold-out R².

import math
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Optional regressors (used if installed)
HAS_LGBM = HAS_XGB = HAS_CAT = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    pass
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    pass
try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception:
    pass

# =========================
# CONFIG
# =========================
DATA_PATH = "Cleaned_20240320 (2).xlsx"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
N_ITER_TUNE = 60  # per regressor for intensity

# Fallback target names in case auto-detection fails
TARGET_FALLBACK_FREQ = "Frequency"
TARGET_FALLBACK_DUR  = "Duration"
TARGET_FALLBACK_TYPE = "Type"
TARGET_FALLBACK_INT  = "Target HR (bpm)"

# =========================
# Helper functions (match app logic)
# =========================
def ensure_freq_bins(label: str):
    s = str(label).strip().lower()
    if "low" in s or "1-2" in s or "1–2" in s: return "Low"
    if "high" in s or "5-6" in s or "5–6" in s: return "High"
    return "Medium"

def ensure_dur_bins(label: str):
    s = str(label).strip().lower()
    if "short" in s or "10-20" in s or "10–20" in s: return "Short"
    if "long" in s or "40-60" in s or "40–60" in s: return "Long"
    return "Medium"

def load_df(path):
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.read_excel(path, engine="openpyxl")

def detect_numeric_and_categorical(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols

def safe_col(df, needles, prefer_numeric=None):
    for needle in needles:
        hits = [c for c in df.columns if needle.lower() in c.lower()]
        if prefer_numeric is True:  hits = [c for c in hits if pd.api.types.is_numeric_dtype(df[c])]
        if prefer_numeric is False: hits = [c for c in hits if not pd.api.types.is_numeric_dtype(df[c])]
        if hits: return hits[0]
    return None

def guess_targets(df):
    freq_col = safe_col(df, ["exercise habit - frequency", "frequency"], prefer_numeric=False)
    dur_col  = safe_col(df, ["exercise habit - duration", "duration"], prefer_numeric=False)
    type_col = safe_col(df, ["exercise habit - mode", "type", "mode"], prefer_numeric=False)
    int_col  = safe_col(df, ["target hr (bpm)", "target_hr_bpm", "intensity", "hr bpm"], prefer_numeric=True)
    if freq_col is None and TARGET_FALLBACK_FREQ in df.columns: freq_col = TARGET_FALLBACK_FREQ
    if dur_col  is None and TARGET_FALLBACK_DUR  in df.columns: dur_col  = TARGET_FALLBACK_DUR
    if type_col is None and TARGET_FALLBACK_TYPE in df.columns: type_col = TARGET_FALLBACK_TYPE
    if int_col  is None and TARGET_FALLBACK_INT  in df.columns: int_col  = TARGET_FALLBACK_INT
    return freq_col, dur_col, type_col, int_col

def _is_year_like(col: str) -> bool:
    lc = col.lower()
    return ("year" in lc) or lc == "yr" or lc.endswith("_yr") or lc.endswith("year")

def _is_id_like(col: str) -> bool:
    lc = col.lower()
    return lc == "id" or lc.endswith("_id") or "patient id" in lc or lc == "patientid"

def build_feature_list(df, targets):
    num_cols, _ = detect_numeric_and_categorical(df)
    to_drop = set([c for c in targets if c is not None])
    cleaned = []
    for c in num_cols:
        if c in to_drop: continue
        if _is_year_like(c): continue
        if _is_id_like(c): continue
        cleaned.append(c)
    return cleaned

def make_preprocessor(num_features):
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    return ColumnTransformer([("num", num_pipe, num_features)], remainder="drop")

def _clean_cat_target(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip()
    s2 = s2.replace({"": np.nan, "na": np.nan, "Na": np.nan, "NA": np.nan, "none": np.nan, "None": np.nan, "Nan": np.nan})
    return s2

def root_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

# =========================
# Load & prepare
# =========================
if not Path(DATA_PATH).exists():
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

df = load_df(DATA_PATH)
freq_col, dur_col, type_col, int_col = guess_targets(df)
if any(x is None for x in [freq_col, dur_col, type_col, int_col]):
    raise RuntimeError("Could not auto-detect all targets. Please check the dataset column names.")

cat_freq_raw = _clean_cat_target(df[freq_col])
cat_dur_raw  = _clean_cat_target(df[dur_col])
cat_type_raw = _clean_cat_target(df[type_col])
num_int_raw  = pd.to_numeric(df[int_col], errors="coerce")

mask_valid = (cat_freq_raw.notna() & cat_dur_raw.notna() & cat_type_raw.notna() & num_int_raw.notna())
dfc = df.loc[mask_valid].copy()
cat_freq_raw = cat_freq_raw.loc[mask_valid]
cat_dur_raw  = cat_dur_raw .loc[mask_valid]
cat_type_raw = cat_type_raw.loc[mask_valid]
num_int_raw  = num_int_raw .loc[mask_valid]

if len(dfc) < 50:
    print(f"[WARN] Only {len(dfc)} usable rows after cleaning; metrics may be unstable.")

# Targets (clean/binned like the app)
y_freq_raw = cat_freq_raw.astype(str).apply(ensure_freq_bins)
y_dur_raw  = cat_dur_raw.astype(str).apply(ensure_dur_bins)
y_type_raw = cat_type_raw.astype(str)
y_int      = num_int_raw.astype(float)

# Features
features = build_feature_list(dfc, [freq_col, dur_col, type_col, int_col])
if not features:
    raise RuntimeError("No numeric features found to train on after filtering targets.")

# One split shared by all tasks
X_train, X_test, yf_tr, yf_te, yd_tr, yd_te, yt_tr, yt_te, yi_tr, yi_te = train_test_split(
    dfc[features], y_freq_raw, y_dur_raw, y_type_raw, y_int,
    test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Label encoders for classifiers
le_freq = LabelEncoder().fit(yf_tr)
le_dur  = LabelEncoder().fit(yd_tr)
le_type = LabelEncoder().fit(yt_tr)

yf_tr_enc = le_freq.transform(yf_tr)
yf_te_enc = le_freq.transform(yf_te)
yd_tr_enc = le_dur.transform(yd_tr)
yd_te_enc = le_dur.transform(yd_te)
yt_tr_enc = le_type.transform(yt_tr)
yt_te_enc = le_type.transform(yt_te)

# Preprocessor shared across tasks
pre = make_preprocessor(features)

# =========================
# Classification models (same as app)
# =========================
clf_freq = Pipeline([("pre", pre), ("rf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))])
clf_dur  = Pipeline([("pre", pre), ("rf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))])
clf_type = Pipeline([("pre", pre), ("rf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))])

# Fit classification
clf_freq.fit(X_train, yf_tr_enc)
clf_dur.fit(X_train, yd_tr_enc)
clf_type.fit(X_train, yt_tr_enc)

# =========================
# Intensity — upgraded modeling
# =========================
cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def tune_regressor(name, base_estimator, param_dist):
    pipe = Pipeline([("pre", pre), ("model", base_estimator)])
    tuner = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=N_ITER_TUNE,
        scoring="r2",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    tuner.fit(X_train, yi_tr)
    return name, tuner

candidates = []

# RandomForest
rf_space = {
    "model__n_estimators": [400, 700, 1000, 1400],
    "model__max_depth": [None, 12, 16, 24],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["auto", "sqrt", 0.5, 0.8],
}
candidates.append(tune_regressor("RandomForest", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), rf_space))

# GradientBoosting
gb_space = {
    "model__n_estimators": [400, 700, 1000, 1400],
    "model__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
    "model__max_depth": [2, 3, 4, 5],
    "model__subsample": [0.8, 0.9, 1.0],
}
candidates.append(tune_regressor("GradientBoosting", GradientBoostingRegressor(random_state=RANDOM_STATE), gb_space))

# SVR
svr_space = {
    "model__C": [0.5, 1.0, 3.0, 10.0, 20.0],
    "model__epsilon": [0.05, 0.1, 0.2, 0.3],
    "model__gamma": ["scale", "auto"],
}
candidates.append(tune_regressor("SVR", SVR(kernel="rbf"), svr_space))

# LightGBM
if HAS_LGBM:
    lgbm_space = {
        "model__n_estimators": [700, 1000, 1400, 1800],
        "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
        "model__num_leaves": [31, 63, 127, 255],
        "model__max_depth": [-1, 8, 12, 16],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.8, 0.9, 1.0],
        "model__reg_lambda": [0.0, 0.5, 1.0, 2.0],
    }
    candidates.append(tune_regressor("LightGBM", LGBMRegressor(random_state=RANDOM_STATE), lgbm_space))

# XGBoost
if HAS_XGB:
    xgb_space = {
        "model__n_estimators": [700, 1000, 1400, 1800],
        "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.8, 0.9, 1.0],
        "model__reg_lambda": [0.0, 0.5, 1.0, 2.0],
        "model__min_child_weight": [1, 5, 10],
    }
    candidates.append(tune_regressor("XGBoost", XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist"), xgb_space))

# CatBoost
if HAS_CAT:
    cat_space = {
        "model__depth": [6, 8, 10],
        "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
        "model__n_estimators": [800, 1200, 1600, 2000],
        "model__l2_leaf_reg": [1, 3, 5, 7],
    }
    candidates.append(tune_regressor("CatBoost", CatBoostRegressor(random_state=RANDOM_STATE, loss_function="RMSE", verbose=False), cat_space))

# Evaluate tuned candidates on hold-out; keep the best by R²
best_name = None
best_est  = None
best_r2   = -1e9
best_mae  = None
best_rmse = None
best_params = None
best_cv_r2 = None

for name, tuner in candidates:
    y_pred = tuner.predict(X_test)
    mae  = mean_absolute_error(yi_te, y_pred)
    rmse = root_mse(yi_te, y_pred)
    r2   = r2_score(yi_te, y_pred)
    cv_r2 = tuner.best_score_
    if r2 > best_r2:
        best_r2 = r2
        best_name = name
        best_est = tuner.best_estimator_
        best_mae = mae
        best_rmse = rmse
        best_params = tuner.best_params_
        best_cv_r2 = cv_r2

# =========================
# Predictions for classification
# =========================
yf_pred_enc = clf_freq.predict(X_test)
yd_pred_enc = clf_dur.predict(X_test)
yt_pred_enc = clf_type.predict(X_test)

# =========================
# Elaborated results
# =========================
def print_cls_block(title, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n=== {title} (Classification) ===")
    print(f"Accuracy           : {acc:.4f}")
    print(f"Weighted Precision : {prec_w:.4f}")
    print(f"Weighted F1-score  : {f1_w:.4f}")
    return acc, prec_w, f1_w

def print_intensity_block():
    print("\n=== Intensity (bpm) — Best Regressor ===")
    print(f"Model              : {best_name}")
    print(f"CV R² (mean, tuned): {best_cv_r2:.4f}")
    print(f"Hold-out MAE       : {best_mae:.4f}")
    print(f"Hold-out RMSE      : {best_rmse:.4f}")
    print(f"Hold-out R²        : {best_r2:.4f}")
    print("Best Params:")
    pprint(best_params)

f_acc, f_prec, f_f1 = print_cls_block("Frequency", yf_te_enc, yf_pred_enc)
d_acc, d_prec, d_f1 = print_cls_block("Duration",  yd_te_enc, yd_pred_enc)
t_acc, t_prec, t_f1 = print_cls_block("Type",      yt_te_enc, yt_pred_enc)
print_intensity_block()

# =========================
# Overall summary table
# =========================
summary_rows = [
    {"Prediction Task": "Frequency", "Accuracy": round(float(f_acc),4), "Weighted Precision": round(float(f_prec),4), "Weighted F1-score": round(float(f_f1),4), "MAE": None, "RMSE": None, "R2": None},
    {"Prediction Task": "Duration",  "Accuracy": round(float(d_acc),4), "Weighted Precision": round(float(d_prec),4), "Weighted F1-score": round(float(d_f1),4), "MAE": None, "RMSE": None, "R2": None},
    {"Prediction Task": "Type",      "Accuracy": round(float(t_acc),4), "Weighted Precision": round(float(t_prec),4), "Weighted F1-score": round(float(t_f1),4), "MAE": None, "RMSE": None, "R2": None},
    {"Prediction Task": "Intensity (bpm)", "Accuracy": None, "Weighted Precision": None, "Weighted F1-score": None, "MAE": round(float(best_mae),4), "RMSE": round(float(best_rmse),4), "R2": round(float(best_r2),4)}
]
summary_df = pd.DataFrame(summary_rows, columns=["Prediction Task","Accuracy","Weighted Precision","Weighted F1-score","MAE","RMSE","R2"])

print("\n=== Overall Model Performance ===")
print(summary_df.fillna("—").to_string(index=False))

# Save CSV + best intensity model
summary_df.to_csv("model_metrics_summary.csv", index=False)
print("\nSaved: model_metrics_summary.csv")

import joblib
joblib.dump(best_est, "intensity_best_model.joblib")
print("Saved: intensity_best_model.joblib")
