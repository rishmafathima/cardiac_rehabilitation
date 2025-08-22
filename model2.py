# evaluate_models_summary.py
# Goals:
# - Make Duration less "perfect" by removing leakage + limiting model capacity
# - Report repeated-split generalization for Duration (so a lucky 1.0 won't stick)
# - Keep Type improvements (LGBM+SMOTE if available; RF fallback)
# - Keep Intensity tuner with fixed param grid (no 'auto' for max_features)

import math
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
import warnings
from collections import Counter

from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, StratifiedShuffleSplit,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

# Optional libs
HAS_LGBM = HAS_XGB = HAS_CAT = False
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
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

HAS_SMOTE = False
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except Exception:
    pass

# =========================
# CONFIG
# =========================
DATA_PATH = "Cleaned_20240320 (2).xlsx"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS_MAX = 5
N_ITER_TUNE = 60
DUR_REPEATS = 7  # repeated splits to avoid lucky 1.0

# Fallback targets
TARGET_FALLBACK_FREQ = "Frequency"
TARGET_FALLBACK_DUR  = "Duration"
TARGET_FALLBACK_TYPE = "Type"
TARGET_FALLBACK_INT  = "Target HR (bpm)"

# =========================
# Helpers
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

def ensure_type_bins(label: str):
    s = str(label).strip().lower()
    if "moderate" in s: return "moderate"
    if "low" in s: return "low"
    if "unknown" in s or s in {"", "na", "none", "nan"}: return "unknown"
    if any(k in s for k in ["walk","walking","cycle","bike","bicycle","treadmill","elliptical","row"]):
        return "moderate"
    return "unknown"

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
    freq_col = safe_col(df, ["exercise habit - frequency","frequency"], prefer_numeric=False)
    dur_col  = safe_col(df, ["exercise habit - duration","duration"], prefer_numeric=False)
    type_col = safe_col(df, ["exercise habit - mode","type","mode"], prefer_numeric=False)
    int_col  = safe_col(df, ["target hr (bpm)","target_hr_bpm","intensity","hr bpm"], prefer_numeric=True)
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

def choose_smote_k(y: pd.Series) -> int:
    counts = Counter(y)
    min_count = min(counts.values())
    if min_count < 2:
        return 0
    return max(1, min(5, min_count - 1))

def safe_stratified_cv(y, kmax=5, random_state=42):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    min_class = y.value_counts().min()
    n_splits = max(2, min(kmax, int(min_class)))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def report_class_dist(name, y):
    vc = pd.Series(y).value_counts()
    print(f"\n[{name}] class distribution (train):")
    print(vc.to_string())

def find_leaky_features(X: pd.DataFrame, y_enc: np.ndarray, label_name="Duration", corr_thr=0.98):
    """Very simple leakage sniff: drop columns that
       1) contain the target name in the column string, or
       2) have |Pearson corr| > corr_thr with the encoded label.
    """
    leaky = []
    lower_cols = {c: c.lower() for c in X.columns}
    for c in X.columns:
        lc = lower_cols[c]
        if "duration" in lc or "dur_" in lc or lc.endswith("_duration"):
            leaky.append(c)

    y_series = pd.Series(y_enc).astype(float)
    for c in X.columns:
        try:
            corr = y_series.corr(pd.Series(X[c]).astype(float))
            if pd.notna(corr) and abs(corr) >= corr_thr:
                leaky.append(c)
        except Exception:
            pass

    leaky = sorted(set(leaky))
    if leaky:
        print(f"\n[Leakage guard] Removing {len(leaky)} suspicious feature(s): {leaky}")
    else:
        print("\n[Leakage guard] No suspicious features detected by simple rules.")
    return leaky

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
cat_dur_raw  = cat_dur_raw.loc[mask_valid]
cat_type_raw = cat_type_raw.loc[mask_valid]
num_int_raw  = num_int_raw.loc[mask_valid]

if len(dfc) < 50:
    print(f"[WARN] Only {len(dfc)} usable rows after cleaning; metrics may be unstable.")

# Targets
y_freq_raw = cat_freq_raw.astype(str).apply(ensure_freq_bins)
y_dur_raw  = cat_dur_raw.astype(str).apply(ensure_dur_bins)
y_type_raw = cat_type_raw.astype(str).apply(ensure_type_bins)
y_int      = num_int_raw.astype(float)

# Features
features_all = build_feature_list(dfc, [freq_col, dur_col, type_col, int_col])
if not features_all:
    raise RuntimeError("No numeric features found to train on after filtering targets.")

# One base split
X_train_full, X_test_full, yf_tr, yf_te, yd_tr, yd_te, yt_tr, yt_te, yi_tr, yi_te = train_test_split(
    dfc[features_all], y_freq_raw, y_dur_raw, y_type_raw, y_int,
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_type_raw
)

# Encode freq/dur
le_freq = LabelEncoder().fit(yf_tr)
le_dur  = LabelEncoder().fit(yd_tr)
yf_tr_enc = le_freq.transform(yf_tr)
yf_te_enc = le_freq.transform(yf_te)
yd_tr_enc = le_dur.transform(yd_tr)
yd_te_enc = le_dur.transform(yd_te)

# Leakage check for Duration (uses training fold only)
leaky_cols = find_leaky_features(X_train_full, yd_tr_enc, label_name="Duration", corr_thr=0.98)
features = [c for c in features_all if c not in leaky_cols]
X_train = X_train_full[features].copy()
X_test  = X_test_full[features].copy()

# Preprocessor
pre = make_preprocessor(features)

# Report class distribution
report_class_dist("Frequency", yf_tr)
report_class_dist("Duration", yd_tr)
report_class_dist("Type", yt_tr)

# =========================
# Frequency — baseline RF
# =========================
clf_freq = Pipeline([("pre", pre), ("rf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))])
clf_freq.fit(X_train, yf_tr_enc)
yf_pred_enc = clf_freq.predict(X_test)

# =========================
# Duration — tuned, constrained RF + repeated splits
# =========================
dur_cv = safe_stratified_cv(pd.Series(yd_tr_enc), kmax=CV_FOLDS_MAX, random_state=RANDOM_STATE)
dur_pipe = Pipeline([("pre", pre), ("rf", RandomForestClassifier(random_state=RANDOM_STATE))])
# Strong constraints to avoid perfect fits
dur_param_dist = {
    "rf__n_estimators": [100, 150, 200, 300],
    "rf__max_depth": [3, 4, 5, 6],
    "rf__min_samples_leaf": [3, 4, 5, 6],
    "rf__min_samples_split": [10, 15, 20],
    "rf__max_features": ["sqrt", 0.5],
    "rf__bootstrap": [True],
}
clf_dur_tuner = RandomizedSearchCV(
    estimator=dur_pipe,
    param_distributions=dur_param_dist,
    n_iter=min(30, int(np.prod([len(v) for v in dur_param_dist.values()]))),
    scoring="f1_weighted",
    cv=dur_cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)
clf_dur_tuner.fit(X_train, yd_tr_enc)
yd_pred_enc = clf_dur_tuner.predict(X_test)

dur_train_acc = clf_dur_tuner.score(X_train, yd_tr_enc)
dur_test_acc  = accuracy_score(yd_te_enc, yd_pred_enc)
print(f"\n[Duration] (base split) Train acc: {dur_train_acc:.4f} | Test acc: {dur_test_acc:.4f}")
print("[Duration] Best params (constrained):")
pprint(clf_dur_tuner.best_params_)

# Repeated-split evaluation for Duration with the tuned hyperparams
dur_test_scores = []
sss = StratifiedShuffleSplit(n_splits=DUR_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
best_params = clf_dur_tuner.best_params_
for i, (idx_tr, idx_te) in enumerate(sss.split(dfc[features], y_dur_raw)):
    Xtr = dfc.iloc[idx_tr][features]
    Xte = dfc.iloc[idx_te][features]
    ytr = le_dur.fit_transform(y_dur_raw.iloc[idx_tr])  # re-fit encoder to keep mapping
    yte = le_dur.transform(y_dur_raw.iloc[idx_te])

    pre_i = make_preprocessor(features)
    dur_model = Pipeline([("pre", pre_i), ("rf", RandomForestClassifier(random_state=RANDOM_STATE, **{
        k.split("__",1)[1]: v for k,v in best_params.items()
    }))])
    dur_model.fit(Xtr, ytr)
    yhat = dur_model.predict(Xte)
    acc = accuracy_score(yte, yhat)
    dur_test_scores.append(acc)
    print(f"[Duration] Repeated split {i+1}/{DUR_REPEATS} — Test acc: {acc:.4f}")

print(f"[Duration] Repeated splits — mean±std: {np.mean(dur_test_scores):.4f} ± {np.std(dur_test_scores):.4f}")

# =========================
# Type — LGBM+SMOTE (fallback RF)
# =========================
type_cv = safe_stratified_cv(pd.Series(yt_tr), kmax=CV_FOLDS_MAX, random_state=RANDOM_STATE)
use_smote = HAS_SMOTE
k_smote = choose_smote_k(pd.Series(yt_tr)) if use_smote else 0
if use_smote and k_smote == 0:
    warnings.warn("SMOTE skipped: at least one class has <2 samples in training.")
    use_smote = False

if HAS_LGBM:
    if use_smote:
        type_pipe = ImbPipeline(steps=[
            ("pre", pre),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=k_smote)),
            ("lgbm", LGBMClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1
            ))
        ])
    else:
        type_pipe = Pipeline(steps=[
            ("pre", pre),
            ("lgbm", LGBMClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1
            ))
        ])
    type_param_dist = {
        "lgbm__n_estimators": [300, 500, 800, 1000],
        "lgbm__learning_rate": [0.02, 0.05, 0.1],
        "lgbm__max_depth": [-1, 6, 8, 12],
        "lgbm__num_leaves": [31, 63, 127],
        "lgbm__subsample": [0.8, 1.0],
        "lgbm__colsample_bytree": [0.8, 1.0]
    }
    type_model_name = "LightGBM"
else:
    if use_smote:
        type_pipe = ImbPipeline(steps=[
            ("pre", pre),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=k_smote)),
            ("rf", RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced_subsample",
                n_jobs=-1
            ))
        ])
    else:
        warnings.warn("imblearn/SMOTE unavailable or unusable. Using class_weight='balanced' only for Type.")
        type_pipe = Pipeline(steps=[
            ("pre", pre),
            ("rf", RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1
            ))
        ])
    type_param_dist = {
        "rf__n_estimators": [300, 500, 800, 1200],
        "rf__max_depth": [None, 8, 12, 16, 24],
        "rf__min_samples_leaf": [1, 2, 3, 4],
        "rf__min_samples_split": [2, 5, 10],
        "rf__max_features": ["sqrt", "log2", 0.5, 0.8]
    }
    type_model_name = "RandomForest"

type_tuner = RandomizedSearchCV(
    estimator=type_pipe,
    param_distributions=type_param_dist,
    n_iter=min(40, int(np.prod([len(v) for v in type_param_dist.values()]))),
    scoring="f1_weighted",
    cv=type_cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)
type_tuner.fit(X_train, yt_tr)
yt_pred = type_tuner.predict(X_test)
yt_pred_train = type_tuner.predict(X_train)
print(f"\n[Type-{type_model_name}] Train acc: {accuracy_score(yt_tr, yt_pred_train):.4f} | Test acc: {accuracy_score(yt_te, yt_pred):.4f}")
print("[Type] Tuned best params:")
pprint(type_tuner.best_params_)

# =========================
# Intensity — multi-regressor tuner (RF grid fixed)
# =========================
cv = KFold(n_splits=CV_FOLDS_MAX, shuffle=True, random_state=RANDOM_STATE)

def tune_regressor(name, base_estimator, param_dist):
    pipe = Pipeline([("pre", pre), ("model", base_estimator)])
    n_possible = int(np.prod([len(v) for v in param_dist.values()]))
    tuner = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=min(N_ITER_TUNE, n_possible),
        scoring="r2",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    tuner.fit(X_train, yi_tr)
    return name, tuner

candidates = []
rf_space = {
    "model__n_estimators": [400, 700, 1000, 1400],
    "model__max_depth": [None, 12, 16, 24],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": [None, "sqrt", 0.5, 0.8],  # fixed: no 'auto'
}
candidates.append(tune_regressor("RandomForest", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), rf_space))

gb_space = {
    "model__n_estimators": [400, 700, 1000, 1400],
    "model__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
    "model__max_depth": [2, 3, 4, 5],
    "model__subsample": [0.8, 0.9, 1.0],
}
candidates.append(tune_regressor("GradientBoosting", GradientBoostingRegressor(random_state=RANDOM_STATE), gb_space))

svr_space = {
    "model__C": [0.5, 1.0, 3.0, 10.0, 20.0],
    "model__epsilon": [0.05, 0.1, 0.2, 0.3],
    "model__gamma": ["scale", "auto"],
}
candidates.append(tune_regressor("SVR", SVR(kernel="rbf"), svr_space))

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

if HAS_CAT:
    cat_space = {
        "model__depth": [6, 8, 10],
        "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
        "model__n_estimators": [800, 1200, 1600, 2000],
        "model__l2_leaf_reg": [1, 3, 5, 7],
    }
    candidates.append(tune_regressor("CatBoost", CatBoostRegressor(random_state=RANDOM_STATE, loss_function="RMSE", verbose=False), cat_space))

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
# Metrics & summary
# =========================
def print_cls_block(title, y_true, y_pred):
    acc   = accuracy_score(y_true, y_pred)
    precw = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    f1w   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n=== {title} (Classification) ===")
    print(f"Accuracy           : {acc:.4f}")
    print(f"Weighted Precision : {precw:.4f}")
    print(f"Weighted F1-score  : {f1w:.4f}")
    return acc, precw, f1w

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
t_acc, t_prec, t_f1 = print_cls_block("Type", yt_te, yt_pred)

print_intensity_block()

summary_rows = [
    {"Prediction Task": "Frequency", "Accuracy": round(float(f_acc),4), "Weighted Precision": round(float(f_prec),4), "Weighted F1-score": round(float(f_f1),4), "MAE": None, "RMSE": None, "R2": None},
    {"Prediction Task": "Duration",  "Accuracy": round(float(d_acc),4), "Weighted Precision": round(float(d_prec),4), "Weighted F1-score": round(float(d_f1),4), "MAE": None, "RMSE": None, "R2": None},
    {"Prediction Task": "Type",      "Accuracy": round(float(t_acc),4), "Weighted Precision": round(float(t_prec),4), "Weighted F1-score": round(float(t_f1),4), "MAE": None, "RMSE": None, "R2": None},
    {"Prediction Task": "Intensity (bpm)", "Accuracy": None, "Weighted Precision": None, "Weighted F1-score": None, "MAE": round(float(best_mae),4), "RMSE": round(float(best_rmse),4), "R2": round(float(best_r2),4)}
]
summary_df = pd.DataFrame(summary_rows, columns=["Prediction Task","Accuracy","Weighted Precision","Weighted F1-score","MAE","RMSE","R2"])

print("\n=== Overall Model Performance ===")
print(summary_df.fillna("—").to_string(index=False))

# Save outputs
summary_df.to_csv("model_metrics_summary.csv", index=False)
print("\nSaved: model_metrics_summary.csv")

import joblib
joblib.dump(best_est, "intensity_best_model.joblib")
print("Saved: intensity_best_model.joblib")
