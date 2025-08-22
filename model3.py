# evaluate_models_summary.py
# Produces a compact summary of overall metrics (no per-class breakdown):
# - Classification (Frequency, Duration, Type): Accuracy, Weighted Precision, Weighted F1
# - Regression (Intensity bpm): MAE, RMSE, R^2
#
# Output: pretty-printed table + model_metrics_summary.csv

import math
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# =========================
# CONFIG
# =========================
DATA_PATH = "Cleaned_20240320 (2).xlsx"  # put this file next to this script or change the path
TEST_SIZE = 0.2
RANDOM_STATE = 42

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
    freq_col = safe_col(df, ["frequency"], prefer_numeric=False)
    dur_col  = safe_col(df, ["duration"], prefer_numeric=False)
    type_col = safe_col(df, ["type", "mode"], prefer_numeric=False)
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
    # Use RMSE directly; keep compatible with all sklearn versions
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

# Preprocessor & Pipelines (same as app)
pre = make_preprocessor(features)
clf_freq = Pipeline([("pre", pre), ("rf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))])
clf_dur  = Pipeline([("pre", pre), ("rf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))])
clf_type = Pipeline([("pre", pre), ("rf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))])
reg_int  = Pipeline([("pre", pre), ("rf", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE))])

# =========================
# Fit
# =========================
clf_freq.fit(X_train, yf_tr_enc)
clf_dur.fit(X_train, yd_tr_enc)
clf_type.fit(X_train, yt_tr_enc)
reg_int.fit(X_train, yi_tr)

# =========================
# Predict & compute summary metrics
# =========================
summary_rows = []

# Classification helper
def add_cls_row(name, y_true_enc, y_pred_enc):
    acc = accuracy_score(y_true_enc, y_pred_enc)
    # Weighted precision/F1 (no per-class reporting)
    prec_w = precision_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)
    f1_w   = f1_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)
    summary_rows.append({
        "Prediction Task": name,
        "Accuracy": round(float(acc), 4),
        "Weighted Precision": round(float(prec_w), 4),
        "Weighted F1-score": round(float(f1_w), 4),
        "MAE": None, "RMSE": None, "R2": None
    })

# Regression helper
def add_reg_row(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mse(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    summary_rows.append({
        "Prediction Task": name,
        "Accuracy": None,
        "Weighted Precision": None,
        "Weighted F1-score": None,
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4)
    })

# Classification predictions
yf_pred_enc = clf_freq.predict(X_test)
yd_pred_enc = clf_dur.predict(X_test)
yt_pred_enc = clf_type.predict(X_test)
add_cls_row("Frequency", yf_te_enc, yf_pred_enc)
add_cls_row("Duration",  yd_te_enc, yd_pred_enc)
add_cls_row("Type",      yt_te_enc, yt_pred_enc)

# Regression predictions
yi_pred = reg_int.predict(X_test)
add_reg_row("Intensity (bpm)", yi_te, yi_pred)

# =========================
# Print + Save summary
# =========================
summary_df = pd.DataFrame(summary_rows, columns=[
    "Prediction Task", "Accuracy", "Weighted Precision", "Weighted F1-score", "MAE", "RMSE", "R2"
])

print("\n=== Overall Model Performance ===")
print(summary_df.fillna("—").to_string(index=False))

out_path = "model_metrics_summary.csv"
summary_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
