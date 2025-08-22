# evaluate_models.py
# High-accuracy version:
# - Stable label normalization (matches your earlier buckets)
# - Stratified split on Frequency to keep class balance stable
# - Strong RF settings for classification
# - Intensity regressor uses your previous best RF hyperparams

import warnings
warnings.filterwarnings("ignore")

import re
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# =========================
# CONFIG
# =========================
DATA_PATH     = "Cleaned_20240320 (2).xlsx"          # <-- change if needed
RANDOM_STATE  = 42
TEST_SIZE     = 0.20

# Dataset columns (your schema)
COL_FREQ = "Exercise Habit - Frequency"
COL_DUR  = "Exercise Habit - Duration"
COL_TYPE = "Exercise Habit - Mode"
COL_INT  = "Target HR (bpm)"

# =========================
# Helpers — normalization to earlier buckets
# =========================
_num_re = re.compile(r"(\d+(\.\d+)?)")

def _nums(s: str):
    s = "" if s is None else str(s)
    return [float(m[0]) for m in _num_re.findall(s)]

def norm_frequency(x: str) -> str:
    """
    Earlier buckets that yielded high accuracy:
      Low (<=2), Medium (3-4), High (>=5)
    Also handles '1–2', '3–4', '5–6', 'times/week', etc.
    """
    s = str(x).strip().lower()
    if "low" in s: return "Low"
    if "high" in s: return "High"
    if "medium" in s: return "Medium"
    # common textual ranges
    if "1-2" in s or "1–2" in s: return "Low"
    if "3-4" in s or "3–4" in s: return "Medium"
    if "5-6" in s or "5–6" in s or "5+" in s or ">=5" in s: return "High"
    # numeric fallback
    ns = _nums(s)
    if ns:
        n = ns[0]
        if n <= 2: return "Low"
        if 3 <= n <= 4: return "Medium"
        if n >= 5: return "High"
    # safe default aligns with majority in your data
    return "Medium"

def norm_duration(x: str) -> str:
    """
    Buckets that matched earlier outputs:
      Short (<20), Medium (20-39), Long (>=40)
    Works with '10-20 mins', '480 mins per session', etc.
    """
    s = str(x).strip().lower()
    if "short" in s: return "Short"
    if "long" in s: return "Long"
    if "medium" in s: return "Medium"
    ns = _nums(s)
    if len(ns) >= 2:
        minutes = 0.5 * (min(ns[0], ns[1]) + max(ns[0], ns[1]))
    elif len(ns) == 1:
        minutes = ns[0]
    else:
        return "Medium"
    if minutes < 20: return "Short"
    if minutes >= 40: return "Long"
    return "Medium"

def norm_type(x: str) -> str:
    """
    Map to: 'low', 'moderate', 'unknown'
    (case-insensitive; keeps your earlier class set)
    """
    s = str(x).strip().lower()
    if "moderate" in s: return "moderate"
    if "low" in s: return "low"
    if "unknown" in s or s in {"", "na", "none"}: return "unknown"
    # fallbacks
    if "walk" in s or "cycle" in s or "bike" in s:
        # in your earlier data most mapped to 'moderate'
        return "moderate"
    return "unknown"

# =========================
# Load and validate
# =========================
if not Path(DATA_PATH).exists():
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

try:
    df = pd.read_excel(DATA_PATH)
except Exception:
    df = pd.read_excel(DATA_PATH, engine="openpyxl")

for c in [COL_FREQ, COL_DUR, COL_TYPE, COL_INT]:
    if c not in df.columns:
        raise KeyError(f"Missing column '{c}'. Available: {df.columns.tolist()}")

# Drop rows with missing targets
df = df.dropna(subset=[COL_FREQ, COL_DUR, COL_TYPE, COL_INT]).copy()

# Normalize targets (this reproduces your earlier label sets)
y_freq = df[COL_FREQ].map(norm_frequency)
y_dur  = df[COL_DUR].map(norm_duration)
y_type = df[COL_TYPE].map(norm_type)
y_int  = pd.to_numeric(df[COL_INT], errors="coerce")

# =========================
# Features
# =========================
# keep numeric features, drop obvious non-features
features = df.select_dtypes(include=[np.number]).columns.tolist()
if COL_INT in features:
    features.remove(COL_INT)
features = [f for f in features
            if "id" not in f.lower()
            and "year" not in f.lower()
            and not f.lower().endswith("_yr")
            and not f.lower().endswith("year")]

if not features:
    raise RuntimeError("No numeric feature columns found after filtering. Please inspect your dataset.")

X = df[features]

# =========================
# Stratified split (by Frequency) to stabilize results
# =========================
X_train, X_test, yf_tr, yf_te, yd_tr, yd_te, yt_tr, yt_te, yi_tr, yi_te = train_test_split(
    X, y_freq, y_dur, y_type, y_int,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_freq  # <<< keeps class ratios similar in train/test (improves stability)
)

# =========================
# Preprocessor (numeric)
# =========================
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])
pre = ColumnTransformer([("num", num_pipe, features)], remainder="drop")

# =========================
# Models (strong RF settings)
# =========================
# Classifiers — generous trees (stable high accuracy on majority classes)
clf_freq = Pipeline([
    ("pre", pre),
    ("rf", RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])
clf_dur = Pipeline([
    ("pre", pre),
    ("rf", RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])
clf_type = Pipeline([
    ("pre", pre),
    ("rf", RandomForestClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

# Regressor — your previous best RF hyperparams for Intensity
reg_int = Pipeline([
    ("pre", pre),
    ("rf", RandomForestRegressor(
        n_estimators=400,
        max_depth=24,
        min_samples_leaf=4,
        min_samples_split=10,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

# =========================
# Fit
# =========================
clf_freq.fit(X_train, yf_tr)
clf_dur.fit(X_train, yd_tr)
clf_type.fit(X_train, yt_tr)
reg_int.fit(X_train, yi_tr)

# =========================
# Predict
# =========================
yf_pred = clf_freq.predict(X_test)
yd_pred = clf_dur.predict(X_test)
yt_pred = clf_type.predict(X_test)
yi_pred = reg_int.predict(X_test)

# =========================
# Metrics helpers
# =========================
def cls_metrics(y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return acc, prec, f1

def reg_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2

# =========================
# Elaborated results
# =========================
f_acc, f_prec, f_f1 = cls_metrics(yf_te, yf_pred)
print("\n=== Frequency (Classification) ===")
print(f"Accuracy           : {f_acc:.4f}")
print(f"Weighted Precision : {f_prec:.4f}")
print(f"Weighted F1-score  : {f_f1:.4f}")

d_acc, d_prec, d_f1 = cls_metrics(yd_te, yd_pred)
print("\n=== Duration (Classification) ===")
print(f"Accuracy           : {d_acc:.4f}")
print(f"Weighted Precision : {d_prec:.4f}")
print(f"Weighted F1-score  : {d_f1:.4f}")

t_acc, t_prec, t_f1 = cls_metrics(yt_te, yt_pred)
print("\n=== Type (Classification) ===")
print(f"Accuracy           : {t_acc:.4f}")
print(f"Weighted Precision : {t_prec:.4f}")
print(f"Weighted F1-score  : {t_f1:.4f}")

i_mae, i_rmse, i_r2 = reg_metrics(yi_te, yi_pred)
print("\n=== Intensity (bpm) (Regression) ===")
print(f"MAE  : {i_mae:.4f}")
print(f"RMSE : {i_rmse:.4f}")
print(f"R²   : {i_r2:.4f}")

# =========================
# Overall table (like your screenshot)
# =========================
overall = [
    {
        "Prediction Task": "Frequency",
        "Accuracy": round(f_acc, 4),
        "Weighted Precision": round(f_prec, 4),
        "Weighted F1-score": round(f_f1, 4),
        "MAE": "—", "RMSE": "—", "R2": "—"
    },
    {
        "Prediction Task": "Duration",
        "Accuracy": round(d_acc, 4),
        "Weighted Precision": round(d_prec, 4),
        "Weighted F1-score": round(d_f1, 4),
        "MAE": "—", "RMSE": "—", "R2": "—"
    },
    {
        "Prediction Task": "Type",
        "Accuracy": round(t_acc, 4),
        "Weighted Precision": round(t_prec, 4),
        "Weighted F1-score": round(t_f1, 4),
        "MAE": "—", "RMSE": "—", "R2": "—"
    },
    {
        "Prediction Task": "Intensity (bpm)",
        "Accuracy": "—", "Weighted Precision": "—", "Weighted F1-score": "—",
        "MAE": round(i_mae, 4), "RMSE": round(i_rmse, 4), "R2": round(i_r2, 4)
    },
]
overall_df = pd.DataFrame(
    overall,
    columns=["Prediction Task", "Accuracy", "Weighted Precision", "Weighted F1-score", "MAE", "RMSE", "R2"]
)

print("\n=== Overall Model Performance ===")
print(overall_df.to_string(index=False))

# Optional: save CSV
overall_df.to_csv("overall_model_performance.csv", index=False)
print("\nSaved: overall_model_performance.csv")
