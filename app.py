# app.py
# Cardiac Rehab Recommendation System (Weeks 1–6) with:
# - Centered title
# - Counterfactual What-if
# - SHAP with a compute button
# - SAFETY RULE: If any Muscle Power is 0/1/2 → show consult message and do not proceed to Week 2
# - Editable cards for Week 1–6 with friendly labels in the edit dropdowns
# - NEW: Weeks 2–6 are inside collapsible expanders

import math
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# CONFIG
DATA_PATH = "Cleaned_20240320 (2).xlsx"
TARGET_FALLBACK_FREQ = "Frequency"
TARGET_FALLBACK_DUR  = "Duration"
TARGET_FALLBACK_TYPE = "Type"
TARGET_FALLBACK_INT  = "Target HR (bpm)"

st.set_page_config(page_title="Cardiac Rehab Recommendation System", page_icon="🫀", layout="wide")

# Styles =====
st.markdown("""
<style>
.small-card{font-size:0.95rem;padding:.6rem .8rem;border:1px solid #e5e7eb;border-radius:.6rem;margin:.35rem 0;background:#fafafa;}
.small-card h4{margin:0 0 .35rem 0;font-size:1rem;}
.small-line{margin:.12rem 0;}
.section-title{margin:0 0 .4rem 0;}
.group{border:1px solid #e5e7eb;border-radius:.6rem;padding:.7rem .9rem;margin:.5rem 0;background:#fcfcfc}
.group-title{display:flex;align-items:center;gap:.5rem;margin:0 0 .4rem 0;font-weight:700;font-size:1rem;color:#111827}
.group-title:before{content:"";}
h1.title-center{
  text-align:center;
  font-size:2.2rem;
  line-height:1.25;
  margin-top:.2rem;
  margin-bottom:.8rem;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Helpers
# ==============================
FREQ_LEVELS = ["Low", "Medium", "High"]
FREQ_DISPLAY_MAP = {"Low": "1–2 sessions/week", "Medium": "3–4 sessions/week", "High": "5–6 sessions/week"}
DUR_LEVELS = ["Short", "Medium", "Long"]
DUR_DISPLAY_MAP  = {"Short": "10–20 mins", "Medium": "20–40 mins", "Long": "40–60 mins"}
ALLOWED_TYPES    = {"Walking", "Jogging", "Cycling"}

FRIENDLY_MAP = {
    "Resting HR": "Resting Heart Rate (bpm)",
    "Resting BP": "Resting Blood Pressure (mmHg)",
    "Target HR (bpm)": "Target Heart Rate (bpm)",
    "Target HR (%)": "Target Heart Rate (%)",
    "Recumbent Bike: MHR": "Recumbent Bike · Max Heart Rate (bpm)",
    "Recumbent Bike: RPE": "Recumbent Bike · Rate of Perceived Exertion",
    "Recumbent Bike: Duration": "Recumbent Bike · Duration (min)",
    "Exercise Habit - Frequency": "Exercise Habit – Frequency (sessions/week)",
    "UL": "Upper Limb",
    "LL": "Lower Limb",
}
def friendly_name(col: str) -> str:
    if col in FRIENDLY_MAP: return FRIENDLY_MAP[col]
    name = col.replace("_", " ").replace("-", " ").strip()
    name = name.replace(" HR", " Heart Rate").replace("hr", "Heart Rate")
    name = name.replace(" BP", " Blood Pressure").replace("bp", "Blood Pressure")
    name = name.replace("MHR", "Max Heart Rate")
    name = name.replace(" UL ", " Upper Limb ").replace(" LL ", " Lower Limb ")
    words = []
    for w in name.split():
        if w.lower() in {"bpm","mmhg","rpe","%","ul","ll"}: words.append(w.upper())
        else: words.append(w.capitalize())
    pretty = " ".join(words)
    pretty = pretty.replace(" ( Bpm", " (bpm").replace(" ( Mmhg", " (mmHg").replace("  ", " ")
    return pretty

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

def display_freq(label: str): return FREQ_DISPLAY_MAP.get(ensure_freq_bins(label), "3–4 sessions/week")
def display_dur(label: str):  return DUR_DISPLAY_MAP.get(ensure_dur_bins(label), "20–40 mins")

def coerce_type(x: str):
    t = str(x).strip().title()
    return t if t in ALLOWED_TYPES else "Walking"

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

def fit_label_encoder(series: pd.Series):
    le = LabelEncoder()
    le.fit(series.astype(str).fillna("Unknown").values)
    return le

def coerce_row_from_widgets(values_dict, df, features):
    row = {}
    for c in features:
        row[c] = int(pd.to_numeric(pd.Series([values_dict.get(c)]), errors="coerce").fillna(0).iloc[0])
    return pd.DataFrame([row], columns=features)

def freq_band_from_count(n: int) -> str:
    if n <= 2: return "Low"
    if n >= 5: return "High"
    return "Medium"

def dur_band_from_minutes(m: int) -> str:
    if m <= 20: return "Short"
    if m >= 40: return "Long"
    return "Medium"

def move_one_step_toward(current_label: str, target_label: str, levels: list[str]) -> str:
    cur = levels.index(current_label); tgt = levels.index(target_label)
    if tgt > cur: return levels[min(cur+1, len(levels)-1)]
    if tgt < cur: return levels[max(cur-1, 0)]
    return levels[cur]

def adapt_intensity(prev_rec_bpm: int, actual_bpm: int | None, w1_bpm: int | None = None) -> int:
    base_target = int(actual_bpm) if actual_bpm is not None else int(prev_rec_bpm)
    blended = round(0.7 * int(prev_rec_bpm) + 0.3 * base_target)
    return int(max(40, min(220, blended)))

def _choose_type(prev_type: str, actual_type: str | None):
    a = coerce_type(actual_type) if actual_type else None
    if a in ALLOWED_TYPES: return a
    t = coerce_type(prev_type)
    return "Jogging" if t == "Jogging" else "Walking"

def week2_from_week1_and_actual(w1_pred, w1_actual):
    intensity_bpm = adapt_intensity(w1_pred["intensity_bpm"], w1_actual.get("actual_int_b"), w1_pred["intensity_bpm"])
    base_freq = "Low" if ensure_freq_bins(w1_pred["freq_label"]) == "High" else ensure_freq_bins(w1_pred["freq_label"])
    actual_freq = freq_band_from_count(int(w1_actual.get("actual_freq_n", 3)))
    freq_label = move_one_step_toward(base_freq, actual_freq, FREQ_LEVELS)
    freq_text = display_freq(freq_label).replace("sessions", "times")
    w1_dur = ensure_dur_bins(w1_pred["dur_label"])
    base_dur = "Custom(15–30)" if w1_dur in {"Long", "Medium"} else "Short"
    actual_dur = dur_band_from_minutes(int(w1_actual.get("actual_dur_m", 30)))
    if base_dur == "Custom(15–30)":
        dur_label, dur_text = ("Medium", "20–40 mins") if actual_dur == "Long" else ("Custom(15–30)", "15–30 mins")
    else:
        new_dur = move_one_step_toward(ensure_dur_bins(base_dur), actual_dur, DUR_LEVELS)
        dur_label, dur_text = new_dur, display_dur(new_dur)
    typ = _choose_type(w1_pred["type_label"], w1_actual.get("actual_type"))
    return {"freq_label": freq_label, "freq_text": freq_text,
            "dur_label": dur_label, "dur_text": dur_text,
            "intensity_bpm": intensity_bpm, "type_label": typ}

def week3_from_week2_and_actual(w2_pred, w2_actual, w1_pred):
    intensity_bpm = adapt_intensity(w2_pred["intensity_bpm"], w2_actual.get("actual_int_b"), w1_pred["intensity_bpm"])
    w2f = ensure_freq_bins(w2_pred["freq_label"])
    base_freq = "Medium" if w2f == "High" else w2f
    actual_freq = freq_band_from_count(int(w2_actual.get("actual_freq_n", 3)))
    freq_label = move_one_step_toward(base_freq, actual_freq, FREQ_LEVELS)
    freq_text = display_freq(freq_label)
    w2d_raw = str(w2_pred.get("dur_label", "Medium"))
    if "Custom(15–30)" in w2d_raw:
        base_dur_label, base_dur_text = "Custom(15–30)", "15–30 mins"
    elif ensure_dur_bins(w2d_raw) == "Long":
        base_dur_label, base_dur_text = "Medium", "20–40 mins"
    else:
        base_dur_label, base_dur_text = ensure_dur_bins(w2d_raw), display_dur(ensure_dur_bins(w2d_raw))
    actual_dur = dur_band_from_minutes(int(w2_actual.get("actual_dur_m", 30)))
    if base_dur_label == "Custom(15–30)":
        dur_label, dur_text = ("Medium", "20–40 mins") if actual_dur == "Long" else ("Custom(15–30)", "15–30 mins")
    else:
        new_dur = move_one_step_toward(base_dur_label, actual_dur, DUR_LEVELS)
        dur_label, dur_text = new_dur, display_dur(new_dur)
    typ = _choose_type(w2_pred["type_label"], w2_actual.get("actual_type"))
    return {"freq_label": freq_label, "freq_text": freq_text,
            "dur_label": dur_label, "dur_text": dur_text,
            "intensity_bpm": intensity_bpm, "type_label": typ}

def weekN_followup(prev_pred, prev_actual, w1_pred, clamp_long_to_medium=True):
    intensity_bpm = adapt_intensity(prev_pred["intensity_bpm"], prev_actual.get("actual_int_b"), w1_pred["intensity_bpm"])
    prev_freq = ensure_freq_bins(prev_pred["freq_label"])
    actual_freq = freq_band_from_count(int(prev_actual.get("actual_freq_n", 3)))
    freq_label = move_one_step_toward(prev_freq, actual_freq, FREQ_LEVELS)
    freq_text = display_freq(freq_label)
    prev_dur_raw = str(prev_pred.get("dur_label", "Medium"))
    if "Custom(15–30)" in prev_dur_raw:
        base_dur_label, base_dur_text = "Custom(15–30)", "15–30 mins"
    elif clamp_long_to_medium and ensure_dur_bins(prev_dur_raw) == "Long":
        base_dur_label, base_dur_text = "Medium", "20–40 mins"
    else:
        base_dur_label, base_dur_text = ensure_dur_bins(prev_dur_raw), display_dur(ensure_dur_bins(prev_dur_raw))
    actual_dur = dur_band_from_minutes(int(prev_actual.get("actual_dur_m", 30)))
    if base_dur_label == "Custom(15–30)":
        dur_label, dur_text = ("Medium", "20–40 mins") if actual_dur == "Long" else ("Custom(15–30)", "15–30 mins")
    else:
        new_dur = move_one_step_toward(base_dur_label, actual_dur, DUR_LEVELS)
        dur_label, dur_text = new_dur, display_dur(new_dur)
    typ = _choose_type(prev_pred["type_label"], prev_actual.get("actual_type"))
    return {"freq_label": freq_label, "freq_text": freq_text,
            "dur_label": dur_label, "dur_text": dur_text,
            "intensity_bpm": intensity_bpm, "type_label": typ}

# ==============================
# Load data & pre-train models
# ==============================
st.markdown('<h1 class="title-center"> AI-Driven Personalized Cardiac Rehabilitation Plan System </h1>', unsafe_allow_html=True)

@st.cache_data(show_spinner=True)
def load_df(path):
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.read_excel(path, engine="openpyxl")

try:
    df = load_df(DATA_PATH)
except Exception as e:
    st.error(f"Couldn't load dataset at `{DATA_PATH}`. Error: {e}")
    st.stop()

freq_col, dur_col, type_col, int_col = guess_targets(df)
missing = [n for n,v in zip(["Frequency","Duration","Type","Intensity"], [freq_col,dur_col,type_col,int_col]) if v is None]
if missing:
    st.error(f"Could not auto-detect target columns: {', '.join(missing)}.")
    st.stop()

def _clean_cat_target(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip()
    s2 = s2.replace({"": np.nan, "na": np.nan, "Na": np.nan, "NA": np.nan, "none": np.nan, "None": np.nan, "Nan": np.nan})
    return s2

cat_freq_raw = _clean_cat_target(df[freq_col]); cat_dur_raw = _clean_cat_target(df[dur_col])
cat_type_raw = _clean_cat_target(df[type_col]); num_int_raw = pd.to_numeric(df[int_col], errors="coerce")

mask_valid = (cat_freq_raw.notna() & cat_dur_raw.notna() & cat_type_raw.notna() & num_int_raw.notna())
df_train = df.loc[mask_valid].copy()
cat_freq_raw = cat_freq_raw.loc[mask_valid]; cat_dur_raw = cat_dur_raw.loc[mask_valid]
cat_type_raw = cat_type_raw.loc[mask_valid]; num_int_raw = num_int_raw.loc[mask_valid]

if len(df_train) < 20:
    st.error("Too few rows with complete targets to train models.")
    st.stop()

features = build_feature_list(df_train, [freq_col, dur_col, type_col, int_col])
if not features:
    st.error("No numeric feature columns found for training.")
    st.stop()

y_freq_raw = cat_freq_raw.astype(str).apply(ensure_freq_bins)
y_dur_raw  = cat_dur_raw.astype(str).apply(ensure_dur_bins)
y_type_raw = cat_type_raw.astype(str)
y_int      = num_int_raw.astype(float)

le_freq = fit_label_encoder(y_freq_raw); le_dur  = fit_label_encoder(y_dur_raw); le_type = fit_label_encoder(y_type_raw)
y_freq = le_freq.transform(y_freq_raw);  y_dur  = le_dur.transform(y_dur_raw);  y_type = le_type.transform(y_type_raw)

pre = make_preprocessor(features)
clf_freq = Pipeline([("pre", pre), ("rf", RandomForestClassifier(n_estimators=300, random_state=42))])
clf_dur  = Pipeline([("pre", pre), ("rf", RandomForestClassifier(n_estimators=300, random_state=42))])
clf_type = Pipeline([("pre", pre), ("rf", RandomForestClassifier(n_estimators=300, random_state=42))])
reg_int  = Pipeline([("pre", pre), ("rf", RandomForestRegressor(n_estimators=300, random_state=42))])

X = df_train[features]
clf_freq.fit(X, y_freq); clf_dur.fit(X, y_dur); clf_type.fit(X, y_type); reg_int.fit(X, y_int)

st.session_state["models"] = {
    "clf_freq": clf_freq, "clf_dur": clf_dur, "clf_type": clf_type, "reg_int": reg_int,
    "le_freq": le_freq, "le_dur": le_dur, "le_type": le_type,
    "features": features, "df_train": df_train
}

# ==============================
# Render helpers (original static + new editable)
# ==============================
def render_week_card(title: str, rec: dict):
    freq_text = rec.get("freq_text", display_freq(rec["freq_label"]))
    dur_text  = rec.get("dur_text",  display_dur(rec["dur_label"]))
    html = f"""
    <div class="small-card">
      <h4 class="section-title">{title}</h4>
      <div class="small-line"><b>Frequency:</b> {freq_text}</div>
      <div class="small-line"><b>Duration:</b> {dur_text}</div>
      <div class="small-line"><b>Intensity:</b> {int(rec['intensity_bpm'])} bpm</div>
      <div class="small-line"><b>Type:</b> {rec['type_label']}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ===== NEW: Editable cards with friendly dropdown labels =====
def _normalize_rec(rec: dict) -> dict:
    rec = dict(rec)
    rec["freq_label"] = ensure_freq_bins(rec.get("freq_label", "Medium"))
    dur_raw = str(rec.get("dur_label", "Medium"))
    if "Custom(15–30)" in dur_raw or "15-30" in dur_raw:
        rec["dur_label"] = "Custom(15–30)"
        rec["dur_text"]  = "15–30 mins"
    else:
        rec["dur_label"] = ensure_dur_bins(dur_raw)
        rec["dur_text"]  = display_dur(rec["dur_label"])
    rec["freq_text"]     = display_freq(rec["freq_label"])
    rec["type_label"]    = coerce_type(rec.get("type_label", "Walking"))
    rec["intensity_bpm"] = int(rec.get("intensity_bpm", 100))
    return rec

# (label_value, label_shown_to_user)
FREQ_EDIT_PAIRS = [
    ("Low",    "Low: 1–2 sessions/week"),
    ("Medium", "Medium: 3–4 sessions/week"),
    ("High",   "High: 5–6 sessions/week"),
]
DUR_EDIT_PAIRS = [
    ("Short",          "Short: 10–20 mins"),
    ("Custom(15–30)",  "Custom: 15–30 mins"),
    ("Medium",         "Medium: 20–40 mins"),
    ("Long",           "Long: 40–60 mins"),
]
FREQ_EDIT_OPTIONS = [shown for _, shown in FREQ_EDIT_PAIRS]
DUR_EDIT_OPTIONS  = [shown for _, shown in DUR_EDIT_PAIRS]
FREQ_SHOWN_TO_VAL = {shown: val for val, shown in FREQ_EDIT_PAIRS}
DUR_SHOWN_TO_VAL  = {shown: val for val, shown in DUR_EDIT_PAIRS}
FREQ_VAL_TO_INDEX = {val: i for i, (val, _) in enumerate(FREQ_EDIT_PAIRS)}
DUR_VAL_TO_INDEX  = {val: i for i, (val, _) in enumerate(DUR_EDIT_PAIRS)}

def render_week_card_editable(title: str, week_key: str):
    """
    week_key: 'week1'...'week6'.
    Uses st.session_state[f'{week_key}_pred'] and st.session_state[f'edit_{week_key}'].
    """
    pred_key = f"{week_key}_pred"
    edit_key = f"edit_{week_key}"
    if pred_key not in st.session_state:
        return

    st.session_state[pred_key] = _normalize_rec(st.session_state[pred_key])
    rec = st.session_state[pred_key]

    if not st.session_state.get(edit_key, False):
        # VIEW MODE
        html = f"""
        <div class="small-card">
          <h4 class="section-title">{title}</h4>
          <div class="small-line"><b>Frequency:</b> {rec['freq_text']}</div>
          <div class="small-line"><b>Duration:</b> {rec['dur_text']}</div>
          <div class="small-line"><b>Intensity:</b> {int(rec['intensity_bpm'])} bpm</div>
          <div class="small-line"><b>Type:</b> {rec['type_label']}</div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
        if st.button("Edit", key=f"btn_edit_{week_key}"):
            st.session_state[edit_key] = True
            st.rerun()
    else:
        # EDIT MODE
        with st.form(f"edit_form_{week_key}", clear_on_submit=False):
            c1, c2 = st.columns(2)

            # Frequency using friendly labels
            freq_index = FREQ_VAL_TO_INDEX.get(rec["freq_label"], 1)
            shown_freq = c1.selectbox(
                "Frequency",
                FREQ_EDIT_OPTIONS,
                index=freq_index,
                key=f"freq_sel_{week_key}",
            )

            # Duration using friendly labels (with Custom option)
            dur_val = rec["dur_label"] if rec["dur_label"] in DUR_VAL_TO_INDEX else ensure_dur_bins(rec["dur_label"])
            dur_index = DUR_VAL_TO_INDEX.get(dur_val, DUR_VAL_TO_INDEX["Medium"])
            shown_dur = c2.selectbox(
                "Duration",
                DUR_EDIT_OPTIONS,
                index=dur_index,
                key=f"dur_sel_{week_key}",
            )

            new_int = st.number_input(
                "Intensity (bpm)",
                min_value=40, max_value=220, step=1,
                value=int(rec["intensity_bpm"]),
                key=f"int_num_{week_key}"
            )
            new_type = st.selectbox(
                "Type",
                sorted(list(ALLOWED_TYPES)),
                index=sorted(list(ALLOWED_TYPES)).index(coerce_type(rec["type_label"])),
                key=f"type_sel_{week_key}",
            )

            save = st.form_submit_button("Save")
            cancel = st.form_submit_button("Cancel")

        if save:
            # Map back from shown labels to canonical values
            new_freq_val = FREQ_SHOWN_TO_VAL[shown_freq]
            new_dur_val  = DUR_SHOWN_TO_VAL[shown_dur]

            rec["freq_label"]    = ensure_freq_bins(new_freq_val)
            rec["dur_label"]     = new_dur_val
            rec["intensity_bpm"] = int(new_int)
            rec["type_label"]    = coerce_type(new_type)

            rec["freq_text"]     = display_freq(rec["freq_label"])
            rec["dur_text"]      = "15–30 mins" if new_dur_val == "Custom(15–30)" else display_dur(ensure_dur_bins(new_dur_val))

            st.session_state[pred_key] = rec
            st.session_state[edit_key] = False
            st.success("Saved.")
            st.rerun()

        if cancel:
            st.session_state[edit_key] = False
            st.rerun()

# ==============================
# Predictors
# ==============================
def predict_week1_from_features(feature_values: dict) -> dict:
    m = st.session_state["models"]
    X_row = coerce_row_from_widgets(feature_values, m["df_train"], m["features"])
    y_freq_idx = m["clf_freq"].predict(X_row)[0]
    y_dur_idx  = m["clf_dur"].predict(X_row)[0]
    y_type_idx = m["clf_type"].predict(X_row)[0]
    y_int_val  = m["reg_int"].predict(X_row)[0]
    freq_label = ensure_freq_bins(m["le_freq"].inverse_transform([int(y_freq_idx)])[0])
    dur_label  = ensure_dur_bins(m["le_dur"].inverse_transform([int(y_dur_idx)])[0])
    type_label = coerce_type(m["le_type"].inverse_transform([int(y_type_idx)])[0])
    intensity_bpm = int(round(y_int_val))
    return {"freq_label": freq_label, "dur_label": dur_label,
            "intensity_bpm": intensity_bpm, "type_label": type_label,
            "features_input": dict(feature_values)}

# ==============================
# Layout
# ==============================
left, right = st.columns([1.1, 0.9])

def bounds_for_feature(name: str, default_low: int, default_high: int):
    n = name.lower()
    if "age" in n:                    return 18, 100
    if "muscle power" in n:           return 1, 5
    if "duration" in n:               return 0, 120
    if "rpe" in n:                    return 1, 10
    if "%" in name or "target hr (%)" in n or "percent" in n:  return 10, 100
    if "recumbent" in n and ("mhr" in n or "max heart rate" in n or "max hr" in n): return 50, 220
    if "max heart rate" in n and "recumbent" not in n:         return 50, 220
    if "target hr" in n and "bpm" in n:                        return 40, 220
    return int(default_low), int(default_high)

with left:
    # ==============================
    # WEEK 1 — Inputs -> Recommendation
    # ==============================
    st.markdown("### Week 1 - Enter Patient Details")

    with st.form("week1_form", clear_on_submit=False):
        inputs = {}
        m = st.session_state["models"]
        feat_set = set(m["features"])

        age_feat = next((f for f in m["features"] if "age" in f.lower()), None)
        gender_feat = next((f for f in feat_set if "gender" in f.lower() or "sex" in f.lower()), None)
        smoking_feat = next((f for f in feat_set if "smok" in f.lower()), None)

        c1, c2, c3 = st.columns(3)
        age_default = 40
        if age_feat:
            s_age = m["df_train"][age_feat]
            med_age = float(s_age.median()) if not math.isnan(s_age.median()) else 40.0
            age_default = int(round(min(max(med_age, 18), 100)))
        age_val = c1.number_input("Age", min_value=18, max_value=100, value=int(age_default), step=1, format="%d")
        if age_feat: inputs[age_feat] = int(age_val)

        gender_val = c2.selectbox("Gender", ["Male", "Female"])
        if gender_feat: inputs[gender_feat] = 1 if gender_val == "Male" else 0

        smoking_val = c3.selectbox("Smoking Status", ["No", "Yes", "Ex-smoker"])
        if smoking_feat: inputs[smoking_feat] = {"No":0, "Yes":1, "Ex-smoker":2}[smoking_val]

        # Recumbent Bike group
        st.markdown('<div class="group"><div class="group-title">Recumbent Bike</div>', unsafe_allow_html=True)
        rec_feats = [f for f in m["features"] if "recumbent" in f.lower()]
        rb_mhr = next((f for f in rec_feats if ("mhr" in f.lower() or "max heart rate" in f.lower() or "max hr" in f.lower())), None)
        rb_dur = next((f for f in rec_feats if "duration" in f.lower()), None)
        rb_rpe = next((f for f in rec_feats if "rpe" in f.lower()), None)

        shown_in_group = set()
        g1, g2, g3 = st.columns(3)
        if rb_mhr:
            s = m["df_train"][rb_mhr]; med = float(s.median()) if not math.isnan(s.median()) else 120.0
            val = g1.number_input(friendly_name(rb_mhr), min_value=50, max_value=220, value=int(round(min(max(med,50),220))), step=1, format="%d")
            inputs[rb_mhr] = int(val); shown_in_group.add(rb_mhr)
        if rb_dur:
            s = m["df_train"][rb_dur]; med = float(s.median()) if not math.isnan(s.median()) else 15.0
            val = g2.number_input(friendly_name(rb_dur), min_value=0, max_value=120, value=int(round(min(max(med,0),120))), step=1, format="%d")
            inputs[rb_dur] = int(val); shown_in_group.add(rb_dur)
        if rb_rpe:
            s = m["df_train"][rb_rpe]; med = float(s.median()) if not math.isnan(s.median()) else 5.0
            val = g3.number_input(friendly_name(rb_rpe), min_value=1, max_value=10, value=int(round(min(max(med,1),10))), step=1, format="%d")
            inputs[rb_rpe] = int(val); shown_in_group.add(rb_rpe)
        st.markdown('</div>', unsafe_allow_html=True)

        # Remaining numeric features
        cols = st.columns(2); idx = 0
        for feat in m["features"]:
            if feat in shown_in_group or feat in {age_feat, gender_feat, smoking_feat}: continue
            s = m["df_train"][feat]
            q10, q90 = float(s.quantile(0.10)), float(s.quantile(0.90))
            median = float(s.median()) if not math.isnan(s.median()) else float(s.mean())
            low = min(q10, median - (q90 - q10)); high = max(q90, median + (q90 - q10))
            if not np.isfinite(low) or not np.isfinite(high) or low == high:
                low, high = (0.0, float(median*2 if median else 100.0))
            low_i, high_i = bounds_for_feature(feat, math.floor(low), math.ceil(high))
            default = int(round(median if np.isfinite(median) else (low + high)/2))
            default = min(max(default, low_i), high_i)
            with cols[idx % 2]:
                label = friendly_name(feat)
                val = st.number_input(label, value=int(default), min_value=int(low_i), max_value=int(high_i), step=1, format="%d")
                inputs[feat] = int(val)
            idx += 1

        submit_w1 = st.form_submit_button("Generate Week 1 Recommendation")

    if submit_w1:
        # SAFETY RULE: block if any Muscle Power is 0/1/2
        muscle_vals = [v for k,v in inputs.items() if ("muscle" in k.lower() and "power" in k.lower())]
        if any(v in (0,1,2) for v in muscle_vals):
            for key in ["week1_pred","week2_pred","week3_pred","week4_pred","week5_pred","week6_pred",
                        "week1_actual","week2_actual","week3_actual","week4_actual","week5_actual"]:
                if key in st.session_state: del st.session_state[key]
            st.session_state["blocked_by_mpower"] = True
            st.error("⚠️ Please consult Dr for further assistance.")
        else:
            st.session_state["blocked_by_mpower"] = False
            st.session_state["week1_pred"] = predict_week1_from_features(inputs)

    if "week1_pred" in st.session_state:
        render_week_card_editable("Week 1 Exercise Recommendation", "week1")

    # ==============================
    # Weeks 2–6 (EXPANDERS)
    # ==============================
    with st.expander("Week 2 - Actual Week 1 Exercise Done", expanded=False):
        if "week1_pred" not in st.session_state or st.session_state.get("blocked_by_mpower"):
            st.info("Week 2 will be available after a valid Week 1 recommendation.")
        else:
            with st.form("week2_actual_form", clear_on_submit=False):
                a_f = st.number_input("Week 1 - Actual Frequency (sessions/week)", min_value=1, max_value=6, value=3, step=1, format="%d")
                a_d = st.number_input("Week 1 - Actual Duration (minutes)", min_value=5, max_value=180, value=30, step=1, format="%d")
                a_i = st.number_input("Week 1 - Actual Intensity (bpm)", min_value=40, max_value=220,
                                      value=int(st.session_state['week1_pred']['intensity_bpm']), step=1, format="%d")
                a_t = st.selectbox("Week 1 - Actual Type", sorted(list(ALLOWED_TYPES | {"Other"})))
                submit_w2 = st.form_submit_button("Generate Week 2 Recommendation")
            if submit_w2:
                w1_act = {"actual_freq_n": int(a_f), "actual_dur_m": int(a_d), "actual_int_b": int(a_i), "actual_type": a_t}
                st.session_state["week2_pred"] = week2_from_week1_and_actual(st.session_state["week1_pred"], w1_act)
                st.session_state["week1_actual"] = w1_act
            if "week2_pred" in st.session_state:
                render_week_card_editable("Week 2 Exercise Recommendation", "week2")

    with st.expander("Week 3 - Actual Week 2 Exercise Done", expanded=False):
        if "week2_pred" not in st.session_state:
            st.info("Generate a Week 2 recommendation first.")
        else:
            with st.form("week3_actual_form", clear_on_submit=False):
                a_f = st.number_input("Week 2 - Actual Frequency (sessions/week)", min_value=1, max_value=6, value=3, step=1, format="%d")
                a_d = st.number_input("Week 2 - Actual Duration (minutes)", min_value=5, max_value=180, value=30, step=1, format="%d")
                a_i = st.number_input("Week 2 - Actual Intensity (bpm)", min_value=40, max_value=220,
                                      value=int(st.session_state['week2_pred']['intensity_bpm']), step=1, format="%d")
                a_t = st.selectbox("Week 2 - Actual Type", sorted(list(ALLOWED_TYPES | {"Other"})))
                submit_w3 = st.form_submit_button("Generate Week 3 Recommendation")
            if submit_w3:
                w2_act = {"actual_freq_n": int(a_f), "actual_dur_m": int(a_d), "actual_int_b": int(a_i), "actual_type": a_t}
                st.session_state["week3_pred"] = week3_from_week2_and_actual(st.session_state["week2_pred"], w2_act, st.session_state["week1_pred"])
                st.session_state["week2_actual"] = w2_act
            if "week3_pred" in st.session_state:
                render_week_card_editable("Week 3 Exercise Recommendation", "week3")

    with st.expander("Week 4 - Actual Week 3 Exercise Done", expanded=False):
        if "week3_pred" not in st.session_state:
            st.info("Generate a Week 3 recommendation first.")
        else:
            with st.form("week4_actual_form", clear_on_submit=False):
                a_f = st.number_input("Week 3 - Actual Frequency (sessions/week)", min_value=1, max_value=6, value=3, step=1, format="%d")
                a_d = st.number_input("Week 3 - Actual Duration (minutes)", min_value=5, max_value=180, value=30, step=1, format="%d")
                a_i = st.number_input("Week 3 - Actual Intensity (bpm)", min_value=40, max_value=220,
                                      value=int(st.session_state['week3_pred']['intensity_bpm']), step=1, format="%d")
                a_t = st.selectbox("Week 3 - Actual Type", sorted(list(ALLOWED_TYPES | {"Other"})))
                submit_w4 = st.form_submit_button("Generate Week 4 Recommendation")
            if submit_w4:
                w3_act = {"actual_freq_n": int(a_f), "actual_dur_m": int(a_d), "actual_int_b": int(a_i), "actual_type": a_t}
                st.session_state["week4_pred"] = weekN_followup(st.session_state["week3_pred"], w3_act, st.session_state["week1_pred"], clamp_long_to_medium=True)
                st.session_state["week3_actual"] = w3_act
            if "week4_pred" in st.session_state:
                render_week_card_editable("Week 4 Exercise Recommendation", "week4")

    with st.expander("Week 5 - Actual Week 4 Exercise Done", expanded=False):
        if "week4_pred" not in st.session_state:
            st.info("Generate a Week 4 recommendation first.")
        else:
            with st.form("week5_actual_form", clear_on_submit=False):
                a_f = st.number_input("Week 4 - Actual Frequency (sessions/week)", min_value=1, max_value=6, value=3, step=1, format="%d")
                a_d = st.number_input("Week 4 - Actual Duration (minutes)", min_value=5, max_value=180, value=30, step=1, format="%d")
                a_i = st.number_input("Week 4 - Actual Intensity (bpm)", min_value=40, max_value=220,
                                      value=int(st.session_state['week4_pred']['intensity_bpm']), step=1, format="%d")
                a_t = st.selectbox("Week 4 - Actual Type", sorted(list(ALLOWED_TYPES | {"Other"})))
                submit_w5 = st.form_submit_button("Generate Week 5 Recommendation")
            if submit_w5:
                w4_act = {"actual_freq_n": int(a_f), "actual_dur_m": int(a_d), "actual_int_b": int(a_i), "actual_type": a_t}
                st.session_state["week5_pred"] = weekN_followup(st.session_state["week4_pred"], w4_act, st.session_state["week1_pred"], clamp_long_to_medium=True)
                st.session_state["week4_actual"] = w4_act
            if "week5_pred" in st.session_state:
                render_week_card_editable("Week 5 Exercise Recommendation", "week5")

    with st.expander("Week 6 - Actual Week 5 Exercise Done", expanded=False):
        if "week5_pred" not in st.session_state:
            st.info("Generate a Week 5 recommendation first.")
        else:
            with st.form("week6_actual_form", clear_on_submit=False):
                a_f = st.number_input("Week 5 - Actual Frequency (sessions/week)", min_value=1, max_value=6, value=3, step=1, format="%d")
                a_d = st.number_input("Week 5 - Actual Duration (minutes)", min_value=5, max_value=180, value=30, step=1, format="%d")
                a_i = st.number_input("Week 5 - Actual Intensity (bpm)", min_value=40, max_value=220,
                                      value=int(st.session_state['week5_pred']['intensity_bpm']), step=1, format="%d")
                a_t = st.selectbox("Week 5 - Actual Type", sorted(list(ALLOWED_TYPES | {"Other"})))
                submit_w6 = st.form_submit_button("Generate Week 6 Recommendation")
            if submit_w6:
                w5_act = {"actual_freq_n": int(a_f), "actual_dur_m": int(a_d), "actual_int_b": int(a_i), "actual_type": a_t}
                st.session_state["week6_pred"] = weekN_followup(st.session_state["week5_pred"], w5_act, st.session_state["week1_pred"], clamp_long_to_medium=True)
                st.session_state["week5_actual"] = w5_act
            if "week6_pred" in st.session_state:
                render_week_card_editable("Week 6 Exercise Recommendation", "week6")

with right:
    # ==============================
    # Counterfactual (What-if)
    # ==============================
    st.markdown("### What-if Simulator (Counterfactual)")
    if "week1_pred" not in st.session_state or st.session_state.get("blocked_by_mpower"):
        st.info("Generate a valid Week 1 recommendation first, then you can run counterfactuals.")
    else:
        m = st.session_state["models"]
        base_inputs = st.session_state["week1_pred"]["features_input"]
        friendly_to_feat = {friendly_name(f): f for f in m["features"]}
        feat_friendly_options = list(friendly_to_feat.keys())
        with st.form("cf_form", clear_on_submit=False):
            sel_friendly = st.selectbox("Select an input to change:", feat_friendly_options)
            cf_feat = friendly_to_feat[sel_friendly]
            s = m["df_train"][cf_feat]
            q10, q90 = float(s.quantile(0.10)), float(s.quantile(0.90))
            median = float(s.median()) if not math.isnan(s.median()) else float(s.mean())
            low = min(q10, median - (q90 - q10)); high = max(q90, median + (q90 - q10))
            if not np.isfinite(low) or not np.isfinite(high) or low == high:
                low, high = (0.0, float(median*2 if median else 100.0))
            def bounds_for_feature(name, dl, dh):
                n = name.lower()
                if "age" in n: return 18, 100
                if "muscle power" in n: return 1, 5
                if "duration" in n: return 0, 120
                if "rpe" in n: return 1, 10
                if "%" in name or "target hr (%)" in n: return 10, 100
                if "recumbent" in n and ("mhr" in n or "max heart rate" in n or "max hr" in n): return 50, 220
                if "max heart rate" in n and "recumbent" not in n: return 50, 220
                if "target hr" in n and "bpm" in n: return 40, 220
                return int(dl), int(dh)
            low_i, high_i = bounds_for_feature(cf_feat, math.floor(low), math.ceil(high))
            default_i = int(base_inputs.get(cf_feat, int(round(median if np.isfinite(median) else (low+high)/2))))
            default_i = min(max(default_i, low_i), high_i)
            new_val = st.number_input(f"What if **{sel_friendly}** were…", value=default_i,
                                      min_value=int(low_i), max_value=int(high_i), step=1, format="%d")
            run_cf = st.form_submit_button("Run What-if")
        if run_cf:
            base_w1 = predict_week1_from_features(base_inputs)
            cf_inputs = dict(base_inputs); cf_inputs[cf_feat] = int(new_val)
            cf_w1 = predict_week1_from_features(cf_inputs)
            st.write("**Baseline Recommendation**");       render_week_card("Baseline Recommendation", base_w1)
            st.write("**Counterfactual Recommendation**"); render_week_card("Counterfactual Recommendation", cf_w1)

    # ==============================
    # SHAP with dropdown + button
    # ==============================
    st.markdown("---")
    st.markdown("### SHAP Explainability")
    if "week1_pred" not in st.session_state or st.session_state.get("blocked_by_mpower"):
        st.info("Generate a valid Week 1 recommendation first to explain it with SHAP.")
    else:
        import shap, matplotlib.pyplot as plt

        def _to_scalar(x, default=0.0):
            try:
                return float(np.asarray(x).squeeze().item())
            except Exception:
                arr = np.asarray(x).squeeze()
                return float(arr.flat[0]) if arr.size else float(default)

        def _ravel1d(x):
            return np.asarray(x).reshape(-1)  # force 1-D

        m = st.session_state["models"]
        model_choice = st.selectbox("Select prediction to explain:",
                                    ["Intensity (bpm)", "Frequency", "Duration", "Type"])
        do_shap = st.button("Compute SHAP Explanation")

        if do_shap:
            try:
                if model_choice == "Intensity (bpm)":
                    pipe = m["reg_int"]; is_classifier = False; inv = None
                elif model_choice == "Frequency":
                    pipe = m["clf_freq"]; is_classifier = True; inv = m["le_freq"].inverse_transform
                elif model_choice == "Duration":
                    pipe = m["clf_dur"];  is_classifier = True; inv = m["le_dur"].inverse_transform
                else:
                    pipe = m["clf_type"]; is_classifier = True; inv = m["le_type"].inverse_transform

                pre: ColumnTransformer = pipe.named_steps["pre"]
                est = pipe.named_steps["rf"]

                X_train = m["df_train"][m["features"]]
                bg = pre.transform(X_train.sample(min(200, len(X_train)), random_state=42))
                x0_df = pd.DataFrame([st.session_state["week1_pred"]["features_input"]], columns=m["features"])
                x0 = pre.transform(x0_df)

                if is_classifier:
                    pred_idx = int(pipe.predict(x0_df)[0])
                    pred_label = inv([pred_idx])[0]
                else:
                    pred_val = _to_scalar(pipe.predict(x0_df)[0])

                explainer = shap.TreeExplainer(est, data=bg, feature_perturbation="interventional")
                shap_vals = explainer.shap_values(x0, check_additivity=False)

                raw_names = pre.get_feature_names_out(m["features"])
                nice_names = [friendly_name(n.split("__", 1)[-1]) for n in raw_names]
                nice_names = list(nice_names)

                if is_classifier:
                    if isinstance(shap_vals, list):
                        shap_vec = _ravel1d(shap_vals[pred_idx])
                        base_val = _to_scalar(np.asarray(explainer.expected_value)[pred_idx])
                    else:
                        shap_vec = _ravel1d(shap_vals)
                        base_val = _to_scalar(explainer.expected_value)
                    caption = f"Predicted: **{pred_label}** • Baseline = {base_val:.3f}"
                else:
                    shap_vec = _ravel1d(shap_vals)
                    base_val = _to_scalar(explainer.expected_value)
                    caption = f"Predicted intensity = **{int(pred_val)} bpm** • Baseline = {base_val:.1f}"

                x0_vals = _ravel1d(x0_df.iloc[0].values)
                L = min(len(nice_names), len(shap_vec), len(x0_vals))
                nice_names, shap_vec, x0_vals = nice_names[:L], shap_vec[:L], x0_vals[:L]

                contrib = pd.DataFrame({
                    "Feature": nice_names,
                    "SHAP value": shap_vec,
                    "Abs SHAP": np.abs(shap_vec),
                    "Value": x0_vals
                }).sort_values("Abs SHAP", ascending=False).reset_index(drop=True)

                st.caption(caption)

                topN = min(10, len(contrib))
                fig, ax = plt.subplots()
                sub = contrib.head(topN).iloc[::-1]
                ax.barh(sub["Feature"], sub["SHAP value"])
                ax.set_xlabel("SHAP value (impact on prediction)")
                ax.set_ylabel("")
                title = "Top factors driving predicted " + ("intensity" if not is_classifier else model_choice.lower())
                ax.set_title(title)
                st.pyplot(fig, clear_figure=True)

                st.dataframe(contrib.drop(columns=["Abs SHAP"]).head(25))
            except ModuleNotFoundError:
                st.warning("SHAP isn’t installed. Run `pip install shap` and reload the app.")
            except Exception as e:
                st.error(f"SHAP explanation failed: {e}")

# ==============================
# Patient Info + Counterfactual + SHAP Explanations (Bottom wording)
# ==============================
st.markdown("---")
st.markdown("### Get To Know...")
st.markdown("""
**Recumbent Bike · Max Heart Rate (MHR)**  
The highest heart rate you reached during the recumbent bike test — shows your peak effort.

**Recumbent Bike · Duration**  
How long you cycled in the test, in minutes. Longer often means better endurance.

**Recumbent Bike · RPE (Rate of Perceived Exertion)**  
How hard the exercise felt to you on a 1–10 scale (1 = very easy, 10 = maximum effort).

**Target Heart Rate (bpm)**  
Beats per minute we aim for during exercise to train safely and effectively.

**Target Heart Rate (%)**  
Your target intensity as a % of your estimated maximum heart rate (roughly 220 − age).

**Muscle Power (1–5)**  
1–2 = very weak/trace movement, 3 = lifts against gravity, 4–5 = good/normal strength.  
If any muscle power is ≤2, we advise medical review before progressing past Week 1.

**Resting Heart Rate**  
Your pulse while sitting quietly.

**Resting Blood Pressure**  
Your blood pressure at rest (mmHg), e.g., 120/80.
""")

st.markdown("### What-if Simulator (Counterfactual)")
st.markdown("""
A counterfactual lets you explore **“what if”** scenarios by changing one input to see how your Week-1
recommendation would change. For example, adjust **Target Heart Rate (%)** and see whether intensity,
frequency, or duration shifts. This helps you understand which inputs influence your plan the most.
""")

st.markdown("### SHAP Explainability")
st.markdown("""
SHAP (SHapley Additive exPlanations) shows which inputs most influenced a prediction.  
- **Positive SHAP value** → pushes the prediction higher  
- **Negative SHAP value** → pushes the prediction lower  
The chart ranks top drivers; the table shows SHAP values alongside your actual inputs.  
This helps explain *why* the AI suggested a specific frequency, duration, intensity, or type.
""")

# Footer
st.markdown("---")
st.caption("Copyright © Rishma Fathima Basher (S2006759)")
