"""
utils.py — shared helpers for all pages
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(ROOT_DIR, "dataset", "diabetes.csv")
MODEL_DIR  = os.path.join(ROOT_DIR, "model")

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]

# ── Dataset loader ─────────────────────────────────────────────────────────────
@staticmethod
def _make_sample_data():
    """Fallback: generate a small representative sample if CSV not found."""
    np.random.seed(42)
    n = 768
    data = {
        "Pregnancies":             np.random.randint(0, 17, n),
        "Glucose":                 np.random.normal(121, 32, n).clip(0, 200).astype(int),
        "BloodPressure":           np.random.normal(69, 19, n).clip(0, 122).astype(int),
        "SkinThickness":           np.random.normal(20, 16, n).clip(0, 99).astype(int),
        "Insulin":                 np.random.normal(80, 115, n).clip(0, 846).astype(int),
        "BMI":                     np.random.normal(32, 8, n).clip(0, 67).round(1),
        "DiabetesPedigreeFunction": np.random.exponential(0.47, n).clip(0.08, 2.42).round(3),
        "Age":                     np.random.randint(21, 82, n),
        "Outcome":                 np.random.binomial(1, 0.349, n),
    }
    return pd.DataFrame(data)

def load_dataset():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        return _make_sample_data()

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    df_clean = df.copy()
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df_clean[col] = df_clean[col].replace(0, np.nan)
        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    return df_clean

# ── Model save / load ──────────────────────────────────────────────────────────
def _pkl_path(name: str) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    return os.path.join(MODEL_DIR, name)

def save_model(obj, name: str):
    with open(_pkl_path(name), "wb") as f:
        pickle.dump(obj, f)

def load_model(name: str):
    p = _pkl_path(name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

# ── Train all models (cached on disk) ─────────────────────────────────────────
def get_models():
    """Return (scaler, rf_model, kmeans_model, iso_model, X_scaled, df_clean, best_k)."""
    clf   = load_model("model_classification.pkl")
    km    = load_model("model_clustering.pkl")
    iso   = load_model("model_anomaly.pkl")
    sc    = load_model("scaler.pkl")

    df    = load_dataset()
    df_c  = preprocess(df)

    if sc is None:
        sc = MinMaxScaler()
        X_all = sc.fit_transform(df_c[FEATURE_NAMES + ["Outcome"]])
    else:
        X_all = sc.transform(df_c[FEATURE_NAMES + ["Outcome"]])

    X = X_all[:, :8]
    Y = X_all[:, 8]

    if clf is None:
        from sklearn.model_selection import train_test_split
        X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2,
                                                    random_state=42, stratify=df_c["Outcome"])
        clf = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42)
        clf.fit(X_tr, Y_tr)
        save_model(clf, "model_classification.pkl")
        save_model(sc,  "scaler.pkl")

    best_k = 3
    if km is None:
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        km.fit(X)
        save_model(km, "model_clustering.pkl")

    if iso is None:
        iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        iso.fit(X)
        save_model(iso, "model_anomaly.pkl")

    return sc, clf, km, iso, X, df_c, best_k
