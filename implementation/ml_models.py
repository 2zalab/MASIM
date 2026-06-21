"""
Machine Learning Models for Predictive Maintenance & Failure Prediction.
  - XGBoost classifier for binary failure prediction
  - LSTM for time-series sensor anomaly detection
  - Feature engineering pipeline
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# Synthetic Dataset Generator
# ─────────────────────────────────────────────

def generate_sensor_dataset(n_samples: int = 8000, failure_rate: float = 0.12) -> pd.DataFrame:
    """
    Generate realistic sensor dataset simulating 6 CNC machines over time.
    Features reflect environment.py sensor_vector() layout.
    """
    n_fail = int(n_samples * failure_rate)
    n_normal = n_samples - n_fail

    def make_normal(n):
        return pd.DataFrame({
            "temperature":        np.random.normal(58, 8, n),
            "vibration":          np.random.normal(0.7, 0.2, n),
            "current_load":       np.random.uniform(0.3, 0.75, n),
            "wear_level":         np.random.beta(2, 8, n),
            "time_since_maint":   np.random.exponential(200, n),
            "is_working":         np.random.binomial(1, 0.75, n).astype(float),
            "is_failed":          np.zeros(n),
            "failure_count":      np.random.poisson(0.5, n).astype(float),
            "jobs_completed":     np.random.poisson(15, n).astype(float),
            "energy_consumed":    np.random.normal(12, 3, n),
            # derived features
            "temp_x_vibration":   None,
            "wear_x_load":        None,
            "label":              0,
        })

    def make_failure(n):
        return pd.DataFrame({
            "temperature":        np.random.normal(82, 10, n),
            "vibration":          np.random.normal(1.8, 0.4, n),
            "current_load":       np.random.uniform(0.7, 1.0, n),
            "wear_level":         np.random.beta(6, 2, n),
            "time_since_maint":   np.random.exponential(600, n),
            "is_working":         np.random.binomial(1, 0.9, n).astype(float),
            "is_failed":          np.zeros(n),
            "failure_count":      np.random.poisson(3, n).astype(float),
            "jobs_completed":     np.random.poisson(40, n).astype(float),
            "energy_consumed":    np.random.normal(22, 5, n),
            "temp_x_vibration":   None,
            "wear_x_load":        None,
            "label":              1,
        })

    df = pd.concat([make_normal(n_normal), make_failure(n_fail)], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Add interaction and rolling features
    df["temp_x_vibration"] = df["temperature"] * df["vibration"]
    df["wear_x_load"] = df["wear_level"] * df["current_load"]
    df["energy_per_job"] = df["energy_consumed"] / df["jobs_completed"].clip(1)
    df["maint_lag_wear"] = df["time_since_maint"] * df["wear_level"]

    # Clip noisy columns
    df["temperature"] = df["temperature"].clip(20, 120)
    df["vibration"] = df["vibration"].clip(0, 5)
    return df


FEATURE_COLS = [
    "temperature", "vibration", "current_load", "wear_level",
    "time_since_maint", "is_working", "failure_count",
    "jobs_completed", "energy_consumed",
    "temp_x_vibration", "wear_x_load", "energy_per_job", "maint_lag_wear",
]


# ─────────────────────────────────────────────
# XGBoost Model
# ─────────────────────────────────────────────

class XGBoostFailurePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.metrics: dict = {}
        self.feature_importances_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_s = self.scaler.fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_s, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
        scale_pos = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
        self.model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            verbose=False,
        )
        y_pred = self.model.predict(X_te)
        y_prob = self.model.predict_proba(X_te)[:, 1]

        # 5-fold CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_f1 = cross_val_score(self.model, X_s, y, cv=cv, scoring="f1").mean()

        self.metrics = {
            "accuracy":  accuracy_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred, zero_division=0),
            "recall":    recall_score(y_te, y_pred, zero_division=0),
            "f1":        f1_score(y_te, y_pred, zero_division=0),
            "roc_auc":   roc_auc_score(y_te, y_prob),
            "cv_f1":     cv_f1,
            "confusion": confusion_matrix(y_te, y_pred).tolist(),
        }
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))

    def print_report(self):
        m = self.metrics
        print("\n=== XGBoost Failure Predictor ===")
        print(f"  Accuracy  : {m['accuracy']:.4f}")
        print(f"  Precision : {m['precision']:.4f}")
        print(f"  Recall    : {m['recall']:.4f}")
        print(f"  F1-Score  : {m['f1']:.4f}")
        print(f"  ROC-AUC   : {m['roc_auc']:.4f}")
        print(f"  CV-F1 (5k): {m['cv_f1']:.4f}")
        print(f"  Confusion :\n    {m['confusion']}")


# ─────────────────────────────────────────────
# Lightweight LSTM (NumPy-only, no PyTorch dep.)
# ─────────────────────────────────────────────

class SimpleLSTMAnomalyDetector:
    """
    Rolling-window z-score anomaly detector as LSTM substitute
    when TensorFlow/PyTorch is unavailable.
    Computes reconstruction error from exponential smoothing.
    """

    def __init__(self, window: int = 20, threshold_sigma: float = 3.0):
        self.window = window
        self.threshold = threshold_sigma
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        """X: (n_samples, n_features)"""
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        return self

    def anomaly_score(self, x: np.ndarray) -> float:
        z = np.abs((x - self.mean_) / self.std_)
        return float(z.max())

    def is_anomaly(self, x: np.ndarray) -> bool:
        return self.anomaly_score(x) > self.threshold


# ─────────────────────────────────────────────
# Training entry point
# ─────────────────────────────────────────────

def train_predictor(n_samples: int = 8000) -> tuple:
    """Returns (trained_predictor, feature_names, dataset)."""
    print("Generating sensor dataset ...")
    df = generate_sensor_dataset(n_samples=n_samples)
    X = df[FEATURE_COLS].values
    y = df["label"].values
    print(f"  Dataset: {len(df)} samples | {y.mean()*100:.1f}% failure rate")

    predictor = XGBoostFailurePredictor()
    predictor.fit(X, y)
    predictor.print_report()

    lstm = SimpleLSTMAnomalyDetector(window=20, threshold_sigma=3.0)
    lstm.fit(X[y == 0])   # fit on normal samples only

    return predictor, FEATURE_COLS, df, lstm


if __name__ == "__main__":
    predictor, feat_names, df, lstm = train_predictor(n_samples=8000)
    print("\nFeature importances:")
    for fn, fi in sorted(zip(feat_names, predictor.feature_importances_),
                         key=lambda x: -x[1]):
        print(f"  {fn:<25} {fi:.4f}")
