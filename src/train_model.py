"""
train_model.py
--------------
Training pipeline for the Phishing Email Detection System.

Trains 4 classifiers:
    - Logistic Regression
    - Multinomial Naive Bayes
    - Random Forest
    - Support Vector Machine (LinearSVC)

Saves the best model to models/phishing_model.pkl
"""

import os
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.feature_extraction import FeatureExtractor

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "phishing_emails.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_model.pkl")
EXTRACTOR_PATH = os.path.join(MODEL_DIR, "feature_extractor.pkl")
RESULTS_PATH = os.path.join(MODEL_DIR, "training_results.pkl")


# ──────────────────────────────────────────────
# Model Definitions
# ──────────────────────────────────────────────

def get_models() -> dict:
    """Return dictionary of model name → estimator."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        ),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        "Support Vector Machine": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, random_state=42),
            cv=3,
        ),
    }


# ──────────────────────────────────────────────
# Training Class
# ──────────────────────────────────────────────

class PhishingModelTrainer:
    """
    Orchestrates the full training pipeline:
        1. Load dataset
        2. Extract features
        3. Train/test split
        4. Train all models
        5. Evaluate and compare
        6. Save best model
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.feature_extractor = FeatureExtractor(max_tfidf_features=5000)
        self.models = get_models()
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.X_test = None
        self.y_test = None

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def run(self):
        """Execute full training pipeline."""
        print("=" * 60)
        print("  PHISHING EMAIL DETECTION - TRAINING PIPELINE")
        print("=" * 60)

        df = self._load_data()
        X_train, X_test, y_train, y_test = self._prepare_features(df)
        self.X_test = X_test
        self.y_test = y_test
        self._train_all(X_train, X_test, y_train, y_test)
        self._select_best()
        self._save_artifacts()
        self._print_summary()
        return self.results

    # ──────────────────────────────────────────────
    # Private methods
    # ──────────────────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        print(f"\n[1/5] Loading dataset from: {DATASET_PATH}")
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(
                f"Dataset not found at {DATASET_PATH}.\n"
                "Run: python notebooks/generate_dataset.py"
            )
        df = pd.read_csv(DATASET_PATH)

        # Validate columns
        required = {"email_text", "label"}
        if not required.issubset(df.columns):
            raise ValueError(f"Dataset must contain columns: {required}")

        # Handle missing values
        df.dropna(subset=["email_text", "label"], inplace=True)
        df["email_text"] = df["email_text"].astype(str).str.strip()
        df = df[df["email_text"].str.len() > 0]

        print(f"    Samples: {len(df):,}")
        print(f"    Phishing: {df['label'].sum():,} | Safe: {(df['label'] == 0).sum():,}")
        return df

    def _prepare_features(self, df: pd.DataFrame):
        print("\n[2/5] Extracting features...")
        X = self.feature_extractor.fit_transform(df["email_text"])
        y = df["label"].values

        # Scale sparse features (for NB compatibility — needs non-negative)
        # We use MaxAbsScaler which preserves sparsity
        self.scaler = MaxAbsScaler()
        X = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        print(f"    Feature matrix: {X.shape}")
        print(f"    Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
        return X_train, X_test, y_train, y_test

    def _train_all(self, X_train, X_test, y_train, y_test):
        print("\n[3/5] Training models...\n")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        for name, model in self.models.items():
            print(f"  >> Training {name}...", end=" ", flush=True)
            t0 = time.time()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            elapsed = time.time() - t0

            # Compute metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            self.results[name] = {
                "model": model,
                "y_pred": y_pred,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "train_time": elapsed,
            }
            print(f"Done ({elapsed:.2f}s) | Acc: {acc:.4f} | F1: {f1:.4f}")

    def _select_best(self):
        """Pick model with highest F1 score."""
        best_name = max(self.results, key=lambda k: self.results[k]["f1_score"])
        self.best_model_name = best_name
        self.best_model = self.results[best_name]["model"]
        print(f"\n[4/5] Best model: {best_name} "
              f"(F1={self.results[best_name]['f1_score']:.4f})")

    def _save_artifacts(self):
        """Save best model, feature extractor, scaler, and results."""
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Save model bundle (model + scaler + extractor together)
        bundle = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "scaler": self.scaler,
            "feature_extractor": self.feature_extractor,
        }
        joblib.dump(bundle, MODEL_PATH)
        joblib.dump(self.results, RESULTS_PATH)

        print(f"[5/5] Saved model bundle -> {MODEL_PATH}")

    def _print_summary(self):
        print("\n" + "=" * 60)
        print("  MODEL COMPARISON SUMMARY")
        print("=" * 60)
        header = f"{'Model':<30} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}"
        print(header)
        print("-" * 60)
        for name, res in sorted(
            self.results.items(),
            key=lambda x: x[1]["f1_score"],
            reverse=True,
        ):
            print(
                f"{name:<30} "
                f"{res['accuracy']:>7.4f} "
                f"{res['precision']:>7.4f} "
                f"{res['recall']:>7.4f} "
                f"{res['f1_score']:>7.4f}"
            )
        print("=" * 60)


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    trainer = PhishingModelTrainer(test_size=0.2, random_state=42)
    results = trainer.run()

    # Trigger evaluation & visualization
    from src.evaluate import ModelEvaluator
    evaluator = ModelEvaluator(results, trainer.y_test)
    evaluator.run_all()
