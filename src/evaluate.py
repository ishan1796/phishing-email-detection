"""
evaluate.py
-----------
Evaluation and visualization module for the Phishing Email Detection System.

Generates:
    - Classification reports for all models
    - Confusion matrix heatmap (best model)
    - Model comparison bar chart
    - Feature importance plot (Random Forest)
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Color Palette
# ──────────────────────────────────────────────
COLORS = {
    "phishing": "#FF4C4C",
    "safe": "#4CAF50",
    "primary": "#6C63FF",
    "secondary": "#FF6584",
    "accent": "#43E8D8",
    "bg": "#1A1A2E",
    "surface": "#16213E",
    "text": "#E0E0E0",
}

PALETTE = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], "#F7C948"]

plt.rcParams.update({
    "figure.facecolor": COLORS["bg"],
    "axes.facecolor": COLORS["surface"],
    "axes.labelcolor": COLORS["text"],
    "text.color": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "grid.color": "#2A2A4A",
    "font.family": "DejaVu Sans",
})


# ──────────────────────────────────────────────
# Evaluator Class
# ──────────────────────────────────────────────

class ModelEvaluator:
    """
    Evaluates trained models and generates visualizations.

    Args:
        results (dict): Training results from PhishingModelTrainer.
        y_test (array-like): Ground-truth test labels.
    """

    def __init__(self, results: dict, y_test):
        self.results = results
        self.y_test = y_test
        self.class_names = ["Safe", "Phishing"]

    # ──────────────────────────────────────────────
    # ----------------------------------------------
    # Public API
    # ----------------------------------------------

    def run_all(self):
        """Run all evaluations and save plots."""
        print("\n" + "=" * 60)
        print("\n  EVALUATION REPORTS")
        print("=" * 60)
        self.print_classification_reports()
        self.plot_confusion_matrices()
        self.plot_model_comparison()
        self.plot_feature_importance()
        print(f"\n[DONE] All plots saved to: {OUTPUTS_DIR}")

    def print_classification_reports(self):
        """Print classification report for every model."""
        for name, res in self.results.items():
            print(f"\n{'─' * 40}")
            print(f"  {name}")
            print(f"{'─' * 40}")
            print(classification_report(
                self.y_test,
                res["y_pred"],
                target_names=self.class_names,
            ))

    def plot_confusion_matrices(self):
        """Plot confusion matrix heatmap for each model."""
        n = len(self.results)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        fig.patch.set_facecolor(COLORS["bg"])

        if n == 1:
            axes = [axes]

        for ax, (name, res) in zip(axes, self.results.items()):
            cm = confusion_matrix(self.y_test, res["y_pred"])
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="RdPu",
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=ax,
                linewidths=0.5,
                annot_kws={"size": 14, "weight": "bold"},
            )
            ax.set_title(
                f"{name}\nAcc: {res['accuracy']:.4f}",
                color=COLORS["text"],
                fontsize=13,
                pad=12,
            )
            ax.set_xlabel("Predicted", fontsize=11)
            ax.set_ylabel("Actual", fontsize=11)
            ax.tick_params(colors=COLORS["text"])

        plt.suptitle(
            "Confusion Matrices - All Models",
            color=COLORS["accent"],
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        save_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        plt.close()
        print(f"\n  [OK] Confusion matrix saved: {save_path}")

    def plot_model_comparison(self):
        """Bar chart comparing all models across all 4 metrics."""
        model_names = list(self.results.keys())
        metrics = {
            "Accuracy": [r["accuracy"] for r in self.results.values()],
            "Precision": [r["precision"] for r in self.results.values()],
            "Recall": [r["recall"] for r in self.results.values()],
            "F1 Score": [r["f1_score"] for r in self.results.values()],
        }

        x = np.arange(len(model_names))
        bar_width = 0.2
        n_metrics = len(metrics)
        offsets = np.linspace(-(n_metrics - 1) * bar_width / 2,
                              (n_metrics - 1) * bar_width / 2, n_metrics)

        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor(COLORS["bg"])
        ax.set_facecolor(COLORS["surface"])

        for i, (metric_name, values) in enumerate(metrics.items()):
            bars = ax.bar(
                x + offsets[i], values,
                width=bar_width,
                label=metric_name,
                color=PALETTE[i],
                alpha=0.88,
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=COLORS["text"],
                    fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=11, color=COLORS["text"])
        ax.set_ylabel("Score", fontsize=12, color=COLORS["text"])
        ax.set_ylim(0.5, 1.05)
        ax.set_title(
            "Model Comparison - Accuracy, Precision, Recall, F1",
            fontsize=15,
            fontweight="bold",
            color=COLORS["accent"],
            pad=18,
        )
        ax.legend(fontsize=11, facecolor=COLORS["surface"], labelcolor=COLORS["text"])
        ax.grid(axis="y", alpha=0.3, color=COLORS["text"])
        ax.spines[:].set_color("#2A2A4A")

        plt.tight_layout()
        save_path = os.path.join(OUTPUTS_DIR, "model_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        plt.close()
        print(f"  [OK] Model comparison chart saved: {save_path}")

    def plot_feature_importance(self):
        """
        Plot top-20 feature importances from Random Forest.
        Falls back to coefficient magnitudes from Logistic Regression.
        """
        # Try Random Forest first
        rf_result = self.results.get("Random Forest")
        lr_result = self.results.get("Logistic Regression")

        importances = None
        feature_names = None
        source_name = ""

        if rf_result is not None:
            model = rf_result["model"]
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                source_name = "Random Forest"

        if importances is None and lr_result is not None:
            model = lr_result["model"]
            if hasattr(model, "coef_"):
                importances = np.abs(model.coef_[0])
                source_name = "Logistic Regression"

        if importances is None:
            print("  [WARN] Feature importance plot skipped (no compatible model).")
            return

        # Attempt to get feature names from trainer's extractor
        try:
            import joblib
            model_dir = os.path.join(PROJECT_ROOT, "models")
            bundle = joblib.load(os.path.join(model_dir, "phishing_model.pkl"))
            fe = bundle["feature_extractor"]
            if hasattr(fe, "feature_names_") and fe.feature_names_:
                feature_names = fe.feature_names_
        except Exception:
            pass

        if feature_names is None or len(feature_names) != len(importances):
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        # Top 20
        top_n = 20
        indices = np.argsort(importances)[-top_n:][::-1]
        top_importances = importances[indices]
        top_names = [feature_names[i] for i in indices]

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(COLORS["bg"])
        ax.set_facecolor(COLORS["surface"])

        colors = plt.cm.plasma(np.linspace(0.3, 0.9, top_n))
        bars = ax.barh(
            range(top_n), top_importances[::-1],
            color=colors[::-1], alpha=0.9, edgecolor="white", linewidth=0.4
        )
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1], fontsize=9, color=COLORS["text"])
        ax.set_xlabel("Importance Score", fontsize=12, color=COLORS["text"])
        ax.set_title(
            f"Top {top_n} Phishing Indicators ({source_name})",
            fontsize=14,
            fontweight="bold",
            color=COLORS["accent"],
            pad=14,
        )
        ax.grid(axis="x", alpha=0.3, color=COLORS["text"])
        ax.spines[:].set_color("#2A2A4A")

        plt.tight_layout()
        save_path = os.path.join(OUTPUTS_DIR, "feature_importance.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        plt.close()
        print(f"  [OK] Feature importance chart saved: {save_path}")


# ──────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import joblib
    results_path = os.path.join(PROJECT_ROOT, "models", "training_results.pkl")
    if not os.path.exists(results_path):
        print("No training results found. Run src/train_model.py first.")
        sys.exit(1)

    results = joblib.load(results_path)
    # Load y_test from saved metadata (placeholder - regenerate if needed)
    print("Results loaded. Re-run train_model.py for full evaluation.")
