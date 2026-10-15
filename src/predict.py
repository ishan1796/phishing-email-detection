"""
predict.py
----------
Prediction module for the Phishing Email Detection System.

Provides:
    - predict_email(text) → (label, confidence)
    - PhishingPredictor class for advanced usage
    - CLI interface for real-time scanning
"""

import os
import sys
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.feature_extraction import extract_handcrafted_features

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "phishing_model.pkl")

# ──────────────────────────────────────────────
# Predictor Class
# ──────────────────────────────────────────────

class PhishingPredictor:
    """
    Loads the saved model bundle and provides prediction capabilities.

    Attributes:
        model_name (str): Name of the loaded best model.
        model: Trained classifier.
        scaler: Fitted MaxAbsScaler.
        feature_extractor: Fitted FeatureExtractor.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at: {model_path}\n"
                "Please run: python src/train_model.py"
            )
        bundle = joblib.load(model_path)
        self.model_name = bundle["model_name"]
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_extractor = bundle["feature_extractor"]

    def predict(self, email_text: str) -> dict:
        """
        Predict whether an email is phishing or safe.

        Args:
            email_text (str): Raw email text to classify.

        Returns:
            dict with keys:
                - label (str): "Phishing" or "Safe"
                - prediction (int): 1 = Phishing, 0 = Safe
                - confidence (float): 0.0 – 1.0
                - confidence_pct (str): e.g. "97.4%"
                - model_used (str): Model name
                - features (dict): Handcrafted feature values
        """
        if not isinstance(email_text, str) or not email_text.strip():
            return {
                "label": "Unknown",
                "prediction": -1,
                "confidence": 0.0,
                "confidence_pct": "0.0%",
                "model_used": self.model_name,
                "features": {},
            }

        import pandas as pd
        text_series = pd.Series([email_text])
        X = self.feature_extractor.transform(text_series)
        X = self.scaler.transform(X)

        prediction = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0]
        confidence = float(proba[prediction])

        label = "Phishing" if prediction == 1 else "Safe"
        features = extract_handcrafted_features(email_text)

        return {
            "label": label,
            "prediction": prediction,
            "confidence": confidence,
            "confidence_pct": f"{confidence * 100:.1f}%",
            "model_used": self.model_name,
            "features": features,
        }

    def predict_batch(self, emails: list) -> list:
        """
        Predict multiple emails at once.

        Args:
            emails (list[str]): List of raw email texts.

        Returns:
            list[dict]: Prediction results for each email.
        """
        return [self.predict(e) for e in emails]


# ──────────────────────────────────────────────
# Module-level convenience function
# ──────────────────────────────────────────────

_predictor = None


def _get_predictor() -> PhishingPredictor:
    """Lazy-load singleton predictor."""
    global _predictor
    if _predictor is None:
        _predictor = PhishingPredictor()
    return _predictor


def predict_email(email_text: str) -> dict:
    """
    Convenience function: predict a single email.

    Args:
        email_text (str): Raw email text.

    Returns:
        dict: Prediction result (see PhishingPredictor.predict).

    Example:
        >>> result = predict_email("Your account has been suspended. Verify now!")
        >>> print(result['label'], result['confidence_pct'])
        Phishing 97.4%
    """
    return _get_predictor().predict(email_text)


# ──────────────────────────────────────────────
# CLI Interface
# ──────────────────────────────────────────────

DEMO_EMAILS = [
    {
        "title": "Phishing Example",
        "text": (
            "Your bank account has been suspended. "
            "Click here to verify immediately: http://192.168.1.45/login?token=abc123xyz "
            "Failure to act within 24 hours will result in permanent account closure. URGENT!"
        ),
    },
    {
        "title": "Safe Example",
        "text": (
            "Hi Sarah, just a friendly reminder about our team meeting tomorrow at 10 AM. "
            "Please review the attached agenda and come prepared with your project updates. "
            "Let me know if you have any questions. Looking forward to seeing everyone!"
        ),
    },
]


def run_demo():
    """Run predictions on built-in demo examples."""
    predictor = _get_predictor()
    print("\n" + "=" * 65)
    print("  PHISHING EMAIL DETECTOR - DEMO PREDICTIONS")
    print(f"  Model: {predictor.model_name}")
    print("=" * 65)

    for demo in DEMO_EMAILS:
        result = predictor.predict(demo["text"])
        icon = "[PHISHING]" if result["prediction"] == 1 else "[SAFE]"
        print(f"\n  [{demo['title']}]")
        print(f"  Email: {demo['text'][:80]}...")
        print(f"  {icon} Prediction : {result['label']}")
        print(f"     Confidence : {result['confidence_pct']}")
        print(f"     Model Used : {result['model_used']}")
        print(f"     Key Features:")
        for k, v in result["features"].items():
            print(f"       - {k}: {v}")


def run_interactive():
    """Interactive CLI scanner — user enters email text."""
    predictor = _get_predictor()
    print("\n" + "=" * 65)
    print("  PHISHING EMAIL REAL-TIME SCANNER")
    print(f"  Model: {predictor.model_name}")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 65)

    while True:
        print("\nPaste email text below (press Enter twice when done):")
        lines = []
        try:
            while True:
                line = input()
                if line.lower() in ("quit", "exit"):
                    print("\nGoodbye! Stay safe from phishing!")
                    return
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            return

        email_text = "\n".join(lines).strip()
        if not email_text:
            print("[!] Empty input. Please try again.")
            continue

        result = predictor.predict(email_text)
        icon = "[PHISHING]" if result["prediction"] == 1 else "[SAFE]"
        print(f"\n  {icon}  Result    : {result['label']}")
        print(f"     Confidence: {result['confidence_pct']}")
        print(f"     Model    : {result['model_used']}")


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phishing Email Predictor")
    parser.add_argument(
        "--mode",
        choices=["demo", "interactive"],
        default="demo",
        help="Run mode: 'demo' (show examples) or 'interactive' (CLI scanner)",
    )
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()
    else:
        run_interactive()
