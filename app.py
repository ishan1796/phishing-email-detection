"""
app.py
------
Flask web application for the Phishing Email Detection System.
Provides a premium dark-themed UI for real-time email scanning.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.predict import PhishingPredictor

# ──────────────────────────────────────────────
# App Initialization
# ──────────────────────────────────────────────

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)

# Lazy-load predictor on first request
_predictor = None


def get_predictor() -> PhishingPredictor:
    global _predictor
    if _predictor is None:
        model_path = os.path.join(PROJECT_ROOT, "models", "phishing_model.pkl")
        _predictor = PhishingPredictor(model_path=model_path)
    return _predictor


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main scanner page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST endpoint for email prediction.

    Request JSON:
        { "email_text": "..." }

    Response JSON:
        {
            "label": "Phishing" | "Safe",
            "prediction": 1 | 0,
            "confidence": 0.974,
            "confidence_pct": "97.4%",
            "model_used": "Random Forest",
            "features": { ... }
        }
    """
    try:
        data = request.get_json(force=True)
        if not data or "email_text" not in data:
            return jsonify({"error": "Missing 'email_text' field."}), 400

        email_text = data["email_text"].strip()
        if len(email_text) < 5:
            return jsonify({"error": "Email text too short."}), 400

        predictor = get_predictor()
        result = predictor.predict(email_text)
        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({
            "error": "Model not found. Please run 'python src/train_model.py' first.",
            "detail": str(e),
        }), 503

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "Phishing Email Detection System"})


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  [*] Phishing Email Detection System - Web App")
    print("  Visit: http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=5000)
