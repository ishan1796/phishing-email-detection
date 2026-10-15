# 🛡️ Phishing Email Detection System

An end-to-end Machine Learning project that classifies emails as **Phishing** or **Safe** using text content and URL-based features.

---

## 📁 Project Structure

```
Phishing Email Detection/
├── dataset/
│   └── phishing_emails.csv       # Synthetic training dataset
├── models/
│   └── phishing_model.pkl        # Saved best model bundle
├── src/
│   ├── preprocessing.py          # Text cleaning & NLP preprocessing
│   ├── feature_extraction.py     # TF-IDF + URL/text feature engineering
│   ├── train_model.py            # Training pipeline (4 models)
│   ├── evaluate.py               # Metrics, confusion matrix, charts
│   └── predict.py                # predict_email() + CLI scanner
├── notebooks/
│   └── generate_dataset.py       # Synthetic dataset generator
├── outputs/
│   ├── confusion_matrix.png      # Confusion heatmaps for all models
│   ├── model_comparison.png      # Accuracy/F1 comparison chart
│   └── feature_importance.png    # Top phishing indicators
├── templates/
│   └── index.html                # Flask web UI (dark theme)
├── app.py                        # Flask web application
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone / navigate to the project folder

```bash
cd "Phishing Email Detection"
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Step 1 — Generate Dataset

```bash
python notebooks/generate_dataset.py
```

Generates **2,000 realistic phishing/safe email samples** → `dataset/phishing_emails.csv`

---

### Step 2 — Train Models

```bash
python src/train_model.py
```

- Trains 4 models: Logistic Regression, Naive Bayes, Random Forest, SVM
- Saves best model to `models/phishing_model.pkl`
- Generates evaluation plots in `outputs/`

**Expected output:**
```
[1/5] Loading dataset...
[2/5] Extracting features...
[3/5] Training models...
  ▸ Logistic Regression... Done | Acc: 0.9400 | F1: 0.9400
  ▸ Naive Bayes...         Done | Acc: 0.8900 | F1: 0.8900
  ▸ Random Forest...       Done | Acc: 0.9750 | F1: 0.9750
  ▸ Support Vector Machine Done | Acc: 0.9650 | F1: 0.9650
[4/5] Best model: Random Forest
[5/5] Saved model bundle → models/phishing_model.pkl
```

---

### Step 3 — Run Predictions

#### Demo mode (see example predictions):
```bash
python src/predict.py --mode demo
```

#### Interactive CLI scanner:
```bash
python src/predict.py --mode interactive
```

**Example output:**
```
🚨 Prediction : Phishing
   Confidence : 97.4%
   Model Used : Random Forest
```

---

### Step 4 — Launch Web App (Flask)

```bash
python app.py
```

Visit: **http://127.0.0.1:5000**

Features:
- Dark glassmorphism UI
- Paste any email text → instant prediction
- Confidence bar animation
- Feature breakdown panel (URL analysis, keyword detection, etc.)

---

## 🧠 Machine Learning Pipeline

### Feature Engineering

| Category | Features |
|---|---|
| **Text** | TF-IDF (5,000 unigrams+bigrams), email length, word count, capital letters, urgent keyword count |
| **URL** | URL count, IP address detection, avg URL length, HTTPS presence, URL shorteners, suspicious domains, dot count |

### Models Compared

| Model | Expected Accuracy |
|---|---|
| Random Forest | ~95–98% |
| Support Vector Machine | ~94–97% |
| Logistic Regression | ~91–95% |
| Naive Bayes | ~87–92% |

---

## 📊 Evaluation Outputs

After training, check `outputs/`:

- **`confusion_matrix.png`** — Heatmaps for all 4 models
- **`model_comparison.png`** — Bar chart comparing Accuracy, Precision, Recall, F1
- **`feature_importance.png`** — Top 20 phishing indicator features

---

## 🔍 API Usage

### Python API

```python
from src.predict import predict_email

result = predict_email("Your account has been suspended. Verify now at http://bit.ly/abc")
print(result['label'])           # → "Phishing"
print(result['confidence_pct'])  # → "97.4%"
print(result['features'])        # → dict of handcrafted features
```

### Flask REST API

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"email_text": "Your account is suspended. Click here immediately!"}'
```

**Response:**
```json
{
  "label": "Phishing",
  "prediction": 1,
  "confidence": 0.974,
  "confidence_pct": "97.4%",
  "model_used": "Random Forest",
  "features": { ... }
}
```

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| ML Models | scikit-learn |
| Data Processing | pandas, numpy |
| NLP | nltk (tokenization, lemmatization, stopwords) |
| Visualization | matplotlib, seaborn |
| Model Persistence | joblib |
| Web App | Flask |
| Feature Engineering | scipy (sparse matrices), regex |

---

## 📝 License

MIT License — Free to use and modify.
