"""
feature_extraction.py
---------------------
Feature engineering module for the Phishing Email Detection System.

Combines:
    - TF-IDF text vectors
    - Handcrafted text features (length, caps, keyword presence)
    - URL-based features (count, IP, shorteners, HTTPS, etc.)
"""

import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import preprocess_text


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

URGENT_KEYWORDS = [
    "verify", "login", "update", "password", "account", "bank",
    "click", "urgent", "limited", "winner", "suspend", "alert",
    "immediately", "confirm", "expire", "credit", "free", "prize",
    "congratulations", "selected", "claim", "refund", "security",
    "locked", "blocked", "restricted", "compromised",
]

URL_SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "short.io", "rb.gy"]

SUSPICIOUS_DOMAINS = [
    "secure-login", "account-verify", "update-info", "bank-secure",
    "paypal-secure", "amazon-verify", "apple-id-verify",
]

URL_PATTERN = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(\\),]|"
    r"(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.[^\s]+"
)

IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


# ──────────────────────────────────────────────
# URL Feature Helpers
# ──────────────────────────────────────────────

def extract_urls(text: str) -> list:
    """Extract all URLs from email text."""
    return URL_PATTERN.findall(text) if isinstance(text, str) else []


def count_urls(text: str) -> int:
    return len(extract_urls(text))


def has_ip_address(text: str) -> int:
    """1 if any URL contains an IP address, else 0."""
    if not isinstance(text, str):
        return 0
    urls = extract_urls(text)
    for url in urls:
        if IP_PATTERN.search(url):
            return 1
    # Also check raw text for IP-like patterns in links
    return 1 if IP_PATTERN.search(text) else 0


def avg_url_length(text: str) -> float:
    """Average length of all URLs in the email."""
    urls = extract_urls(text)
    if not urls:
        return 0.0
    return float(np.mean([len(u) for u in urls]))


def has_https(text: str) -> int:
    """1 if any HTTPS URL is present."""
    if not isinstance(text, str):
        return 0
    urls = extract_urls(text)
    return int(any(u.lower().startswith("https") for u in urls))


def has_url_shortener(text: str) -> int:
    """1 if any known URL shortening service is used."""
    if not isinstance(text, str):
        return 0
    lower = text.lower()
    return int(any(s in lower for s in URL_SHORTENERS))


def count_dots_in_urls(text: str) -> int:
    """Total number of dots across all URLs (indicator of subdomain abuse)."""
    urls = extract_urls(text)
    return sum(u.count(".") for u in urls)


def has_suspicious_domain(text: str) -> int:
    """1 if any suspicious domain pattern is found in URLs."""
    if not isinstance(text, str):
        return 0
    lower = text.lower()
    return int(any(d in lower for d in SUSPICIOUS_DOMAINS))


# ──────────────────────────────────────────────
# Text Feature Helpers
# ──────────────────────────────────────────────

def email_length(text: str) -> int:
    return len(text) if isinstance(text, str) else 0


def word_count(text: str) -> int:
    return len(text.split()) if isinstance(text, str) else 0


def capital_letter_count(text: str) -> int:
    return sum(1 for c in text if c.isupper()) if isinstance(text, str) else 0


def urgent_keyword_count(text: str) -> int:
    """Count how many urgent/phishing keywords appear in the email."""
    if not isinstance(text, str):
        return 0
    lower = text.lower()
    return sum(1 for kw in URGENT_KEYWORDS if kw in lower)


# ──────────────────────────────────────────────
# Feature Extraction Class
# ──────────────────────────────────────────────

class FeatureExtractor:
    """
    Extracts combined TF-IDF + handcrafted features from email text.

    Usage:
        fe = FeatureExtractor()
        X_train = fe.fit_transform(train_texts)
        X_test  = fe.transform(test_texts)
    """

    def __init__(self, max_tfidf_features: int = 5000):
        self.tfidf = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=(1, 2),      # unigrams + bigrams
            sublinear_tf=True,       # log normalization
            min_df=2,
        )
        self.feature_names_ = None

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def fit_transform(self, texts: pd.Series):
        """
        Fit TF-IDF on training texts and return combined feature matrix.

        Args:
            texts (pd.Series | list): Raw email texts.

        Returns:
            scipy.sparse matrix: Combined feature matrix.
        """
        cleaned = texts.apply(preprocess_text)
        tfidf_matrix = self.tfidf.fit_transform(cleaned)
        handcrafted = self._extract_handcrafted(texts)
        self.feature_names_ = self._build_feature_names()
        return hstack([tfidf_matrix, csr_matrix(handcrafted)])

    def transform(self, texts: pd.Series):
        """
        Transform new texts using already-fitted TF-IDF.

        Args:
            texts (pd.Series | list): Raw email texts.

        Returns:
            scipy.sparse matrix: Combined feature matrix.
        """
        if isinstance(texts, str):
            texts = pd.Series([texts])
        cleaned = texts.apply(preprocess_text)
        tfidf_matrix = self.tfidf.transform(cleaned)
        handcrafted = self._extract_handcrafted(texts)
        return hstack([tfidf_matrix, csr_matrix(handcrafted)])

    def get_tfidf_feature_names(self) -> list:
        return self.tfidf.get_feature_names_out().tolist()

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    def _extract_handcrafted(self, texts) -> np.ndarray:
        """
        Build the (n_samples, 11) handcrafted feature matrix.

        Features (in order):
            0: email_length
            1: word_count
            2: capital_letter_count
            3: urgent_keyword_count
            4: num_urls
            5: has_ip_address
            6: avg_url_length
            7: has_https
            8: has_url_shortener
            9: count_dots_in_urls
           10: has_suspicious_domain
        """
        records = []
        for text in texts:
            records.append([
                email_length(text),
                word_count(text),
                capital_letter_count(text),
                urgent_keyword_count(text),
                count_urls(text),
                has_ip_address(text),
                avg_url_length(text),
                has_https(text),
                has_url_shortener(text),
                count_dots_in_urls(text),
                has_suspicious_domain(text),
            ])
        return np.array(records, dtype=np.float32)

    def _build_feature_names(self) -> list:
        tfidf_names = self.get_tfidf_feature_names()
        handcrafted_names = [
            "email_length", "word_count", "capital_letter_count",
            "urgent_keyword_count", "num_urls", "has_ip_address",
            "avg_url_length", "has_https", "has_url_shortener",
            "count_dots_in_urls", "has_suspicious_domain",
        ]
        return tfidf_names + handcrafted_names


# ──────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────

def extract_handcrafted_features(text: str) -> dict:
    """
    Return a human-readable dict of handcrafted features for a single email.
    Useful for debugging and the web app's feature display.
    """
    return {
        "Email Length": email_length(text),
        "Word Count": word_count(text),
        "Capital Letters": capital_letter_count(text),
        "Urgent Keywords Found": urgent_keyword_count(text),
        "Number of URLs": count_urls(text),
        "IP Address in URL": bool(has_ip_address(text)),
        "Avg URL Length": round(avg_url_length(text), 1),
        "HTTPS Present": bool(has_https(text)),
        "URL Shortener Detected": bool(has_url_shortener(text)),
        "Dots in URLs": count_dots_in_urls(text),
        "Suspicious Domain": bool(has_suspicious_domain(text)),
    }
