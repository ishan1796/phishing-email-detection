"""
preprocessing.py
----------------
Text preprocessing module for the Phishing Email Detection System.
Handles cleaning, tokenization, stopword removal, and lemmatization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def download_nltk_resources():
    """Download required NLTK resources if not already present."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


# Download on module import
download_nltk_resources()

# ──────────────────────────────────────────────
# Module-level objects (created once, reused)
# ──────────────────────────────────────────────
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

# Preserve phishing-related keywords even if they appear in stopwords
PHISHING_KEYWORDS = {
    "verify", "login", "update", "password", "account", "bank",
    "click", "urgent", "limited", "winner", "suspend", "secure",
    "confirm", "alert", "immediately", "expired", "credit", "free",
}
_stop_words -= PHISHING_KEYWORDS  # keep phishing signals


class TextPreprocessor:
    """
    Cleans and normalizes email text for ML feature extraction.

    Steps:
        1. Lowercase
        2. Remove URLs (stored separately for URL features)
        3. Remove email addresses
        4. Remove special characters & punctuation
        5. Remove digits
        6. Tokenize
        7. Remove stopwords
        8. Lemmatize
    """

    def __init__(self):
        self.lemmatizer = _lemmatizer
        self.stop_words = _stop_words

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """
        Full preprocessing pipeline.

        Args:
            text (str): Raw email text.

        Returns:
            str: Cleaned, lemmatized text.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self._to_lowercase(text)
        text = self._remove_urls(text)
        text = self._remove_emails(text)
        text = self._remove_special_chars(text)
        text = self._remove_extra_spaces(text)

        tokens = self._tokenize(text)
        tokens = self._remove_stopwords(tokens)
        tokens = self._lemmatize(tokens)

        return " ".join(tokens)

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _to_lowercase(text: str) -> str:
        return text.lower()

    @staticmethod
    def _remove_urls(text: str) -> str:
        """Replace URLs with empty string (URLs extracted separately)."""
        url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
            r"|www\.[^\s]+"
        )
        return url_pattern.sub("", text)

    @staticmethod
    def _remove_emails(text: str) -> str:
        return re.sub(r"\S+@\S+", "", text)

    @staticmethod
    def _remove_special_chars(text: str) -> str:
        """Remove punctuation and digits, keep spaces."""
        text = re.sub(r"[^\w\s]", " ", text)   # punctuation → space
        text = re.sub(r"\d+", " ", text)         # digits → space
        return text

    @staticmethod
    def _remove_extra_spaces(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _tokenize(text: str) -> list:
        return word_tokenize(text)

    def _remove_stopwords(self, tokens: list) -> list:
        return [t for t in tokens if t not in self.stop_words and len(t) > 2]

    def _lemmatize(self, tokens: list) -> list:
        return [self.lemmatizer.lemmatize(t) for t in tokens]


# ──────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────

_preprocessor = TextPreprocessor()


def preprocess_text(text: str) -> str:
    """
    Module-level convenience function.

    Args:
        text (str): Raw email text.

    Returns:
        str: Cleaned text ready for vectorization.
    """
    return _preprocessor.clean(text)
