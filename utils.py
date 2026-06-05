"""
utils.py
--------
Utility helpers: model persistence, label mapping,
confidence formatting, and chart generation.
"""

import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

LABEL_MAP = {
    "positive": "Positive 😊",
    "neutral":  "Neutral 😐",
    "negative": "Negative 😞",
}

SENTIMENT_COLORS = {
    "positive": "#4CAF50",
    "neutral":  "#FFC107",
    "negative": "#F44336",
}

# ── Model I/O ────────────────────────────────────────────────────────────────

def ensure_model_dir():
    """Create the models/ directory if it doesn't exist."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_artifact(obj, path: Path):
    """Pickle-serialize an object to *path*."""
    ensure_model_dir()
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Saved artifact → %s", path)


def load_artifact(path: Path):
    """Load a pickle artifact from *path*."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, vectorizer, label_encoder):
    """Persist the trained pipeline components."""
    save_artifact(model, MODEL_PATH)
    save_artifact(vectorizer, VECTORIZER_PATH)
    save_artifact(label_encoder, LABEL_ENCODER_PATH)
    logger.info("All model artifacts saved.")


def load_model():
    """
    Load model, vectorizer, and label encoder from disk.

    Returns
    -------
    tuple : (model, vectorizer, label_encoder)

    Raises
    ------
    FileNotFoundError if any artifact is missing.
    """
    model = load_artifact(MODEL_PATH)
    vectorizer = load_artifact(VECTORIZER_PATH)
    label_encoder = load_artifact(LABEL_ENCODER_PATH)
    return model, vectorizer, label_encoder


def model_exists() -> bool:
    """Return True if all model artifacts are present on disk."""
    return MODEL_PATH.exists() and VECTORIZER_PATH.exists() and LABEL_ENCODER_PATH.exists()


# ── Prediction helpers ────────────────────────────────────────────────────────

def predict_sentiment(text: str, model, vectorizer, label_encoder, raw_text: str = None):
    """
    Run the full inference pipeline on a single text.

    Supports two vectorizer formats:
    - Legacy : a single TfidfVectorizer instance
    - New    : a dict with keys 'word' (TfidfVectorizer), 'char' (TfidfVectorizer),
               and optionally 'svc' (CalibratedClassifierCV) for ensemble inference.

    Parameters
    ----------
    text : str  (already preprocessed)
    model, vectorizer, label_encoder : trained artifacts
    raw_text : str, optional (un-preprocessed text for VADER scoring)

    Returns
    -------
    dict with keys: label, display_label, confidence, probabilities
    """
    import scipy.sparse as sp

    # ── Build feature vector ──────────────────────────────────────────────────
    if isinstance(vectorizer, dict):
        # New combined vectorizer
        X_word = vectorizer["word"].transform([text])
        X_char = vectorizer["char"].transform([text])
        if vectorizer.get("use_vader"):
            from train_model import _compute_vader_features
            vader_feats = sp.csr_matrix(_compute_vader_features([raw_text if raw_text is not None else text]))
            X = sp.hstack([X_word, X_char, vader_feats], format="csr")
        else:
            X = sp.hstack([X_word, X_char], format="csr")
    else:
        # Legacy single TF-IDF vectorizer
        X = vectorizer.transform([text])

    # ── Predict ───────────────────────────────────────────────────────────────
    if isinstance(vectorizer, dict) and "svc" in vectorizer:
        # Ensemble: soft-vote LR + SVC
        lr_proba  = model.predict_proba(X)[0]
        svc_proba = vectorizer["svc"].predict_proba(X)[0]
        proba     = (lr_proba + svc_proba) / 2.0
        pred_idx  = int(np.argmax(proba))
    else:
        # Single model
        pred_idx = int(model.predict(X)[0])
        proba    = model.predict_proba(X)[0]

    label   = label_encoder.inverse_transform([pred_idx])[0]
    classes = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))

    return {
        "label":         label,
        "display_label": LABEL_MAP.get(label, label.capitalize()),
        "confidence":    float(np.max(proba)),
        "probabilities": dict(zip(classes, proba.tolist())),
    }


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_probability_bar(probabilities: dict, predicted_label: str):
    """
    Horizontal bar chart of class probabilities.

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = list(probabilities.keys())
    values = [probabilities[l] for l in labels]
    colors = [SENTIMENT_COLORS.get(l, "#888888") for l in labels]
    display_labels = [LABEL_MAP.get(l, l.capitalize()) for l in labels]

    fig, ax = plt.subplots(figsize=(6, 2.8))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    bars = ax.barh(display_labels, values, color=colors, edgecolor="none", height=0.5)

    # Percentage labels
    for bar, val in zip(bars, values):
        ax.text(
            min(val + 0.02, 0.95),
            bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Probability", color="#AAAAAA", fontsize=10)
    ax.tick_params(colors="#CCCCCC", labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.set_visible(False)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: list):
    """
    Render a styled confusion matrix heatmap.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=30, ha="right", color="white", fontsize=11)
    ax.set_yticklabels(class_names, color="white", fontsize=11)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] < thresh else "black",
                fontsize=13, fontweight="bold",
            )

    ax.set_ylabel("True Label", color="white", fontsize=12)
    ax.set_xlabel("Predicted Label", color="white", fontsize=12)
    ax.set_title("Confusion Matrix", color="white", fontsize=14, pad=12)
    plt.tight_layout()
    return fig


def format_confidence(score: float) -> str:
    """Return a human-readable confidence string."""
    return f"{score * 100:.1f}%"
