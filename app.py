"""
app.py  –  Sentiment Analysis AI  (Streamlit front-end)
"""

import io
import logging
import warnings

import numpy as np
import pandas as pd
import streamlit as st

from preprocess import preprocess_text, batch_preprocess
from utils import (
    load_model, model_exists, predict_sentiment,
    plot_probability_bar, plot_confusion_matrix,
    format_confidence, SENTIMENT_COLORS, LABEL_MAP,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment AI", page_icon="🧠",
    layout="wide", initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[class*="css"]{font-family:'Segoe UI',sans-serif;}
.main{background-color:#0E1117;}
.header-banner{
    background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
    border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem;
    border:1px solid #1e3a5f;}
.header-banner h1{color:#e2e8f0;font-size:2.4rem;font-weight:700;margin:0;}
.header-banner p{color:#94a3b8;font-size:1rem;margin-top:0.4rem;}
.card{background:#161b27;border:1px solid #2d3748;border-radius:12px;padding:1.5rem;margin-bottom:1rem;}
.badge{display:inline-block;padding:0.45rem 1.2rem;border-radius:999px;font-size:1.1rem;font-weight:700;margin-top:0.5rem;}
.badge-positive{background:#1a3a1a;color:#4ade80;border:1px solid #4ade80;}
.badge-neutral {background:#3a3000;color:#fbbf24;border:1px solid #fbbf24;}
.badge-negative{background:#3a0a0a;color:#f87171;border:1px solid #f87171;}
.metric-box{background:#1e2433;border-radius:10px;padding:1rem 1.2rem;text-align:center;border:1px solid #2d3748;}
.metric-box .value{font-size:2rem;font-weight:700;color:#e2e8f0;}
.metric-box .label{font-size:0.8rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;}
[data-testid="stSidebar"]{background-color:#0d111b;border-right:1px solid #1e293b;}
[data-testid="stSidebar"] *{color:#cbd5e1 !important;}
.stButton>button{
    background:linear-gradient(90deg,#3b82f6,#6366f1);color:white !important;
    border:none;border-radius:8px;padding:0.55rem 2rem;
    font-weight:600;font-size:1rem;transition:opacity 0.2s;width:100%;}
.stButton>button:hover{opacity:0.88;}
.stTextArea textarea{
    background-color:#161b27 !important;color:#e2e8f0 !important;
    border:1px solid #2d3748 !important;border-radius:8px !important;font-size:1rem !important;}
hr{border-color:#1e293b;}
.stDataFrame{border-radius:8px;overflow:hidden;}
.stAlert{border-radius:8px !important;}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
for _k, _v in {
    "model": None, "vectorizer": None, "label_encoder": None,
    "trained": False, "cm": None, "cm_classes": None,
    "accuracy": None, "report": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _store_model(model, vectorizer, le, cm=None, classes=None, acc=None, report=None):
    """Write all training artifacts into session state."""
    st.session_state.model         = model
    st.session_state.vectorizer    = vectorizer
    st.session_state.label_encoder = le
    st.session_state.trained       = True
    if cm      is not None: st.session_state.cm         = cm
    if classes is not None: st.session_state.cm_classes = classes
    if acc     is not None: st.session_state.accuracy   = acc
    if report  is not None: st.session_state.report     = report


# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_saved_model():
    """Load pickled model from disk. Returns (model, vectorizer, le)."""
    return load_model()


# ── Startup: load or train ────────────────────────────────────────────────────
if not st.session_state.trained:
    if model_exists():
        with st.spinner("🔄 Loading saved model…"):
            try:
                m, v, le = _load_saved_model()
                _store_model(m, v, le)
            except Exception as exc:
                st.error(f"Failed to load model: {exc}")
                st.stop()
    else:
        with st.spinner("🏋️ No saved model found — training on Twitter dataset…"):
            try:
                from train_model import train
                m, v, le, cm, cls = train()
                _store_model(m, v, le, cm, cls)
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
_acc_pct = (
    f"{st.session_state.accuracy * 100:.1f}%"
    if st.session_state.accuracy is not None
    else "—"
)
st.markdown(f"""
<div class="header-banner">
    <h1>🧠 Sentiment Analysis AI</h1>
    <p>Powered by TF-IDF + Ensemble (LR &amp; LinearSVC) &nbsp;|&nbsp;
       Classifies text as <b>Positive</b>, <b>Neutral</b>, or <b>Negative</b>
       &nbsp;|&nbsp; <span style="color:#a78bfa;font-weight:700;">Test Accuracy: {_acc_pct}</span></p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
from train_model import twitter_dataset_available, swiggy_dataset_available, sentiment_csv_available, train_csv_available

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    # Dataset badge
    if train_csv_available():
        st.markdown(
            '<div style="background:#1a2a3a;border:1px solid #a78bfa;border-radius:8px;'
            'padding:0.6rem 0.9rem;margin-bottom:0.8rem;">'
            '<span style="color:#a78bfa;font-weight:700;">🚀 train.csv Dataset</span><br>'
            '<span style="color:#c4b5fd;font-size:0.8rem;">27,481 real labeled tweets · Top priority</span>'
            '</div>', unsafe_allow_html=True)
    elif sentiment_csv_available():
        st.markdown(
            '<div style="background:#1a2a3a;border:1px solid #60a5fa;border-radius:8px;'
            'padding:0.6rem 0.9rem;margin-bottom:0.8rem;">'
            '<span style="color:#60a5fa;font-weight:700;">✅ Sentiment Dataset</span><br>'
            '<span style="color:#93c5fd;font-size:0.8rem;">sentiment-analysis.csv · 96 rows</span>'
            '</div>', unsafe_allow_html=True)
    elif swiggy_dataset_available():
        st.markdown(
            '<div style="background:#1a3a1a;border:1px solid #4ade80;border-radius:8px;'
            'padding:0.6rem 0.9rem;margin-bottom:0.8rem;">'
            '<span style="color:#4ade80;font-weight:700;">✅ Swiggy Dataset</span><br>'
            '<span style="color:#86efac;font-size:0.8rem;">Auto-labelled via VADER</span>'
            '</div>', unsafe_allow_html=True)
    elif twitter_dataset_available():
        st.markdown(
            '<div style="background:#1a3a1a;border:1px solid #4ade80;border-radius:8px;'
            'padding:0.6rem 0.9rem;margin-bottom:0.8rem;">'
            '<span style="color:#4ade80;font-weight:700;">✅ Twitter Dataset</span><br>'
            '<span style="color:#86efac;font-size:0.8rem;">75,682 real tweets</span>'
            '</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="background:#3a2a00;border:1px solid #fbbf24;border-radius:8px;'
            'padding:0.6rem 0.9rem;margin-bottom:0.8rem;">'
            '<span style="color:#fbbf24;font-weight:700;">⚠️ Synthetic Data</span><br>'
            '<span style="color:#fde68a;font-size:0.8rem;">Upload a real dataset for better accuracy</span>'
            '</div>', unsafe_allow_html=True)

    if st.button("🔁 Re-train Model"):
        _load_saved_model.clear()
        with st.spinner("Training…"):
            try:
                from train_model import train
                m, v, le, cm, cls = train()
                _store_model(m, v, le, cm, cls)
                # force metrics re-evaluation on next visit
                st.session_state.accuracy = None
                st.session_state.report   = None
                st.success("✅ Re-trained successfully!")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    st.markdown("---")
    st.markdown("""### ℹ️ About
**Model:** Logistic Regression  
**Features:** TF-IDF (1–2 grams, 15k vocab)  
**Preprocessing:** NLTK pipeline  
**Classes:** Positive · Neutral · Negative""")

    st.markdown("---")
    st.markdown("### 🏷️ Legend")
    for key, color in SENTIMENT_COLORS.items():
        st.markdown(
            f'<span style="color:{color};font-size:1.1rem;">⬤</span>&nbsp;&nbsp;{key.capitalize()}',
            unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_metrics = st.tabs(
    ["📝 Single Prediction", "📂 Batch CSV", "📊 Model Metrics"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Single prediction
# ═══════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.markdown("### Enter your text below")
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        user_text = st.text_area(
            "text", placeholder="Type or paste any feedback, review, or comment here…",
            height=200, label_visibility="collapsed")
        st.caption(f"Characters: {len(user_text)}  |  Words: {len(user_text.split()) if user_text.strip() else 0}")
        predict_btn = st.button("🔍 Analyse Sentiment", use_container_width=True)

    with col_out:
        if predict_btn:
            if not user_text.strip():
                st.warning("⚠️ Please enter some text first.")
            else:
                with st.spinner("Analysing…"):
                    clean  = preprocess_text(user_text)
                    result = predict_sentiment(
                        clean, st.session_state.model,
                        st.session_state.vectorizer, st.session_state.label_encoder,
                        raw_text=user_text)

                label     = result["label"]
                badge_cls = f"badge-{label}"
                st.markdown(f"""
<div class="card">
    <p style="color:#94a3b8;margin:0;font-size:0.85rem;">SENTIMENT</p>
    <div class="badge {badge_cls}">{result['display_label']}</div>
    <hr style="margin:0.8rem 0;border-color:#2d3748;">
    <p style="color:#94a3b8;margin:0;font-size:0.85rem;">CONFIDENCE</p>
    <p style="color:#e2e8f0;font-size:1.6rem;font-weight:700;margin:0.2rem 0;">
        {format_confidence(result['confidence'])}
    </p>
</div>""", unsafe_allow_html=True)

                # Low-confidence warning
                if result['confidence'] < 0.55:
                    st.warning(
                        "⚠️ **Low confidence** — the model is uncertain about this text. "
                        "Try rephrasing or adding more context.",
                        icon="🤔",
                    )

                st.markdown("**Class Probabilities**")
                st.pyplot(plot_probability_bar(result["probabilities"], label), use_container_width=True)

                with st.expander("🔬 Raw probabilities"):
                    for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
                        color = SENTIMENT_COLORS.get(cls, "#888")
                        st.markdown(
                            f'<span style="color:{color};">■</span> **{cls.capitalize()}**: {prob:.4f}',
                            unsafe_allow_html=True)
        else:
            st.markdown("""
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
    height:220px;color:#4a5568;border:2px dashed #2d3748;border-radius:12px;
    font-size:1rem;gap:0.5rem;">
    <span style="font-size:2.5rem;">💬</span>Results will appear here
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Batch CSV
# ═══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### Upload a CSV for batch predictions")
    st.info("📋 Your CSV must have a column named **`text`**.", icon="ℹ️")

    uploaded_file = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        try:
            df_up = pd.read_csv(uploaded_file)
            if "text" not in df_up.columns:
                st.error("❌ Column `text` not found. Rename your text column to `text`.")
            else:
                st.success(f"✅ {len(df_up):,} rows loaded.")
                st.dataframe(df_up.head(5), use_container_width=True)

                if st.button("⚡ Run Batch Predictions", use_container_width=True):
                    with st.spinner(f"Processing {len(df_up):,} rows…"):
                        clean_texts = batch_preprocess(df_up["text"].fillna("").tolist())
                        preds, confs, probas = [], [], []
                        raw_texts_list = df_up["text"].fillna("").tolist()
                        for ct, rt in zip(clean_texts, raw_texts_list):
                            r = predict_sentiment(ct, st.session_state.model,
                                                  st.session_state.vectorizer,
                                                  st.session_state.label_encoder,
                                                  raw_text=rt)
                            preds.append(r["label"])
                            confs.append(r["confidence"])
                            probas.append(r["probabilities"])

                        df_up["predicted_sentiment"] = preds
                        df_up["confidence"] = [f"{c*100:.1f}%" for c in confs]
                        for cls in ["positive", "neutral", "negative"]:
                            df_up[f"prob_{cls}"] = [f"{p.get(cls,0)*100:.1f}%" for p in probas]

                    counts = pd.Series(preds).value_counts()
                    c1, c2, c3 = st.columns(3)
                    for col_w, sent in zip([c1,c2,c3], ["positive","neutral","negative"]):
                        cnt = counts.get(sent, 0)
                        pct = cnt / len(preds) * 100
                        with col_w:
                            st.markdown(
                                f'<div class="metric-box">'
                                f'<div class="value" style="color:{SENTIMENT_COLORS[sent]};">{cnt}</div>'
                                f'<div class="label">{sent} ({pct:.0f}%)</div></div>',
                                unsafe_allow_html=True)

                    st.dataframe(df_up, use_container_width=True)
                    buf = io.StringIO()
                    df_up.to_csv(buf, index=False)
                    st.download_button("⬇️ Download Results CSV", buf.getvalue(),
                                       "sentiment_predictions.csv", "text/csv",
                                       use_container_width=True)
        except Exception as exc:
            st.error(f"Error: {exc}")
    else:
        sample = pd.DataFrame({"text": ["I love this!","It's okay.","Terrible waste."]})
        st.dataframe(sample, use_container_width=True)
        st.download_button("⬇️ Download Sample CSV", sample.to_csv(index=False),
                           "sample_input.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Model metrics
# ═══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.markdown("### 📈 Model Performance Metrics")

    # Compute metrics on-the-fly if not already stored
    if st.session_state.cm is None or st.session_state.accuracy is None:
        with st.spinner("Evaluating on held-out test set…"):
            try:
                from train_model import (
                    build_synthetic_dataset, twitter_dataset_available, load_twitter_dataset,
                    train_csv_available, load_train_csv,
                )
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

                if train_csv_available():
                    df_eval = load_train_csv()
                elif twitter_dataset_available():
                    df_eval = load_twitter_dataset()
                else:
                    df_eval = build_synthetic_dataset(3000)

                df_eval["clean_text"] = batch_preprocess(df_eval["text"].tolist())
                df_eval = df_eval[df_eval["clean_text"].str.strip() != ""]

                le = st.session_state.label_encoder
                known = set(le.classes_)
                df_eval = df_eval[df_eval["sentiment"].isin(known)]

                y_all = le.transform(df_eval["sentiment"])
                _, X_te, _, y_te = train_test_split(
                    df_eval["clean_text"], y_all,
                    test_size=0.2, random_state=42, stratify=y_all)

                if isinstance(st.session_state.vectorizer, dict):
                    import scipy.sparse as sp
                    X_word = st.session_state.vectorizer["word"].transform(X_te)
                    X_char = st.session_state.vectorizer["char"].transform(X_te)
                    if st.session_state.vectorizer.get("use_vader"):
                        from train_model import _compute_vader_features
                        raw_test = df_eval["text"].loc[X_te.index].tolist()
                        vader_feats = sp.csr_matrix(_compute_vader_features(raw_test))
                        X_te_vec = sp.hstack([X_word, X_char, vader_feats], format="csr")
                    else:
                        X_te_vec = sp.hstack([X_word, X_char], format="csr")

                    if "svc" in st.session_state.vectorizer:
                        lr_proba = st.session_state.model.predict_proba(X_te_vec)
                        svc_proba = st.session_state.vectorizer["svc"].predict_proba(X_te_vec)
                        y_pred = np.argmax((lr_proba + svc_proba) / 2.0, axis=1)
                    else:
                        y_pred = st.session_state.model.predict(X_te_vec)
                else:
                    X_te_vec = st.session_state.vectorizer.transform(X_te)
                    y_pred   = st.session_state.model.predict(X_te_vec)

                st.session_state.cm         = confusion_matrix(y_te, y_pred)
                st.session_state.cm_classes = le.inverse_transform(st.session_state.model.classes_).tolist()
                st.session_state.accuracy   = accuracy_score(y_te, y_pred)
                st.session_state.report     = classification_report(
                    y_te, y_pred, target_names=st.session_state.cm_classes)
            except Exception as exc:
                st.error(f"Evaluation error: {exc}")

    if st.session_state.cm is not None:
        col_cm, col_rep = st.columns([1, 1], gap="large")

        with col_cm:
            st.markdown("#### Confusion Matrix")
            st.pyplot(plot_confusion_matrix(
                st.session_state.cm, st.session_state.cm_classes),
                use_container_width=True)

        with col_rep:
            st.markdown("#### Classification Report")
            if st.session_state.accuracy is not None:
                st.markdown(
                    f'<div class="metric-box" style="margin-bottom:1rem;">'
                    f'<div class="value">{st.session_state.accuracy*100:.1f}%</div>'
                    f'<div class="label">Test Accuracy</div></div>',
                    unsafe_allow_html=True)
            if st.session_state.report:
                st.code(st.session_state.report, language="text")

        st.markdown("---")
        st.markdown("#### Per-Class Accuracy")
        cm_arr  = st.session_state.cm
        classes = st.session_state.cm_classes
        pcols   = st.columns(len(classes))
        for i, (pc, cls) in enumerate(zip(pcols, classes)):
            correct = cm_arr[i, i]
            total   = cm_arr[i].sum()
            pct     = correct / total * 100 if total else 0
            color   = SENTIMENT_COLORS.get(cls, "#888")
            pc.markdown(
                f'<div class="metric-box">'
                f'<div class="value" style="color:{color};">{pct:.1f}%</div>'
                f'<div class="label">{cls.capitalize()}</div></div>',
                unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p style="text-align:center;color:#4a5568;font-size:0.85rem;">'
            '🧠 Sentiment Analysis AI &nbsp;|&nbsp; Built with Streamlit &amp; Scikit-learn</p>',
            unsafe_allow_html=True)
