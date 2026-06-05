# Sentiment Analysis AI

Automatically classifies any text feedback as **Positive**, **Neutral**, or **Negative** using a TF-IDF + ensemble (Logistic Regression + LinearSVC) pipeline with NLTK preprocessing and VADER sentiment features.

---

## 📁 Project Structure

```
sentimental/
├── app.py              # Streamlit UI (main entry point)
├── train_model.py      # Dataset loading, training pipeline, model persistence
├── preprocess.py       # NLTK text preprocessing pipeline
├── utils.py            # Model I/O, prediction, chart helpers
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
├── Procfile            # Generic web deployment
├── setup.sh            # Streamlit Cloud startup script
├── .streamlit/
│   └── config.toml     # Dark theme + server settings
├── models/             # Auto-created after first training run
│   ├── sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
├── train.csv           # Optional: real labeled tweets dataset (27k+ rows)
└── sentiment-analysis.csv  # Optional: smaller labeled dataset (96+ rows)
```

---

## 🧠 Model Details

| Component | Details |
|-----------|---------|
| Algorithm | Ensemble — Logistic Regression + Calibrated LinearSVC (soft voting) |
| Features | Combined TF-IDF: word n-grams (1–2, 20k vocab) + character n-grams (3–5, 30k vocab) + VADER polarity scores |
| Preprocessing | NLTK pipeline: contraction expansion, repeated-char normalisation, punctuation-to-tokens, stopword removal (preserving negations), lemmatisation |
| Class weight | `balanced` — handles class imbalance automatically |
| Train/Test | 80% / 20% stratified split |
| Hyperparameter tuning | GridSearchCV over C values |

---

## 📦 Datasets

The training pipeline auto-detects the best available dataset in priority order:

1. **`train.csv`** (top priority) — 27,000+ real labeled tweets with Positive / Neutral / Negative classes
2. **`sentiment-analysis.csv`** — smaller labeled dataset (augmented to 3,000 balanced samples)
3. **`swiggydataset.csv`** — auto-labelled via VADER sentiment analysis
4. **`twitter_training.csv`** + **`twitter_validation.csv`** — Twitter sentiment dataset
5. **Synthetic fallback** — generated from curated templates if no external dataset is present

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/sentimental.git
cd sentimental

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Train manually
python train_model.py

# 5. Launch the Streamlit app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🖥️ App Features

| Feature | Description |
|---|---|
| 📝 Single prediction | Multiline text area → predict button → colour-coded result |
| 🎨 Colour-coded output | 🟢 Green = Positive · ⚫ Yellow = Neutral · 🔴 Red = Negative |
| 📊 Confidence score | Horizontal bar chart showing probability for each class |
| 📂 Batch CSV upload | Upload CSV with `text` column → predict all rows → download results |
| 🗃️ Train on your data | Upload any labelled CSV, map columns and labels, retrain the model |
| 📈 Model metrics | Confusion matrix + classification report + per-class accuracy |
| 🔁 Re-train button | Sidebar button to retrain without restarting the app |

---

## ☁️ Deployment

### Streamlit Cloud (free, easiest)
1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set main file to `app.py` → Deploy

### Render (free tier)
1. Push repo to GitHub
2. Sign up at [render.com](https://render.com)
3. New → Web Service → connect repo → `render.yaml` auto-configures everything

### Hugging Face Spaces (free, always-on)
1. Create account at [huggingface.co](https://huggingface.co)
2. New Space → Streamlit template
3. Push files to the Space repo

---

## 📋 Batch CSV Format

Your CSV must have a `text` column:

```csv
text
"I love this product!"
"It's okay, nothing special."
"Very disappointed, waste of money."
```

Output adds: `predicted_sentiment`, `confidence`, `prob_positive`, `prob_neutral`, `prob_negative`

---

## 🛠️ Development

### Project Dependencies

```
streamlit>=1.32.0
scikit-learn>=1.4.0
nltk>=3.8.1
pandas>=2.2.0
numpy>=1.26.0
matplotlib>=3.8.0
```

### Running Tests

```bash
# If tests are added in the future
pytest tests/
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 🙌 Acknowledgements

- [Streamlit](https://streamlit.io/) for the web framework
- [scikit-learn](https://scikit-learn.org/) for ML utilities
- [NLTK](https://www.nltk.org/) for NLP preprocessing
- [VADER](https://github.com/cjhutto/vaderSentiment) for sentiment lexicon features
