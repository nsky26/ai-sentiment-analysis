"""
preprocess.py
-------------
Text preprocessing pipeline for sentiment analysis.
Handles: contraction expansion, punctuation-as-features, repeated-char
         normalisation, lowercase, URL removal, tokenization,
         stopword removal (preserving negations), and lemmatization.
"""

import re
import nltk
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

# Download required NLTK resources (silent if already present)
def download_nltk_resources():
    """Download all required NLTK data packages."""
    resources = [
        ("tokenizers/punkt",     "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords",    "stopwords"),
        ("corpora/wordnet",      "wordnet"),
        ("corpora/omw-1.4",      "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

download_nltk_resources()

# ── Globals initialised once ──────────────────────────────────────────────────
_lemmatizer = WordNetLemmatizer()
_stop_words  = set(stopwords.words("english"))

# Extended negation / modal keep-list — these flip or weaken sentiment
_KEEP_WORDS = {
    # core negations
    "not", "no", "nor", "never", "neither", "n't",
    # contracted forms (after expansion they appear as individual words)
    "cannot", "cant", "wont", "wouldnt", "couldnt", "shouldnt",
    "doesnt", "didnt", "isnt", "wasnt", "werent", "hasnt", "hadnt",
    "dont", "aint", "neednt", "mightnt", "mustnt",
    # negative adverbs / degree words
    "barely", "hardly", "scarcely", "rarely", "seldom",
    "little", "few", "less", "least",
}
_FILTERED_STOPS = _stop_words - _KEEP_WORDS

# ── Contraction expansion map ─────────────────────────────────────────────────
_CONTRACTIONS = {
    r"won't":       "will not",
    r"can't":       "cannot",
    r"n't\b":       " not",       # catches isn't, wasn't, didn't …
    r"'re\b":       " are",
    r"'s\b":        " is",
    r"'d\b":        " would",
    r"'ll\b":       " will",
    r"'t\b":        " not",
    r"'ve\b":       " have",
    r"'m\b":        " am",
    r"i'm\b":       "i am",
    r"you're\b":    "you are",
    r"they're\b":   "they are",
    r"we're\b":     "we are",
    r"he's\b":      "he is",
    r"she's\b":     "she is",
    r"it's\b":      "it is",
    r"that's\b":    "that is",
    r"there's\b":   "there is",
    r"what's\b":    "what is",
    r"let's\b":     "let us",
    r"who's\b":     "who is",
}


def expand_contractions(text: str) -> str:
    """Expand English contractions before other cleaning steps."""
    text = text.lower()
    for pattern, replacement in _CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text)
    return text


def normalize_repeated_chars(text: str) -> str:
    """
    Collapse 3+ repeated characters to 2.
    'loooove' → 'loove', 'terribleeee' → 'terribleee' → 'terrible'
    (Two passes handles cases like 'loooooove'.)
    """
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def punctuation_to_tokens(text: str) -> str:
    """
    Convert sentiment-bearing punctuation to explicit text tokens
    BEFORE stripping all non-alpha characters.

    e.g. 'Loved it!!!' → 'Loved it exclaim exclaim exclaim'
         'What?? Really??' → 'What question question Really question question'
    """
    # Replace each ! and ? with a placeholder token
    text = re.sub(r"!", " exclaim ", text)
    text = re.sub(r"\?", " question ", text)
    return text


def remove_urls(text: str) -> str:
    """Remove http/https URLs and bare www. links."""
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_special_characters(text: str) -> str:
    """Remove non-alphabetic characters, keeping spaces."""
    return re.sub(r"[^a-zA-Z\s]", " ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces into one and strip edges."""
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline:
      1. Expand contractions          (won't → will not)
      2. Normalize repeated chars     (loooove → loove)
      3. Convert ! / ? to tokens      (! → exclaim)
      4. Lowercase
      5. Remove URLs
      6. Remove special characters / digits
      7. Normalize whitespace
      8. Tokenize
      9. Remove stopwords (preserve negations + negative adverbs)
     10. Lemmatize
     11. Re-join tokens

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Cleaned, lemmatized text ready for vectorisation.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    try:
        text = expand_contractions(text)
        text = normalize_repeated_chars(text)
        text = punctuation_to_tokens(text)
        # lowercase already done in expand_contractions; do it again to be safe
        text = text.lower()
        text = remove_urls(text)
        text = remove_special_characters(text)
        text = normalize_whitespace(text)

        tokens = word_tokenize(text)

        tokens = [
            _lemmatizer.lemmatize(token)
            for token in tokens
            if token not in _FILTERED_STOPS and len(token) > 1
        ]

        return " ".join(tokens)

    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        return text  # return original on failure


def batch_preprocess(texts) -> list:
    """
    Preprocess an iterable of texts.

    Parameters
    ----------
    texts : iterable of str

    Returns
    -------
    list of str
    """
    return [preprocess_text(t) for t in texts]
