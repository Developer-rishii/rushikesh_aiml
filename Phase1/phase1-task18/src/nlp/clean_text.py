"""
nlp/clean_text.py — Step 1: clean and tokenise the text data. Direct
guard against the pitfall "Skipping text cleaning": lowercasing,
punctuation stripping, stopword removal, and a minimum token length are
all applied and independently verifiable (tested), not just assumed to
have run because a vectorizer was called.
"""
import re
import string
import logging
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

log = logging.getLogger("src.nlp.clean_text")

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str, cfg) -> str:
    if not isinstance(text, str):
        raise ValueError(f"Expected a string, got {type(text)}: {text!r}")
    t = text
    if cfg.lowercase:
        t = t.lower()
    if cfg.strip_punctuation:
        t = t.translate(_PUNCT_TABLE)
    t = re.sub(r"\s+", " ", t).strip()

    tokens = t.split(" ")
    if cfg.remove_stopwords:
        tokens = [tok for tok in tokens if tok not in ENGLISH_STOP_WORDS]
    tokens = [tok for tok in tokens if len(tok) >= cfg.min_token_length]
    return " ".join(tokens)


def clean_corpus(df: pd.DataFrame, cfg) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Cannot clean an empty corpus.")
    out = df.copy()
    out["clean_text"] = out[cfg.text_col].apply(lambda t: clean_text(t, cfg))
    empty_after_cleaning = (out["clean_text"].str.len() == 0).sum()
    if empty_after_cleaning > 0:
        log.warning("[Step 1] %s document(s) became EMPTY after cleaning (all-stopword docs) — "
                     "these are unusable and flagged, not silently kept.", empty_after_cleaning)
    log.info("[Step 1] Cleaned %s documents. Example: %r -> %r",
              len(out), out[cfg.text_col].iloc[0][:60], out["clean_text"].iloc[0][:60])
    return out, int(empty_after_cleaning)
