"""data/split.py — one stratified train/test split; test touched only in Step 5."""
import logging
from sklearn.model_selection import train_test_split

log = logging.getLogger("src.data.split")


def split_train_test(X, y, cfg):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_frac, stratify=y, random_state=cfg.seed,
    )
    log.info("Split: train=%s test=%s (test held out until Step 5)", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test
