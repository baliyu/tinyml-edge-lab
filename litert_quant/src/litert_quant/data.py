"""Digits data and one fixed split.

AI-assisted: drafted with Claude, checked by the author.
Pixels are scaled to 0..1 and shaped (N, 8, 8, 1). The split is stratified
and seeded, so every script and the board test see the same test set.
"""
import hashlib

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 0
TEST_FRACTION = 0.2


def load_split(seed=SEED, test_fraction=TEST_FRACTION):
    """Return x_train, x_test, y_train, y_test (scikit-learn order)."""
    d = load_digits()
    x = (d.images / 16.0).astype(np.float32)[..., np.newaxis]
    y = d.target.astype(np.int64)
    return train_test_split(
        x, y, test_size=test_fraction, random_state=seed, stratify=y
    )


def fingerprint(x, y):
    """SHA-256 of the arrays, to prove two runs used the same samples."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()
