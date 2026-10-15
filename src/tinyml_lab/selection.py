"""Model selection that never touches the test volunteers.

Folds are split by group (volunteer), so a person never appears on both sides of a split, and the chosen
model is the least complex one whose score is within one standard error of the best.
"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


def group_folds(groups, n_splits=5):
    """A GroupKFold with at most n_splits folds (fewer if there are fewer groups). Raises ValueError for fewer than two distinct groups,
    where grouped cross-validation is impossible.
    """
    groups = np.asarray(groups)
    k = min(n_splits, len(np.unique(groups)))
    if k < 2:
        raise ValueError("need at least two distinct groups for grouped cross-validation")
    return GroupKFold(n_splits=k)


def assert_disjoint(groups, train_idx, val_idx):
    """Fail loudly if one group (volunteer) is on both sides of a split."""
    groups = np.asarray(groups)
    if set(groups[train_idx]) & set(groups[val_idx]):
        raise AssertionError("a volunteer appears in both the training and the validation fold")


def group_cv_macro_f1(estimator, X, y, groups, n_splits=5):
    """Macro-F1 on each held-out fold, with folds split by volunteer. Returns one score per fold."""
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    scores = []
    for tr, va in group_folds(groups, n_splits).split(X, y, groups):
        assert_disjoint(groups, tr, va)
        model = clone(estimator).fit(X[tr], y[tr])
        scores.append(f1_score(y[va], model.predict(X[va]), average="macro"))
    return np.asarray(scores)


def one_se_choice(means, ses, complexities):
    """Index of the least complex candidate whose mean score is within one standard error of the best mean."""
    means, ses, complexities = (np.asarray(a, dtype=float) for a in (means, ses, complexities))
    best = int(np.argmax(means))
    ok = np.flatnonzero(means >= means[best] - ses[best])
    return int(ok[np.argmin(complexities[ok])])
