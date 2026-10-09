import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from tinyml_lab.selection import assert_disjoint, group_cv_macro_f1, group_folds, one_se_choice


def test_one_se_prefers_the_smaller_model_within_one_standard_error():
    # best is index 2 (0.810, SE 0.01) so anything >= 0.800 qualifies; the least complex of those is index 1
    assert one_se_choice([0.70, 0.805, 0.810, 0.806], [0.01] * 4, [2, 3, 4, 5]) == 1


def test_one_se_returns_the_best_when_the_standard_error_is_tiny():
    assert one_se_choice([0.70, 0.805, 0.810, 0.806], [1e-6] * 4, [2, 3, 4, 5]) == 2


def test_one_se_uses_complexity_not_list_order():
    # candidate 0 is the most complex; candidate 2 is the simplest one inside the band
    assert one_se_choice([0.81, 0.80, 0.805], [0.02, 0.02, 0.02], [100, 50, 10]) == 2


def test_assert_disjoint_raises_when_a_volunteer_is_on_both_sides():
    groups = np.array([1, 1, 2, 2])
    assert_disjoint(groups, np.array([0, 1]), np.array([2, 3]))
    with pytest.raises(AssertionError):
        assert_disjoint(groups, np.array([0, 2]), np.array([1, 3]))


def test_group_folds_needs_two_groups():
    with pytest.raises(ValueError):
        group_folds(np.array([1, 1, 1]))


def test_grouped_cv_works_even_when_rows_are_shuffled():
    """Plain K-fold on shuffled rows would split volunteers across folds; grouped folds must not."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 12)
    y = rng.integers(0, 3, size=len(groups))
    X = rng.normal(size=(len(groups), 4)) + y[:, None]
    order = rng.permutation(len(groups))
    scores = group_cv_macro_f1(DecisionTreeClassifier(max_depth=3, random_state=0), X[order], y[order], groups[order], n_splits=5)
    assert len(scores) == 5
    assert ((scores >= 0) & (scores <= 1)).all()
