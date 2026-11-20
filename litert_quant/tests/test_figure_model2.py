"""Tests for the model 2 figure, on small synthetic results.

AI-assisted: drafted with Claude, checked by the author.
"""
import json

import numpy as np

from litert_quant.figure_model2 import disagreement_counts, make_model2_figure
from litert_quant.har_data import LABELS


def test_disagreement_counts_known_case():
    y = np.array([0, 1, 2, 3])
    f = np.array([0, 1, 0, 0])
    q = np.array([0, 0, 2, 1])
    assert disagreement_counts(y, f, q) == {
        "float_right": 1,
        "int8_right": 1,
        "both_wrong": 1,
        "total": 3,
    }


def test_disagreement_counts_add_up():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, 500)
    f = rng.integers(0, 6, 500)
    q = rng.integers(0, 6, 500)
    c = disagreement_counts(y, f, q)
    assert c["float_right"] + c["int8_right"] + c["both_wrong"] == c["total"]
    assert c["total"] == int(np.sum(f != q))


def test_figure_is_written_as_png(tmp_path):
    rng = np.random.default_rng(1)
    y = rng.integers(0, 6, 300)

    def noisy():
        p = y.copy()
        flip = rng.random(300) < 0.2
        p[flip] = rng.integers(0, 6, int(flip.sum()))
        return p

    np.savez(tmp_path / "p.npz", y_test=y, keras=noisy(), float=noisy(), int8=noisy())
    recall = {name: 0.8 for name in LABELS}
    result = {"parameters": 3334, "seed": 0, "epochs": 30,
              "per_class_recall_float": recall, "per_class_recall_int8": recall}
    (tmp_path / "r.json").write_text(json.dumps(result))
    base = {"test_accuracy": 0.8, "n_test": 300, "per_class_recall": recall}
    (tmp_path / "b.json").write_text(json.dumps(base))
    out = make_model2_figure(tmp_path / "p.npz", tmp_path / "r.json", tmp_path / "b.json",
                             tmp_path / "figs" / "m2.png")
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 5000
