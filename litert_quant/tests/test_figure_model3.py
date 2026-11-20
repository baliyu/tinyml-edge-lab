"""Tests for the model 3 figure, on synthetic scores and results (not real data).

AI-assisted: drafted with Claude, checked by the author.
"""
import json

import numpy as np

from litert_quant.figure_model3 import make_model3_figure


def _inputs(tmp_path):
    rng = np.random.default_rng(0)
    y = np.array([0] * 20 + [1] * 80)
    f = np.where(y == 0, 0.5, 20.0) * rng.lognormal(0, 0.3, size=100)
    q = f * rng.lognormal(0, 0.05, size=100)
    rms = np.where(y == 0, 0.06, 0.5) * rng.lognormal(0, 0.2, size=100)
    train = 0.5 * rng.lognormal(0, 0.3, size=50)
    np.savez(tmp_path / "s.npz", y_test=y, float=f, int8=q, rms=rms, train_float=train)
    auc = {
        "overall": 1.0,
        "by_kind": {"B": 1.0, "IR": 1.0, "OR": 0.9999},
        "by_size_mil": {"7": 1.0, "14": 1.0, "21": 1.0},
        "by_load_hp": {"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0},
    }
    r = {
        "parameters": 4520, "seed": 0, "epochs": 60, "n_test_normal": 20,
        "scores": {
            "float": {"auc": auc, "operating_point": {"threshold": 1.0}},
            "int8": {"auc": auc, "operating_point": {"threshold": 1.0}},
        },
        "baselines": {"rms": auc},
        "float_vs_int8": {"spearman": 0.99, "decision_disagree": 2},
    }
    (tmp_path / "r.json").write_text(json.dumps(r))
    return tmp_path / "s.npz", tmp_path / "r.json"


def test_figure_is_written_as_a_png(tmp_path):
    npz, js = _inputs(tmp_path)
    out = make_model3_figure(npz, js, tmp_path / "figs" / "m3.png")
    data = out.read_bytes()
    assert data[:4] == b"\x89PNG" and len(data) > 10000


def test_figure_handles_missing_groups(tmp_path):
    npz, js = _inputs(tmp_path)
    r = json.loads(js.read_text())
    r["scores"]["float"]["auc"]["by_kind"] = {}
    js.write_text(json.dumps(r))
    out = make_model3_figure(npz, js, tmp_path / "m3b.png")
    assert out.is_file()
