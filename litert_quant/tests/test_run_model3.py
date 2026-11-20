"""Tests for the model 3 PC run, on synthetic .mat files (not real data).

AI-assisted: drafted with Claude, checked by the author.
"""
import json

import numpy as np
from scipy.io import savemat

from litert_quant.cwru_data import Recording
from litert_quant.run_model3 import agreement, auc_breakdown, main, operating_point

RECS = [
    Recording(1, "normal", 0, 0), Recording(2, "normal", 0, 1), Recording(3, "normal", 0, 2),
    Recording(4, "normal", 0, 3), Recording(5, "IR", 7, 0), Recording(6, "B", 14, 2),
    Recording(7, "OR", 21, 3),
]


def _write(root):
    rng = np.random.default_rng(0)
    for r in RECS:
        amp = 0.06 if r.kind == "normal" else 0.5
        x = (rng.normal(size=4 * 2048) * amp).reshape(-1, 1)
        savemat(str(root / f"{r.num}.mat"), {f"X{r.num:03d}_DE_time": x})


def test_auc_breakdown_groups():
    y = np.array([0, 0, 1, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.9, 0.8, 0.15, 0.7])
    kind = np.array(["normal", "normal", "IR", "IR", "B", "B"])
    size = np.array([0, 0, 7, 7, 14, 14])
    load = np.array([3, 3, 0, 0, 2, 2])
    out = auc_breakdown(y, score, kind, size, load)
    assert out["by_kind"]["IR"] == 1.0
    assert 0.0 < out["by_kind"]["B"] < 1.0
    assert set(out["by_size_mil"]) == {"7", "14"} and set(out["by_load_hp"]) == {"0", "2"}
    assert 0.5 < out["overall"] <= 1.0


def test_operating_point_uses_training_percentile():
    train_scores = np.arange(100, dtype=float)
    y = np.array([0, 0, 1, 1])
    score = np.array([10.0, 99.5, 98.5, 50.0])
    op = operating_point(train_scores, y, score, percentile=99.0)
    assert abs(op["threshold"] - 98.01) < 1e-9
    assert op["fpr"] == 0.5 and op["tpr"] == 0.5


def test_agreement_fields():
    f = np.array([1.0, 2.0, 3.0, 4.0])
    q = np.array([1.1, 1.9, 3.2, 3.9])
    a = agreement(f, q, f > 2.5, q > 2.5)
    assert a["spearman"] == 1.0 and a["decision_agreement"] == 1.0 and a["decision_disagree"] == 0


def test_main_runs_and_writes_everything(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    _write(root)
    out = tmp_path / "out"
    r = main(root=root, recordings=RECS, epochs=3, out_root=out)
    assert (out / "models" / "cwru_ae_float.tflite").is_file()
    assert (out / "models" / "cwru_ae_int8.tflite").is_file()
    assert (out / "results" / "model3_pc_scores.npz").is_file()
    assert r["n_train"] == 48 and r["n_test_normal"] == 16 and r["n_test_fault"] == 48
    assert r["parameters"] == 4520
    assert set(r["train_recordings"]).isdisjoint(r["test_recordings"])
    for name in ("float", "int8"):
        assert r["scores"][name]["auc"]["overall"] > 0.95
        op = r["scores"][name]["operating_point"]
        assert 0.0 <= op["fpr"] <= 1.0 and 0.0 <= op["tpr"] <= 1.0
    assert set(r["scores"]["float"]["auc"]["by_kind"]) == {"IR", "B", "OR"}
    assert r["baselines"]["rms"]["overall"] > 0.95
    saved = json.loads((out / "results" / "model3_pc.json").read_text())
    assert saved["test_fingerprint"] == r["test_fingerprint"]
    d = np.load(out / "results" / "model3_pc_scores.npz")
    assert d["float"].shape == (64,) and d["int8"].shape == (64,) and d["train_float"].shape == (48,)
    assert len(r["file_sha256"]) == 7
