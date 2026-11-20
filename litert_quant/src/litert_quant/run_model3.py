"""Run model 3 on the PC: train on normal windows, convert, score the test windows.

AI-assisted: drafted with Claude, checked by the author.
Writes the two .tflite files, results/model3_pc.json and results/model3_pc_scores.npz.
Scores are per-window reconstruction errors in training-std units. AUC treats faults as
positive. The operating point uses the 99th percentile of that model's own training
scores. RMS and kurtosis of the raw window are scored as simple baselines.
"""
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.stats import kurtosis, spearmanr
from sklearn.metrics import roc_auc_score

from litert_quant.cwru_data import DEFAULT_ROOT, FS, WINDOW, catalogue, file_hashes, fingerprint, make_split
from litert_quant.model3 import (
    fit_scaler,
    input_quantization,
    mse_scores,
    predict_scores,
    preprocess,
    representative_set,
    to_float_tflite,
    to_int8_tflite,
    train,
)

ROOT = Path(__file__).resolve().parents[2]
SEED = 0
EPOCHS = 60
REP_SAMPLES = 300
THRESHOLD_PERCENTILE = 99.0


def _group_auc(y, score, fault_mask):
    sel = (y == 0) | fault_mask
    return float(roc_auc_score(y[sel], score[sel]))


def auc_breakdown(y, score, kind, size, load):
    """AUC overall and for each fault kind, size and load against all normal test windows."""
    y, score = np.asarray(y), np.asarray(score)
    kind, size, load = np.asarray(kind), np.asarray(size), np.asarray(load)
    fault = y == 1
    out = {"overall": float(roc_auc_score(y, score)), "by_kind": {}, "by_size_mil": {}, "by_load_hp": {}}
    for k in sorted(set(kind[fault].tolist())):
        out["by_kind"][str(k)] = _group_auc(y, score, fault & (kind == k))
    for v in sorted(set(size[fault].tolist())):
        out["by_size_mil"][str(int(v))] = _group_auc(y, score, fault & (size == v))
    for v in sorted(set(load[fault].tolist())):
        out["by_load_hp"][str(int(v))] = _group_auc(y, score, fault & (load == v))
    return out


def operating_point(train_scores, y, score, percentile=THRESHOLD_PERCENTILE):
    """Threshold at a percentile of the training scores, with the rates it gives on the test set."""
    y, score = np.asarray(y), np.asarray(score)
    thr = float(np.percentile(train_scores, percentile))
    flagged = score > thr
    return {
        "percentile": percentile,
        "threshold": thr,
        "tpr": float(flagged[y == 1].mean()),
        "fpr": float(flagged[y == 0].mean()),
    }


def agreement(f_scores, q_scores, f_flag, q_flag):
    f, q = np.asarray(f_scores), np.asarray(q_scores)
    return {
        "spearman": float(spearmanr(f, q).statistic),
        "median_relative_difference": float(np.median(np.abs(q - f) / np.maximum(f, 1e-12))),
        "decision_agreement": float(np.mean(np.asarray(f_flag) == np.asarray(q_flag))),
        "decision_disagree": int(np.sum(np.asarray(f_flag) != np.asarray(q_flag))),
    }


def main(root=DEFAULT_ROOT, recordings=None, epochs=EPOCHS, out_root=ROOT):
    recs = catalogue() if recordings is None else list(recordings)
    s = make_split(root, recs, WINDOW)
    std, clip = fit_scaler(s.x_train)
    xtr, xte = preprocess(s.x_train, std, clip), preprocess(s.x_test, std, clip)
    sizes = {r.num: r.size_mil for r in recs}
    size_te = np.array([sizes[int(n)] for n in s.rec_test])

    model = train(xtr, epochs=epochs, seed=SEED)
    keras_tr = mse_scores(xtr, model.predict(xtr, verbose=0))
    keras_te = mse_scores(xte, model.predict(xte, verbose=0))
    float_blob = to_float_tflite(model)
    int8_blob = to_int8_tflite(model, representative_set(xtr, REP_SAMPLES, SEED))
    f_tr, f_te = predict_scores(float_blob, xtr), predict_scores(float_blob, xte)
    q_tr, q_te = predict_scores(int8_blob, xtr), predict_scores(int8_blob, xte)

    rms = np.sqrt(np.mean(s.x_test.astype(np.float64) ** 2, axis=1))
    kurt = kurtosis(s.x_test.astype(np.float64), axis=1)

    out_root = Path(out_root)
    models_dir, results_dir = out_root / "models", out_root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "cwru_ae_float.tflite").write_bytes(float_blob)
    (models_dir / "cwru_ae_int8.tflite").write_bytes(int8_blob)
    np.savez(
        results_dir / "model3_pc_scores.npz",
        y_test=s.y_test, rec_test=s.rec_test, load_test=s.load_test, kind_test=s.kind_test,
        size_test=size_te, keras=keras_te, float=f_te, int8=q_te, rms=rms, kurtosis=kurt,
        train_keras=keras_tr, train_float=f_tr, train_int8=q_tr,
    )

    ops = {
        "keras": operating_point(keras_tr, s.y_test, keras_te),
        "float": operating_point(f_tr, s.y_test, f_te),
        "int8": operating_point(q_tr, s.y_test, q_te),
    }
    args = (s.y_test, None, s.kind_test, size_te, s.load_test)

    def breakdown(score):
        return auc_breakdown(args[0], score, *args[2:])

    scale, zero = input_quantization(int8_blob)
    result = {
        "model": "cwru_autoencoder",
        "parameters": int(model.count_params()),
        "seed": SEED,
        "epochs": epochs,
        "window": WINDOW,
        "sampling_rate_hz": FS,
        "representative_samples": min(REP_SAMPLES, int(len(xtr))),
        "scaler_std": std,
        "scaler_clip": clip,
        "n_train": int(len(xtr)),
        "n_test_normal": int(np.sum(s.y_test == 0)),
        "n_test_fault": int(np.sum(s.y_test == 1)),
        "train_recordings": sorted(set(s.rec_train.tolist())),
        "test_recordings": sorted(set(s.rec_test.tolist())),
        "test_fingerprint": fingerprint(s.x_test, s.y_test),
        "file_sha256": {str(k): v for k, v in file_hashes(root, recs).items()},
        "scores": {
            "keras": {"auc": breakdown(keras_te), "operating_point": ops["keras"]},
            "float": {"auc": breakdown(f_te), "operating_point": ops["float"]},
            "int8": {"auc": breakdown(q_te), "operating_point": ops["int8"]},
        },
        "float_vs_int8": agreement(f_te, q_te, f_te > ops["float"]["threshold"], q_te > ops["int8"]["threshold"]),
        "baselines": {"rms": breakdown(rms), "kurtosis": breakdown(kurt)},
        "float_bytes": len(float_blob),
        "int8_bytes": len(int8_blob),
        "input_scale": scale,
        "input_zero_point": zero,
        "versions": {
            "tensorflow": tf.__version__,
            "ai_edge_litert": version("ai-edge-litert"),
            "numpy": np.__version__,
        },
    }
    (results_dir / "model3_pc.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
