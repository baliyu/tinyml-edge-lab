"""Run model 2 on the PC: train on the HAR train split, convert, score on the test split.

AI-assisted: drafted with Claude, checked by the author.
Writes the two .tflite files, results/model2_pc.json and the predictions.
The JSON stores correct counts, so accuracy intervals can be computed from
them later. The decision tree numbers are copied from results/har_baseline.json.
"""
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, recall_score

from litert_quant.har_data import DEFAULT_ROOT, LABELS, fingerprint, load_har
from litert_quant.model2 import (
    accuracy,
    input_quantization,
    predict_tflite,
    representative_set,
    to_float_tflite,
    to_int8_tflite,
    to_model_input,
    train,
)

ROOT = Path(__file__).resolve().parents[2]
SEED = 0
REP_SAMPLES = 300
BASELINE = ROOT.parent / "results" / "har_baseline.json"
BASELINE_KEYS = ("test_accuracy", "test_macro_f1", "nodes", "leaves", "chosen_depth", "n_test")


def _per_class_recall(y, pred):
    r = recall_score(y, pred, labels=list(range(len(LABELS))), average=None, zero_division=0)
    return {name: float(v) for name, v in zip(LABELS, r)}


def _macro_f1(y, pred):
    return float(f1_score(y, pred, labels=list(range(len(LABELS))), average="macro", zero_division=0))


def _tree_baseline(path):
    if path is None or not Path(path).is_file():
        return None
    data = json.loads(Path(path).read_text())
    return {k: data.get(k) for k in BASELINE_KEYS}


def main(root=DEFAULT_ROOT, epochs=30, out_root=ROOT, baseline_path=BASELINE):
    x_train, x_test, y_train, y_test, s_train, s_test = load_har(root)
    xtr, xte = to_model_input(x_train), to_model_input(x_test)
    model = train(xtr, y_train, epochs=epochs, seed=SEED)
    keras_pred = np.argmax(model.predict(xte, verbose=0), axis=1)
    float_blob = to_float_tflite(model)
    int8_blob = to_int8_tflite(model, representative_set(xtr, REP_SAMPLES, SEED))
    float_pred = predict_tflite(float_blob, xte)
    int8_pred = predict_tflite(int8_blob, xte)

    out_root = Path(out_root)
    models_dir = out_root / "models"
    results_dir = out_root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "har_cnn_float.tflite").write_bytes(float_blob)
    (models_dir / "har_cnn_int8.tflite").write_bytes(int8_blob)
    np.savez(
        results_dir / "model2_pc_predictions.npz",
        y_test=y_test,
        keras=keras_pred,
        float=float_pred,
        int8=int8_pred,
        subjects_test=s_test,
    )

    scale, zero = input_quantization(int8_blob)
    n_test = int(len(y_test))
    result = {
        "model": "har_cnn",
        "parameters": int(model.count_params()),
        "seed": SEED,
        "epochs": epochs,
        "representative_samples": min(REP_SAMPLES, int(len(xtr))),
        "n_train": int(len(y_train)),
        "n_test": n_test,
        "train_subjects": sorted(int(s) for s in np.unique(s_train)),
        "test_subjects": sorted(int(s) for s in np.unique(s_test)),
        "test_fingerprint": fingerprint(x_test, y_test),
        "correct_keras": int(np.sum(keras_pred == y_test)),
        "correct_float_tflite": int(np.sum(float_pred == y_test)),
        "correct_int8_tflite": int(np.sum(int8_pred == y_test)),
        "accuracy_keras": accuracy(keras_pred, y_test),
        "accuracy_float_tflite": accuracy(float_pred, y_test),
        "accuracy_int8_tflite": accuracy(int8_pred, y_test),
        "macro_f1_float": _macro_f1(y_test, float_pred),
        "macro_f1_int8": _macro_f1(y_test, int8_pred),
        "per_class_recall_float": _per_class_recall(y_test, float_pred),
        "per_class_recall_int8": _per_class_recall(y_test, int8_pred),
        "agree_keras_float": float(np.mean(keras_pred == float_pred)),
        "agree_float_int8": float(np.mean(float_pred == int8_pred)),
        "disagree_float_int8": int(np.sum(float_pred != int8_pred)),
        "float_bytes": len(float_blob),
        "int8_bytes": len(int8_blob),
        "input_scale": scale,
        "input_zero_point": zero,
        "tree_baseline": _tree_baseline(baseline_path),
        "versions": {
            "tensorflow": tf.__version__,
            "ai_edge_litert": version("ai-edge-litert"),
            "numpy": np.__version__,
        },
    }
    (results_dir / "model2_pc.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
