"""Run model 1 on the PC: train, convert, and score float and int8.

AI-assisted: drafted with Claude, checked by the author.
Writes the two .tflite files, results/model1_pc.json and the predictions.
"""
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import tensorflow as tf

from litert_quant.data import SEED, TEST_FRACTION, fingerprint, load_split
from litert_quant.model1 import (
    accuracy,
    input_quantization,
    predict_tflite,
    to_float_tflite,
    to_int8_tflite,
    train,
)

ROOT = Path(__file__).resolve().parents[2]
REP_SAMPLES = 300


def main(epochs=40, out_root=ROOT):
    out_root = Path(out_root)
    x_train, x_test, y_train, y_test = load_split()
    model = train(x_train, y_train, epochs=epochs, seed=SEED)
    keras_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    float_blob = to_float_tflite(model)
    int8_blob = to_int8_tflite(model, x_train[:REP_SAMPLES])
    float_pred = predict_tflite(float_blob, x_test)
    int8_pred = predict_tflite(int8_blob, x_test)

    models_dir = out_root / "models"
    results_dir = out_root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "digits_cnn_float.tflite").write_bytes(float_blob)
    (models_dir / "digits_cnn_int8.tflite").write_bytes(int8_blob)
    np.savez(
        results_dir / "model1_pc_predictions.npz",
        y_test=y_test,
        keras=keras_pred,
        float=float_pred,
        int8=int8_pred,
    )

    scale, zero = input_quantization(int8_blob)
    result = {
        "model": "digits_cnn",
        "parameters": int(model.count_params()),
        "seed": SEED,
        "epochs": epochs,
        "test_fraction": TEST_FRACTION,
        "representative_samples": REP_SAMPLES,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "test_fingerprint": fingerprint(x_test, y_test),
        "accuracy_keras": accuracy(keras_pred, y_test),
        "accuracy_float_tflite": accuracy(float_pred, y_test),
        "accuracy_int8_tflite": accuracy(int8_pred, y_test),
        "agree_keras_float": float(np.mean(keras_pred == float_pred)),
        "agree_float_int8": float(np.mean(float_pred == int8_pred)),
        "disagree_float_int8": int(np.sum(float_pred != int8_pred)),
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
    (results_dir / "model1_pc.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
