"""Model 3: a small dense autoencoder on 128-sample CWRU windows, float and int8 LiteRT files.

AI-assisted: drafted with Claude, checked by the author.
It is trained on normal windows only. The anomaly score is the mean squared error
between the preprocessed input window and the reconstruction. For an int8 model the
score uses the dequantised int8 input and output, which is the arithmetic the board
can do. Windows are divided by the training std and clipped at the largest scaled
training value, so float and int8 see the same input and the int8 input range is set
by normal data only. The network is four FULLY_CONNECTED layers.
"""
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

# Reused unchanged from models 1 and 2.
from litert_quant.model1 import input_quantization  # noqa: F401
from litert_quant.model2 import representative_set  # noqa: F401

WINDOW = 128
HIDDEN = (16, 8, 16)


def fit_scaler(x_train):
    """Return (std, clip) from the training windows only."""
    x = np.asarray(x_train, dtype=np.float64)
    std = float(np.std(x))
    clip = float(np.ceil(np.max(np.abs(x)) / std))
    return std, clip


def preprocess(x, std, clip):
    """Divide by the training std and clip, as float32."""
    scaled = np.asarray(x, dtype=np.float32) / np.float32(std)
    return np.clip(scaled, -clip, clip).astype(np.float32)


def build_autoencoder(batch_size=None):
    layers = [tf.keras.layers.Input(shape=(WINDOW,), batch_size=batch_size)]
    for units in HIDDEN:
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
    layers.append(tf.keras.layers.Dense(WINDOW))
    return tf.keras.Sequential(layers)


def train(x_train, epochs=60, seed=0):
    """Fixed epochs, no validation split and no selection on the test set."""
    tf.keras.utils.set_random_seed(seed)
    model = build_autoencoder()
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_train, x_train, epochs=epochs, batch_size=64, verbose=0, shuffle=True)
    return model


def mse_scores(x, recon):
    """Per-window mean squared error."""
    return np.mean((np.asarray(recon) - np.asarray(x)) ** 2, axis=1).astype(np.float64)


def fixed_batch_copy(model):
    """Same weights with the batch size fixed at 1; the counter reset keeps exported bytes stable."""
    tf.keras.backend.clear_session()
    copy = build_autoencoder(batch_size=1)
    copy.set_weights(model.get_weights())
    return copy


def to_float_tflite(model):
    return tf.lite.TFLiteConverter.from_keras_model(fixed_batch_copy(model)).convert()


def to_int8_tflite(model, x_rep):
    """Full-integer model with int8 input and output; x_rep chooses the scales."""
    conv = tf.lite.TFLiteConverter.from_keras_model(fixed_batch_copy(model))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep():
        for i in range(len(x_rep)):
            yield [x_rep[i : i + 1]]

    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    # Dense weights per-tensor, as measured on the board for models 1 and 2.
    conv._experimental_disable_per_channel_quantization_for_dense_layers = True
    return conv.convert()


def predict_scores(tflite_bytes, x):
    """Anomaly score of each window from a .tflite model, run with the LiteRT interpreter."""
    interp = Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    scores = np.empty(len(x), dtype=np.float64)
    for i in range(len(x)):
        sample = np.asarray(x[i : i + 1], dtype=np.float32)
        if inp["dtype"] == np.int8:
            q = np.clip(np.round(sample / in_scale) + in_zero, -128, 127).astype(np.int8)
            interp.set_tensor(inp["index"], q)
            ref = (q.astype(np.float32) - in_zero) * in_scale
        else:
            interp.set_tensor(inp["index"], sample)
            ref = sample
        interp.invoke()
        y = interp.get_tensor(out["index"]).astype(np.float32)
        if out["dtype"] == np.int8:
            y = (y - out_zero) * out_scale
        scores[i] = float(np.mean((y - ref) ** 2))
    return scores
