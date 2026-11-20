"""Model 2: a small 1D-CNN on raw HAR windows, as float and int8 LiteRT files.

AI-assisted: drafted with Claude, checked by the author.
The 1D convolutions are written as Conv2D layers with (5, 1) kernels on
windows shaped (128, 1, 3). That is the same arithmetic as Conv1D, and it
keeps the exported graph free of EXPAND_DIMS and SQUEEZE ops, so the board
needs only CONV_2D, RESHAPE and FULLY_CONNECTED.
"""
import numpy as np
import tensorflow as tf

# Reused unchanged from model 1.
from litert_quant.model1 import accuracy, input_quantization, predict_tflite  # noqa: F401

N_CLASSES = 6
WINDOW = 128
CHANNELS = 3


def to_model_input(x):
    """(N, 128, 3) windows to (N, 128, 1, 3), the shape the network takes."""
    return np.ascontiguousarray(x[:, :, np.newaxis, :])


def build_cnn(batch_size=None):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(WINDOW, 1, CHANNELS), batch_size=batch_size),
            tf.keras.layers.Conv2D(8, (5, 1), strides=(2, 1), activation="relu"),
            tf.keras.layers.Conv2D(16, (5, 1), strides=(2, 1), activation="relu"),
            tf.keras.layers.Conv2D(16, (5, 1), strides=(2, 1), activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(N_CLASSES),
        ]
    )


def train(x_train, y_train, epochs=30, seed=0):
    """Fixed epochs, no validation split and no selection on the test set."""
    tf.keras.utils.set_random_seed(seed)
    model = build_cnn()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=0)
    return model


def representative_set(x_train, n=300, seed=0):
    """Seeded random training windows, used only to choose the int8 scales."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(x_train), size=min(n, len(x_train)), replace=False))
    return x_train[idx]


def fixed_batch_copy(model):
    """Same weights with the batch size fixed at 1 (no shape-handling ops).

    Keras numbers layers and tensors with a global counter and the numbers end
    up as tensor names in the .tflite file, so the counter is reset first. That
    makes the exported bytes independent of how many models were built before.
    """
    tf.keras.backend.clear_session()
    copy = build_cnn(batch_size=1)
    copy.set_weights(model.get_weights())
    return copy


def to_float_tflite(model):
    return tf.lite.TFLiteConverter.from_keras_model(fixed_batch_copy(model)).convert()


def to_int8_tflite(model, x_rep):
    """Full-integer model with int8 input and output.

    x_rep is the representative set used to choose the scales.
    """
    conv = tf.lite.TFLiteConverter.from_keras_model(fixed_batch_copy(model))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep():
        for i in range(len(x_rep)):
            yield [x_rep[i : i + 1]]

    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    # TFLite Micro applies one output scale to every output of a dense layer,
    # so dense weights must be per-tensor (measured on the Nano 33 BLE).
    conv._experimental_disable_per_channel_quantization_for_dense_layers = True
    return conv.convert()
