"""Model 1: a small CNN on the 8x8 digits, as float and int8 LiteRT files.

AI-assisted: drafted with Claude, checked by the author.
"""
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter


def build_cnn(batch_size=None):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(8, 8, 1), batch_size=batch_size),
            tf.keras.layers.Conv2D(8, 3, activation="relu"),
            tf.keras.layers.Conv2D(16, 3, strides=2, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ]
    )


def train(x_train, y_train, epochs=40, seed=0):
    tf.keras.utils.set_random_seed(seed)
    model = build_cnn()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
    )
    return model


def fixed_batch_copy(model):
    """Same weights with the batch size fixed at 1.

    The exported graph then has no shape-handling ops, which would only
    add kernels to the flash image of a microcontroller.
    """
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
    return conv.convert()


def input_quantization(tflite_bytes):
    """Return (scale, zero_point) of the model input."""
    interp = Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    scale, zero = interp.get_input_details()[0]["quantization"]
    return float(scale), int(zero)


def predict_tflite(tflite_bytes, x):
    """Class predictions of a .tflite model, run with the LiteRT interpreter."""
    interp = Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    scale, zero = inp["quantization"]
    preds = []
    for i in range(len(x)):
        sample = x[i : i + 1]
        if inp["dtype"] == np.int8:
            sample = np.clip(np.round(sample / scale) + zero, -128, 127)
            sample = sample.astype(np.int8)
        interp.set_tensor(inp["index"], sample)
        interp.invoke()
        preds.append(int(np.argmax(interp.get_tensor(out["index"])[0])))
    return np.array(preds)


def accuracy(pred, y):
    return float(np.mean(np.asarray(pred) == np.asarray(y)))
