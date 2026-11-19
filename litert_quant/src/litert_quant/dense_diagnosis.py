"""Diagnose the dense-layer scale problem seen on the board.

AI-assisted: drafted with Claude, checked by the author.
Before the fix the exported int8 model had one scale per dense output channel and
the board behaved as if it used the scale of channel 0 for every output. This module
reproduces that from the saved model, the saved board log and the stored test inputs.
"""
import json
import re
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType


def parse_logits(text, n_classes=10):
    rows = {}
    for line in text.splitlines():
        if line.startswith("L,"):
            parts = line.strip().split(",")
            values = [int(v) for v in parts[2:]]
            if len(values) != n_classes:
                raise ValueError("L line has the wrong number of outputs")
            rows[int(parts[1])] = values
    if not rows or sorted(rows) != list(range(len(rows))):
        raise ValueError("L lines are missing or out of order")
    return np.array([rows[i] for i in range(len(rows))], dtype=np.int64)


def load_inputs(header_text, count=360, size=64):
    m = re.search(r"test_input_int8[^{]*\{([^}]*)\}", header_text)
    if m is None:
        raise ValueError("test_input_int8 not found")
    values = [int(v) for v in re.findall(r"-?\d+", m.group(1))]
    if len(values) != count * size:
        raise ValueError("unexpected number of input values")
    return np.array(values, dtype=np.int8).reshape(count, 8, 8, 1)


def per_class_matches(a, b):
    return [int(v) for v in (np.asarray(a) == np.asarray(b)).sum(axis=0)]


def requantise(acc, multiplier, zero_point):
    v = np.asarray(acc) * multiplier
    r = np.sign(v) * np.floor(np.abs(v) + 0.5)
    return np.clip(r + zero_point, -128, 127).astype(np.int64)


def _interpreter(blob):
    it = Interpreter(
        model_content=blob,
        experimental_preserve_all_tensors=True,
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    it.allocate_tensors()
    return it


def run_dense(blob, x):
    """Run the model on x and return the pieces of its dense layer."""
    it = _interpreter(blob)
    fc = [o for o in it._get_ops_details() if o["op_name"] == "FULLY_CONNECTED"]
    if len(fc) != 1:
        raise ValueError("expected exactly one FULLY_CONNECTED op")
    a_i, w_i, b_i = [int(v) for v in fc[0]["inputs"][:3]]
    o_i = int(fc[0]["outputs"][0])
    det = {t["index"]: t for t in it.get_tensor_details()}
    in_i = it.get_input_details()[0]["index"]
    acts, outs = [], []
    for k in range(len(x)):
        it.set_tensor(in_i, x[k:k + 1])
        it.invoke()
        acts.append(it.get_tensor(a_i)[0].astype(np.int64))
        outs.append(it.get_tensor(o_i)[0].astype(np.int64))
    qa = det[a_i]["quantization_parameters"]
    qo = det[o_i]["quantization_parameters"]
    qw = det[w_i]["quantization_parameters"]
    return {
        "acts": np.array(acts),
        "pc": np.array(outs),
        "weights": it.get_tensor(w_i).astype(np.int64),
        "bias": (it.get_tensor(b_i).astype(np.int64) if b_i >= 0
                 else np.zeros(len(it.get_tensor(w_i)), dtype=np.int64)),
        "sx": float(qa["scales"][0]),
        "zx": int(qa["zero_points"][0]),
        "so": float(qo["scales"][0]),
        "zo": int(qo["zero_points"][0]),
        "sw": np.array(qw["scales"], dtype=np.float64),
    }


def recompute(parts):
    """Recompute the dense outputs with each channel's own scale and with channel 0's."""
    acc = (parts["acts"] - parts["zx"]) @ parts["weights"].T + parts["bias"]
    own = requantise(acc, parts["sx"] * parts["sw"] / parts["so"], parts["zo"])
    first = requantise(acc, parts["sx"] * parts["sw"][0] / parts["so"], parts["zo"])
    return own, first


def analyse(prefix_model, prefix_log, fixed_model, fixed_log, header, out_json):
    x = load_inputs(Path(header).read_text())
    pre = run_dense(Path(prefix_model).read_bytes(), x)
    fix = run_dense(Path(fixed_model).read_bytes(), x)
    pre_board = parse_logits(Path(prefix_log).read_text())
    fix_board = parse_logits(Path(fixed_log).read_text())
    own, first = recompute(pre)
    result = {
        "samples": int(len(x)),
        "outputs": int(len(x) * 10),
        "before_fix": {
            "dense_weight_scales": int(len(pre["sw"])),
            "per_class_exact_board_vs_pc": per_class_matches(pre_board, pre["pc"]),
            "exact_board_vs_pc": int((pre_board == pre["pc"]).sum()),
            "recompute_own_scales_vs_pc": int((own == pre["pc"]).sum()),
            "recompute_own_scales_vs_board": int((own == pre_board).sum()),
            "recompute_channel0_scale_vs_board": int((first == pre_board).sum()),
        },
        "after_fix": {
            "dense_weight_scales": int(len(fix["sw"])),
            "per_class_exact_board_vs_pc": per_class_matches(fix_board, fix["pc"]),
            "exact_board_vs_pc": int((fix_board == fix["pc"]).sum()),
            "max_abs_diff_board_vs_pc": int(np.abs(fix_board - fix["pc"]).max()),
        },
    }
    Path(out_json).write_text(json.dumps(result, indent=2) + "\n")
    return result


def make_diagnosis_figure(result_json, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads(Path(result_json).read_text())
    n = res["samples"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    panels = (
        ("before_fix", "Before the fix: dense weights per-channel", "#7F7F7F"),
        ("after_fix", "After the fix: dense weights per-tensor", "#4C78A8"),
    )
    for ax, (key, title, color) in zip(axes, panels):
        counts = res[key]["per_class_exact_board_vs_pc"]
        ax.bar(range(len(counts)), counts, color=color)
        for i, v in enumerate(counts):
            ax.text(i, v + 4, str(v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(counts)))
        ax.set_ylim(0, n * 1.12)
        ax.set_xlabel("output class")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("samples with identical output (of {})".format(n))
    b = res["before_fix"]
    fig.suptitle("Nano 33 BLE int8 outputs vs the PC run, per output class", fontsize=13)
    fig.text(
        0.5, 0.01,
        "Before the fix: recomputing with channel 0's scale for every output matches the board on "
        "{} of {} values; with each channel's own scale, {}.".format(
            b["recompute_channel0_scale_vs_board"], res["outputs"], b["recompute_own_scales_vs_board"]
        ),
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    with open(out_png, "wb") as fh:
        fig.savefig(fh, format="png", dpi=150)
    plt.close(fig)


def main(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    res = root / "results"
    out_json = res / "dense_scale_diagnosis.json"
    result = analyse(
        root / "models" / "digits_cnn_int8_perchannel_dense.tflite",
        res / "board_int8_logits_run1.txt",
        root / "models" / "digits_cnn_int8.tflite",
        res / "board_int8_fixed_run1.txt",
        root / "firmware" / "digits_bench" / "test_data.h",
        out_json,
    )
    print("before_fix", json.dumps(result["before_fix"]))
    print("after_fix", json.dumps(result["after_fix"]))
    print("wrote", out_json)
    out_png = res / "figures" / "dense_scale_diagnosis.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    make_diagnosis_figure(out_json, out_png)
    print("wrote", out_png)
    return 0
