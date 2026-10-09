"""Measure the flash and static RAM an exported model needs on a Cortex-M, with the real ARM compiler.

Method: the exported classifier is linked with a tiny test harness for each target, then an otherwise identical build with
an empty stub in place of the classifier is subtracted, so the numbers are the classifier's own cost (code, constants and any
compiler helper routines it pulls in, such as soft-float compares on a core without an FPU). Stack, speed and feature
extraction are NOT included, and nothing here runs on a board.
"""
import ctypes
import glob
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from .export_c import forest_to_c, tree_to_c

TARGETS = {
    "cortex-m0plus": {"label": "Cortex-M0+ (no FPU): STM32L0, SAMD21",
                      "flags": ["-mcpu=cortex-m0plus", "-mthumb", "-mfloat-abi=soft"]},
    "cortex-m4f": {"label": "Cortex-M4F (FPU): nRF52840 on the Nano 33 BLE, Apollo3 on the SparkFun Edge",
                   "flags": ["-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=hard", "-mfpu=fpv4-sp-d16"]},
}
COMMON_FLAGS = ["-Os", "-ffunction-sections", "-fdata-sections"]
LINK_FLAGS = ["-nostartfiles", "-nostdlib", "-Wl,--gc-sections", "-Wl,-e,main"]
GCC_NAMES = ("arm-none-eabi-gcc", "arm-zephyr-eabi-gcc")
SDK_DIRS = ("~/zephyr-sdk*", "~/*/zephyr-sdk*", "~/.local/zephyr-sdk*", "/opt/zephyr-sdk*", "/usr/local/zephyr-sdk*", "~/ncs/toolchains/*")
SDK_SUBPATHS = ("gnu/arm-zephyr-eabi/bin/arm-zephyr-eabi-gcc", "arm-zephyr-eabi/bin/arm-zephyr-eabi-gcc", "bin/arm-none-eabi-gcc")


def _tool_from_gcc(gcc):
    gcc = Path(gcc)
    size = gcc.with_name(gcc.name[:-3] + "size")
    if not (gcc.is_file() and size.is_file()):
        return None
    try:
        ver = subprocess.run([str(gcc), "--version"], capture_output=True, text=True, timeout=30).stdout.splitlines()[0]
    except Exception:  # noqa: BLE001
        return None
    return {"gcc": str(gcc), "size": str(size), "version": ver.strip()}


def find_toolchain(env=None):
    """Find an ARM gcc (and its size tool). Order: ARM_GCC, ZEPHYR_SDK_INSTALL_DIR, PATH, the usual SDK folders."""
    env = os.environ if env is None else env
    cands = []
    if env.get("ARM_GCC"):
        cands.append(env["ARM_GCC"])
    sdk = env.get("ZEPHYR_SDK_INSTALL_DIR")
    if sdk:
        cands += [f"{sdk}/gnu/arm-zephyr-eabi/bin/arm-zephyr-eabi-gcc", f"{sdk}/arm-zephyr-eabi/bin/arm-zephyr-eabi-gcc"]
    for name in GCC_NAMES:
        found = shutil.which(name, path=env.get("PATH"))
        if found:
            cands.append(found)
    for pattern in SDK_DIRS:
        for d in sorted(glob.glob(os.path.expanduser(pattern)), reverse=True):
            cands += [f"{d}/{sub}" for sub in SDK_SUBPATHS]
    for c in cands:
        tool = _tool_from_gcc(c)
        if tool:
            return tool
    return None


def parse_berkeley_size(text):
    """Parse `size -B` output into {'text', 'data', 'bss'} (bytes)."""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and all(p.isdigit() for p in parts[:3]):
            return {"text": int(parts[0]), "data": int(parts[1]), "bss": int(parts[2])}
    raise ValueError("could not read the size tool's output")


def _build(tool, target, workdir, files, tag):
    elf = Path(workdir) / f"{tag}.elf"
    cmd = [tool["gcc"], *TARGETS[target]["flags"], *COMMON_FLAGS, *LINK_FLAGS, *[str(f) for f in files], "-lgcc", "-o", str(elf)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"compiling for {target} failed:\n{r.stderr.strip()[:1500]}")
    s = subprocess.run([tool["size"], "-B", str(elf)], capture_output=True, text=True, check=True)
    return parse_berkeley_size(s.stdout)


def measure(c_source, func_name, n_features, target, tool, workdir=None):
    """Flash (code + constants + initialised data) and static RAM of the classifier alone, for one target."""
    own = workdir is None
    workdir = Path(tempfile.mkdtemp() if own else workdir)
    try:
        harness = workdir / "harness.c"
        harness.write_text(
            f"extern int {func_name}(const float *x);\n"
            f"static volatile float input[{n_features}];\n"
            "int main(void)\n{\n"
            f"    float x[{n_features}];\n    int i;\n"
            f"    for (i = 0; i < {n_features}; ++i) x[i] = input[i];\n"
            f"    return {func_name}(x);\n}}\n")
        model = workdir / "model.c"
        model.write_text(c_source)
        stub = workdir / "stub.c"
        stub.write_text(f"int {func_name}(const float *x)\n{{\n    (void)x;\n    return 0;\n}}\n")
        with_model = _build(tool, target, workdir, [harness, model], "with_model")
        empty = _build(tool, target, workdir, [harness, stub], "stub")
        return {"flash_bytes": (with_model["text"] + with_model["data"]) - (empty["text"] + empty["data"]),
                "ram_bytes": (with_model["data"] + with_model["bss"]) - (empty["data"] + empty["bss"]),
                "linked_flash_bytes": with_model["text"] + with_model["data"],
                "stub_flash_bytes": empty["text"] + empty["data"]}
    finally:
        if own:
            shutil.rmtree(workdir, ignore_errors=True)


def host_agreement(c_source, func_name, model, X):
    """On this PC: how often the compiled C function returns the same class as scikit-learn on the rows of X."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    with tempfile.TemporaryDirectory() as d:
        c, so = Path(d) / "m.c", Path(d) / "m.so"
        c.write_text(c_source)
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", str(so), str(c)], check=True, capture_output=True)
        lib = ctypes.CDLL(str(so))
        fn = getattr(lib, func_name)
        fn.argtypes = [ctypes.POINTER(ctypes.c_float)]
        fn.restype = ctypes.c_int
        got = np.array([fn(row.ctypes.data_as(ctypes.POINTER(ctypes.c_float))) for row in X])
    want = model.predict(X)
    return {"matches": int(np.sum(got == want)), "total": int(len(X)), "rate": float(np.mean(got == want))}


def variants_for(model):
    """(variant name, function name, C source) for every export of a fitted model."""
    if isinstance(model, RandomForestClassifier):
        return [("float thresholds", "forest_predict", forest_to_c(model, "forest_predict"))]
    return [("double thresholds", "tree_predict", tree_to_c(model, "tree_predict", "double")),
            ("float thresholds", "tree_predict", tree_to_c(model, "tree_predict", "float"))]


def measure_model(model, X_test, tool, targets=tuple(TARGETS)):
    """All variants of one model: C size, flash/RAM per target, and agreement with scikit-learn on X_test."""
    n_features = int(X_test.shape[1])
    out = []
    for variant, func, src in variants_for(model):
        row = {"variant": variant, "c_source_bytes": len(src.encode("utf-8")),
               "agreement": host_agreement(src, func, model, X_test), "targets": {}}
        for t in targets:
            row["targets"][t] = measure(src, func, n_features, t, tool)
        out.append(row)
    return out


def refit_models(report, train_sets, y_train):
    """Rebuild exactly the models of results/compare.json from its recorded settings. Returns {id: (feature_set, model)}."""
    seed = report["protocol"]["seed"]
    ch = report["chosen"]
    fx = report["protocol"]["fixed_depth"]
    out = {
        "tree_fixed6": ("baseline", DecisionTreeClassifier(max_depth=fx, random_state=seed)),
        "tree_baseline_cv": ("baseline", DecisionTreeClassifier(max_depth=ch["tree_baseline"]["max_depth"], random_state=seed)),
        "tree_extended_cv": ("extended", DecisionTreeClassifier(max_depth=ch["tree_extended"]["max_depth"], random_state=seed)),
        "forest_extended_cv": ("extended", RandomForestClassifier(n_estimators=ch["forest_extended"]["n_estimators"],
                                                                  max_depth=ch["forest_extended"]["max_depth"],
                                                                  random_state=seed, n_jobs=-1)),
    }
    return {k: (fs, est.fit(train_sets[fs], y_train)) for k, (fs, est) in out.items()}


def _node_count(est):
    if isinstance(est, RandomForestClassifier):
        return int(sum(t.tree_.node_count for t in est.estimators_))
    return int(est.tree_.node_count)


def refit_matches_saved(report, models, test_sets, y_test):
    """Does each refitted model reproduce the test accuracy AND the node count saved in compare.json? {id: bool}"""
    saved = {m["id"]: m for m in report["models"]}
    return {k: bool(abs(accuracy_score(y_test, est.predict(test_sets[fs])) - saved[k]["test_accuracy"]) < 1e-12
                    and _node_count(est) == saved[k]["nodes"])
            for k, (fs, est) in models.items()}
