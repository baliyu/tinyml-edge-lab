"""Compile the C feature code with the host compiler and compare it with the Python features.

The C code (firmware/sparkfun_edge/gesture_features/edge_features.c) is what will run on the Edge. The Python features
(tinyml_lab.edge_gestures.feature_matrix) are what the models are trained on. Compiling the C file as 'double' shows
that the algorithm is the same; compiling it as 'float', what the board runs, shows what float rounding costs, and
prediction_agreement() shows whether that ever changes a trained model's answer.
"""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from . import edge_gestures as eg

C_DIR = Path(__file__).resolve().parents[2] / "firmware" / "sparkfun_edge" / "gesture_features"
N_FEATURES = {"raw": 40, "motion": 37}


def find_gcc():
    return shutil.which("gcc") or shutil.which("cc")


class CFeatures:
    """The C feature code, built with the host compiler as 'float' or 'double' and called through ctypes."""

    def __init__(self, real: str = "float", workdir=None):
        if real not in ("float", "double"):
            raise ValueError("real must be 'float' or 'double'")
        gcc = find_gcc()
        if gcc is None:
            raise RuntimeError("no C compiler found (install gcc)")
        workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="edge_features_"))
        workdir.mkdir(parents=True, exist_ok=True)
        so = workdir / f"libedge_features_{real}.so"
        subprocess.run([gcc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic", "-shared", "-fPIC",
                        f"-DEF_REAL={real}", "-I", str(C_DIR), str(C_DIR / "edge_features.c"), "-o", str(so), "-lm"],
                       check=True, capture_output=True, text=True)
        self.real = real
        self._ctype = ctypes.c_double if real == "double" else ctypes.c_float
        self._dtype = np.float64 if real == "double" else np.float32
        lib = ctypes.CDLL(str(so))
        ptr = ctypes.POINTER(self._ctype)
        self._fn = {"raw": lib.ef_raw_features, "motion": lib.ef_motion_features}
        for fn in self._fn.values():
            fn.argtypes = [ptr, ptr]
            fn.restype = None

    def compute(self, windows, feature_set: str = "motion") -> np.ndarray:
        """Features of every window, as float64. windows: (N, 128, 3) milli-g."""
        w = np.asarray(windows)
        if w.ndim != 3 or w.shape[1:] != (eg.WINDOW, 3):
            raise ValueError(f"windows must have shape (N, {eg.WINDOW}, 3)")
        fn, n_out = self._fn[feature_set], N_FEATURES[feature_set]
        ptr = ctypes.POINTER(self._ctype)
        out = np.zeros((len(w), n_out), dtype=np.float64)
        buf = np.zeros(n_out, dtype=self._dtype)
        for i in range(len(w)):
            a = np.ascontiguousarray(w[i], dtype=self._dtype).reshape(-1)
            fn(a.ctypes.data_as(ptr), buf.ctypes.data_as(ptr))
            out[i] = buf
        return out


def compare_features(py: np.ndarray, c: np.ndarray, feature_set: str) -> dict:
    """How far apart two feature matrices are. Errors are measured against each feature's own size (its largest value)."""
    scale = np.maximum(np.abs(py).max(axis=0), 1e-12)
    err = np.abs(np.asarray(c) - py) / scale
    worst = int(np.argmax(err.max(axis=0)))
    names = eg.feature_names_for(feature_set)
    names_dom = [i for i, n in enumerate(names) if n.endswith("dom_freq")]
    return {"windows": int(len(py)), "max_error_vs_feature_size": float(err.max()), "worst_feature": names[worst],
            "windows_with_a_different_dominant_bin": int(np.sum(np.any(np.abs(np.asarray(c)[:, names_dom] - py[:, names_dom]) > 1e-6, axis=1)))}


def _models(seed=0):
    return {"tree (depth 6)": lambda: DecisionTreeClassifier(max_depth=6, random_state=seed),
            "forest (100 trees)": lambda: RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)}


def prediction_agreement(X_py: np.ndarray, X_c: np.ndarray, y, sessions) -> dict:
    """How often a trained model answers differently on the float C features than on the Python features.

    'seen': the model is trained on all windows and asked about the same windows (the optimistic case).
    'unseen sessions': for each session, train on the others and ask about it (the case that matters on the board).
    Both feature matrices are cast to float32 first, which is what a float32 tree sees."""
    y, sessions = np.asarray(y), np.asarray(sessions)
    Xp, Xc = np.asarray(X_py, dtype=np.float32), np.asarray(X_c, dtype=np.float32)
    out = {}
    for name, make in _models().items():
        clf = make().fit(Xp, y)
        res = {"seen": {"windows": int(len(y)), "different": int(np.sum(clf.predict(Xp) != clf.predict(Xc)))}}
        held, diff = 0, 0
        for s in sorted(set(sessions.tolist())):
            tr, te = sessions != s, sessions == s
            if len(set(sessions.tolist())) < 2 or len(set(y[tr].tolist())) < 2:
                continue
            m = make().fit(Xp[tr], y[tr])
            held += int(te.sum())
            diff += int(np.sum(m.predict(Xp[te]) != m.predict(Xc[te])))
        if held:
            res["unseen sessions"] = {"windows": held, "different": diff}
        out[name] = res
    return out
