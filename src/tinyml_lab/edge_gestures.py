"""Desk-gesture data from the SparkFun Edge: reading captures, resampling, windowing and features.

Input: the CSV files written by tools/edge_capture.py (usually through tools/record_session.py). Each file is one
gesture recorded in one session, with its settings in '#' header lines and a body of "t_s,x_mg,y_mg,z_mg".

Why resample: the Edge's accelerometer ran at about 51.2 Hz for a configured 50 Hz, and its sample times jitter by
about a millisecond. The recorded timestamps are real, so each recording is interpolated onto an exact 50 Hz grid,
the rate of the UCI data and of anything that will run on a device later.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import batch_features, feature_names, with_magnitude

FS = 50.0                 # Hz, the grid every recording is resampled to
WINDOW = 128              # samples per window (2.56 s), as for the UCI data
STEP = 64                 # 50 % overlap
TRIM_S = 2.0              # seconds dropped at both ends of a recording (getting started, stopping)
N_BANDS = 4
GESTURES = ("rest", "shake_lr", "shake_ud", "circle", "tilt")   # the recommended set; other labels also work
FEATURE_SETS = ("raw", "motion")
HEADER = "t_s,x_mg,y_mg,z_mg"
_TEXT_KEYS = ("label", "subject", "session", "note", "verdict", "rate_note")


@dataclass
class Capture:
    """One capture file as read by read_capture: its label, subject, session and note, the verdict and timing from its header, and the samples
    (t_s in seconds, xyz_mg in milli-g).
    """
    path: Path
    label: str
    subject: str
    session: str
    note: str
    verdict: str
    odr_hz: int | None
    tick_hz: int | None
    timer: str
    measured_rate_hz: float | None
    synthetic: bool
    t_s: np.ndarray
    xyz_mg: np.ndarray


@dataclass
class Dataset:
    """Windows cut from every usable capture under a folder (load_dataset): the windows (N, win, 3) in milli-g, with each window's gesture, session,
    subject and file, the settings used, and the files used and skipped.
    """
    windows: np.ndarray           # (N, win, 3) milli-g
    y: np.ndarray                 # (N,) gesture names
    session: np.ndarray           # (N,) session names
    subject: np.ndarray
    file: np.ndarray              # (N,) file names
    fs: float
    win: int
    step: int
    trim_s: float
    synthetic: bool
    files: list = field(default_factory=list)      # one dict per file used
    skipped: list = field(default_factory=list)    # (file name, reason)


def _parse_comment(line: str, meta: dict) -> None:
    m = re.match(r"#\s*(label|subject|session|note|verdict|rate_note)=(.*)$", line)
    if m:
        meta[m.group(1)] = m.group(2).strip()
        return
    for k, v in re.findall(r"(\w+)=(\S+)", line):
        meta.setdefault(k, v)


def read_capture(path) -> Capture:
    """Read one capture file. Handles Windows (CRLF) and Unix line endings; raises ValueError with the line number."""
    path = Path(path)
    meta: dict = {}
    rows = []
    header_seen = False
    for lineno, raw in enumerate(path.read_bytes().decode("utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            _parse_comment(line, meta)
            continue
        if not header_seen:
            if line != HEADER:
                raise ValueError(f"{path.name}:{lineno}: expected the header '{HEADER}', found {line!r}")
            header_seen = True
            continue
        parts = line.split(",")
        if len(parts) != 4:
            raise ValueError(f"{path.name}:{lineno}: expected 4 comma-separated values, found {len(parts)}")
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            raise ValueError(f"{path.name}:{lineno}: not numbers: {line!r}") from None
    if not header_seen:
        raise ValueError(f"{path.name}: no '{HEADER}' header line")
    if len(rows) < 2:
        raise ValueError(f"{path.name}: fewer than two samples")
    arr = np.asarray(rows, dtype=np.float64)
    if not np.all(np.diff(arr[:, 0]) > 0):
        raise ValueError(f"{path.name}: timestamps are not strictly increasing")

    def _num(key, cast):
        try:
            return cast(meta[key])
        except (KeyError, ValueError):
            return None

    return Capture(path=path, label=meta.get("label", "").strip().lower(), subject=meta.get("subject", ""),
                   session=meta.get("session", "").strip().lower(), note=meta.get("note", ""),
                   verdict=meta.get("verdict", ""), odr_hz=_num("odr_hz", int), tick_hz=_num("tick_hz", int),
                   timer=meta.get("timer", ""), measured_rate_hz=_num("measured_rate_hz", float),
                   synthetic=meta.get("synthetic", "0") not in ("0", ""), t_s=arr[:, 0], xyz_mg=arr[:, 1:4])


def resample_to_grid(t_s, xyz, fs: float = FS) -> np.ndarray:
    """Linear interpolation onto an exact grid 0, 1/fs, 2/fs, ... within the recorded span. Returns (n, 3)."""
    t = np.asarray(t_s, dtype=np.float64)
    t = t - t[0]
    xyz = np.asarray(xyz, dtype=np.float64)
    n = int(t[-1] * fs) + 1
    grid = np.arange(n) / fs
    return np.column_stack([np.interp(grid, t, xyz[:, i]) for i in range(xyz.shape[1])])


def trim_edges(x: np.ndarray, fs: float = FS, trim_s: float = TRIM_S) -> np.ndarray:
    """Drop trim_s seconds from both ends of a recording (getting started, stopping). No change if trim_s is half a sample or less."""
    k = int(round(trim_s * fs))
    return x[k:len(x) - k] if k > 0 else x


def make_windows(x: np.ndarray, win: int = WINDOW, step: int = STEP) -> np.ndarray:
    """(n_samples, c) -> (n_windows, win, c); no padding, a partial last window is dropped."""
    x = np.asarray(x)
    if len(x) < win:
        return np.empty((0, win, x.shape[1]))
    n = (len(x) - win) // step + 1
    idx = np.arange(win)[None, :] + step * np.arange(n)[:, None]
    return x[idx]


def load_dataset(root, fs: float = FS, win: int = WINDOW, step: int = STEP, trim_s: float = TRIM_S,
                 allow_problems: bool = False) -> Dataset:
    """Every usable capture under `root` as windows. Files that cannot be used are listed in `.skipped` with the reason."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} does not exist. Record sessions with tools/record_session.py first.")
    wins, ys, sess, subj, names, files, skipped, synth = [], [], [], [], [], [], [], set()
    for p in sorted(root.rglob("*.csv")):
        if p.name.endswith("_50hz.csv"):          # the capture tool's own resampled copy: we resample from the raw file
            continue
        try:
            cap = read_capture(p)
        except ValueError as e:
            skipped.append((p.name, str(e)))
            continue
        if not cap.label:
            skipped.append((p.name, "no label in the header"))
            continue
        if not cap.session:
            skipped.append((p.name, "no session in the header (record it again with --session, or use tools/record_session.py)"))
            continue
        if cap.verdict.startswith("PROBLEMS") and not allow_problems:
            skipped.append((p.name, f"the capture tool flagged it: {cap.verdict}"))
            continue
        x = trim_edges(resample_to_grid(cap.t_s, cap.xyz_mg, fs), fs, trim_s)
        w = make_windows(x, win, step)
        if len(w) == 0:
            skipped.append((p.name, f"too short: no complete {win}-sample window after trimming {trim_s:g} s at each end"))
            continue
        synth.add(cap.synthetic)
        wins.append(w)
        n = len(w)
        ys += [cap.label] * n; sess += [cap.session] * n; subj += [cap.subject] * n; names += [p.name] * n
        files.append({"file": p.name, "label": cap.label, "session": cap.session, "windows": n,
                      "measured_rate_hz": cap.measured_rate_hz})
    if len(synth) > 1:
        raise ValueError("synthetic and real recordings are mixed in " + str(root) + "; keep them in separate folders")
    if not wins:
        why = "; ".join(f"{n}: {r}" for n, r in skipped[:5]) or "no .csv files found"
        raise ValueError(f"no usable recordings under {root} ({why})")
    return Dataset(windows=np.concatenate(wins), y=np.array(ys), session=np.array(sess), subject=np.array(subj),
                   file=np.array(names), fs=fs, win=win, step=step, trim_s=trim_s, synthetic=synth.pop(),
                   files=files, skipped=skipped)


def design_problems(ds: Dataset) -> list[str]:
    """Things that make a leave-one-session-out test weak or impossible. Empty means the design is sound."""
    out = []
    sessions = sorted(set(ds.session.tolist()))
    labels = sorted(set(ds.y.tolist()))
    if len(labels) < 2:
        out.append("only one gesture label: there is nothing to classify")
    if len(sessions) < 2:
        out.append(f"only {len(sessions)} session: leave-one-session-out needs at least two (three or more is better)")
    elif len(sessions) == 2:
        out.append("only two sessions: each test uses a model trained on a single sitting; three or more is better")
    for lab in labels:
        have = sorted(set(ds.session[ds.y == lab].tolist()))
        if len(have) < 2:
            out.append(f"gesture '{lab}' appears in only {len(have)} session: it cannot be tested on an unseen sitting")
    for s in sessions:
        missing = [lab for lab in labels if not np.any((ds.session == s) & (ds.y == lab))]
        if missing:
            out.append(f"session '{s}' has no recording of: {', '.join(missing)}")
    return out


def feature_names_for(feature_set: str, n_bands: int = N_BANDS) -> list[str]:
    """Feature names of a feature set: 'raw' (x, y, z and the magnitude, 10 features each) or 'motion' (the same without the x, y and z means, so
    orientation drops out). Raises ValueError for any other name.
    """
    if feature_set == "raw":
        return feature_names(["x", "y", "z", "mag"], n_bands)
    if feature_set == "motion":
        drop = {"x_mean", "y_mean", "z_mean"}
        return [n for n in feature_names(["x", "y", "z", "mag"], n_bands) if n not in drop]
    raise ValueError(f"feature_set must be one of {FEATURE_SETS}")


def feature_matrix(windows, fs: float = FS, feature_set: str = "motion", n_bands: int = N_BANDS) -> np.ndarray:
    """Features per window.

    raw:    x, y, z and the vector magnitude, as measured. The means carry the board's orientation (gravity direction),
            which tells a model how the board was held rather than what the hand did.
    motion: each window's own mean is removed from x, y and z first, and 'mag' is the size of what is left, so gravity
            and orientation drop out to first order. The x, y, z means (all zero by then) are not used.
    """
    w = np.asarray(windows, dtype=np.float64)
    if feature_set == "raw":
        return batch_features(with_magnitude(w), fs, n_bands)
    if feature_set == "motion":
        ac = w - w.mean(axis=1, keepdims=True)
        chans = np.concatenate([ac, np.linalg.norm(ac, axis=-1, keepdims=True)], axis=-1)
        X = batch_features(chans, fs, n_bands)
        names = feature_names(["x", "y", "z", "mag"], n_bands)
        keep = [i for i, n in enumerate(names) if n not in ("x_mean", "y_mean", "z_mean")]
        return np.ascontiguousarray(X[:, keep])   # dropping columns leaves column-major data; rows must be contiguous to hand to C
    raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
