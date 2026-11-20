"""CWRU bearing data for model 3: drive-end windows, one fixed split by recording and load.

AI-assisted: drafted with Claude, checked by the author.
Only the drive-end (DE) channel is used. The files carry no sampling rate; 48 kHz is
assumed from the recording lengths and checked by spectrum (see the README limits).
The signal is decimated by 4 with an FIR anti-alias filter to 12 kHz, then cut into
non-overlapping windows. File numbers follow the CWRU numbering and were checked
against the RPM stored in the files that have one (about 1797, 1772, 1750, 1725 for 0 to
3 hp; 98.mat and 99.mat have none).
Some files also hold arrays named for other files (99.mat has X098, 217.mat has X215,
175.mat has X217), so each file's own channel is used. 174.mat holds only X173_DE_time, so
its provenance is uncertain and it is excluded.
Split: normal recordings at 0, 1 and 2 hp train the autoencoder; normal at 3 hp and all
fault recordings are test. No recording is in both sets (checked by assertion).
"""
import hashlib
from collections import namedtuple
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate

FS_RAW = 48000
DECIMATE = 4
FS = FS_RAW // DECIMATE
WINDOW = 128
TRAIN_LOADS = (0, 1, 2)
DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "cwru"

Recording = namedtuple("Recording", "num kind size_mil load_hp")
Split = namedtuple("Split", "x_train rec_train x_test y_test rec_test load_test kind_test")

NORMAL_FILES = {0: 97, 1: 98, 2: 99, 3: 100}
FAULT_GROUPS = (
    ("IR", 7, (109, 110, 111, 112)),
    ("B", 7, (122, 123, 124, 125)),
    ("OR", 7, (135, 136, 137, 138)),
    ("IR", 14, (174, 175, 176, 177)),
    ("B", 14, (189, 190, 191, 192)),
    ("OR", 14, (201, 202, 203, 204)),
    ("IR", 21, (213, 214, 215, 217)),
    ("B", 21, (226, 227, 228, 229)),
    ("OR", 21, (238, 239, 240, 241)),
)
EXCLUDED = {174: "only channel is X173_DE_time"}


def catalogue():
    """Return the usable recordings (39): 4 normal and 35 fault, 174 excluded."""
    recs = []
    for load, num in NORMAL_FILES.items():
        recs.append(Recording(num, "normal", 0, load))
    for kind, size, nums in FAULT_GROUPS:
        for load, num in enumerate(nums):
            if num not in EXCLUDED:
                recs.append(Recording(num, kind, size, load))
    return recs


def role(rec):
    """'train' for normal recordings at the training loads, else 'test'."""
    if rec.kind == "normal" and rec.load_hp in TRAIN_LOADS:
        return "train"
    return "test"


def load_de(root, num):
    """Return the drive-end signal of one .mat file as a 1-D float64 array."""
    path = Path(root) / f"{num}.mat"
    if not path.is_file():
        raise FileNotFoundError(f"missing CWRU file {path}")
    d = loadmat(str(path))
    # 175.mat also holds an X217_DE_time array, so use the channel named for the file.
    key = f"X{num:03d}_DE_time"
    if key not in d:
        found = [k for k in d if k.endswith("_DE_time")]
        raise KeyError(f"{path.name}: no {key} channel, found {found}")
    return np.asarray(d[key], dtype=np.float64).ravel()


def downsample(x, factor=DECIMATE):
    """FIR anti-alias decimation, zero phase, returned as float32."""
    return decimate(x, factor, ftype="fir", zero_phase=True).astype(np.float32)


def windows(x, window=WINDOW, stride=WINDOW):
    """Cut a 1-D signal into (n, window) float32 windows."""
    if len(x) < window:
        raise ValueError("signal shorter than one window")
    n = (len(x) - window) // stride + 1
    idx = np.arange(window)[None, :] + stride * np.arange(n)[:, None]
    return np.asarray(x)[idx].astype(np.float32)


def check_disjoint(train_ids, test_ids):
    """Fail if any recording is in both train and test."""
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise AssertionError(f"recordings in both train and test: {sorted(overlap)}")


def make_split(root=DEFAULT_ROOT, recordings=None, window=WINDOW):
    """Return a Split. y_test is 0 for normal and 1 for fault."""
    recs = catalogue() if recordings is None else list(recordings)
    xs_tr, rec_tr = [], []
    xs_te, y_te, rec_te, load_te, kind_te = [], [], [], [], []
    for rec in recs:
        w = windows(downsample(load_de(root, rec.num)), window, window)
        if role(rec) == "train":
            xs_tr.append(w)
            rec_tr += [rec.num] * len(w)
        else:
            xs_te.append(w)
            y_te += [0 if rec.kind == "normal" else 1] * len(w)
            rec_te += [rec.num] * len(w)
            load_te += [rec.load_hp] * len(w)
            kind_te += [rec.kind] * len(w)
    check_disjoint(rec_tr, rec_te)
    if not xs_tr or not xs_te or len(set(y_te)) < 2:
        raise ValueError("split needs train windows and both normal and fault test windows")
    return Split(
        np.concatenate(xs_tr), np.array(rec_tr, dtype=np.int64),
        np.concatenate(xs_te), np.array(y_te, dtype=np.int64),
        np.array(rec_te, dtype=np.int64), np.array(load_te, dtype=np.int64),
        np.array(kind_te),
    )


def file_hashes(root, recordings=None):
    """SHA-256 of each source .mat file, keyed by file number."""
    recs = catalogue() if recordings is None else list(recordings)
    return {r.num: hashlib.sha256((Path(root) / f"{r.num}.mat").read_bytes()).hexdigest() for r in recs}


def fingerprint(x, y):
    """SHA-256 of the arrays, to prove two runs used the same samples."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()
