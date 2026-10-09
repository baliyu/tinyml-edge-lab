"""Loader for the official UCI "Human Activity Recognition Using Smartphones" data.

Layout after unzipping (as published by UCI):
  UCI HAR Dataset/{train,test}/Inertial Signals/{total_acc,body_acc,body_gyro}_{x,y,z}_{train,test}.txt
        (128 samples per row, 50 Hz; total_acc = body_acc + gravity, separated by UCI with a low-pass filter)
  UCI HAR Dataset/{train,test}/y_{train,test}.txt        activity ids 1..6
  UCI HAR Dataset/{train,test}/subject_{train,test}.txt  volunteer ids

UCI's own train/test split is by volunteer (70% / 30%), so no person appears in both.
"""
from pathlib import Path
import numpy as np

FS = 50.0
WINDOW = 128
LABELS = {1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
          4: "SITTING", 5: "STANDING", 6: "LAYING"}
SIGNALS = ("total_acc", "body_acc", "body_gyro")


def find_root(path):
    """Return the folder that contains train/ and test/ (accepts the parent folder too)."""
    path = Path(path)
    for cand in [path, path / "UCI HAR Dataset"] + sorted(path.glob("*/UCI HAR Dataset")):
        if (cand / "train").is_dir() and (cand / "test").is_dir():
            return cand
    raise FileNotFoundError(f"No UCI HAR train/ and test/ folders found under {path}")


def load_signal(root, split, signal="total_acc"):
    """Windows of one inertial signal as an array (N, 128, 3) for axes x, y, z."""
    base = find_root(root) / split / "Inertial Signals"
    chans = [np.loadtxt(base / f"{signal}_{axis}_{split}.txt") for axis in "xyz"]
    return np.stack(chans, axis=-1)


def load_meta(root, split):
    """(labels, subjects) for a split."""
    base = find_root(root) / split
    y = np.loadtxt(base / f"y_{split}.txt", dtype=int)
    subjects = np.loadtxt(base / f"subject_{split}.txt", dtype=int)
    return y, subjects


def load_split(root, split, signal="total_acc"):
    """Return (windows [N, 128, 3], labels [N], subjects [N]) for split 'train' or 'test'."""
    windows = load_signal(root, split, signal)
    y, subjects = load_meta(root, split)
    if not (len(windows) == len(y) == len(subjects)):
        raise ValueError("row counts of signals, labels and subjects do not match")
    return windows, y, subjects


def assert_subject_disjoint(train_subjects, test_subjects):
    """Fail loudly if any volunteer appears in both splits (that would leak information)."""
    overlap = set(np.unique(train_subjects)) & set(np.unique(test_subjects))
    if overlap:
        raise AssertionError(f"subjects in both train and test: {sorted(overlap)}")
