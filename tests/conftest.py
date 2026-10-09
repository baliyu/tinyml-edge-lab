import numpy as np
import pytest

FS, N = 50.0, 128


def _write_split(base, split, subjects, per_class, rng):
    sig = base / split / "Inertial Signals"
    sig.mkdir(parents=True)
    ys, subs, X = [], [], []
    t = np.arange(N) / FS
    for subj in subjects:
        for label in range(1, 7):
            for _ in range(per_class):
                freq = 0.5 + 0.9 * label            # class-dependent frequency
                amp = 0.2 + 0.25 * label            # class-dependent amplitude
                phase = rng.uniform(0, 2 * np.pi, 3)
                w = np.stack([amp * np.sin(2 * np.pi * freq * t + p) + rng.normal(0, 0.05, N) for p in phase], axis=-1)
                X.append(w); ys.append(label); subs.append(subj)
    X = np.asarray(X)
    for i, axis in enumerate("xyz"):
        np.savetxt(sig / f"total_acc_{axis}_{split}.txt", X[:, :, i])
    np.savetxt(base / split / f"y_{split}.txt", ys, fmt="%d")
    np.savetxt(base / split / f"subject_{split}.txt", subs, fmt="%d")


@pytest.fixture
def fake_uci_root(tmp_path):
    """A tiny synthetic dataset in the real UCI layout (NOT real data; used only to test the code)."""
    rng = np.random.default_rng(0)
    root = tmp_path / "UCI HAR Dataset"
    _write_split(root, "train", subjects=[1, 2, 3], per_class=6, rng=rng)
    _write_split(root, "test", subjects=[4, 5], per_class=6, rng=rng)
    return tmp_path
