import numpy as np
import pytest

FS, N = 50.0, 128


def _write_split(base, split, subjects, per_class, rng):
    sig = base / split / "Inertial Signals"
    sig.mkdir(parents=True)
    ys, subs, TOT, BODY, GYRO = [], [], [], [], []
    t = np.arange(N) / FS
    for subj in subjects:
        for label in range(1, 7):
            for _ in range(per_class):
                freq = 0.5 + 0.9 * label            # class-dependent frequency
                amp = 0.2 + 0.25 * label            # class-dependent amplitude
                phase = rng.uniform(0, 2 * np.pi, 3)
                body = np.stack([amp * np.sin(2 * np.pi * freq * t + p) + rng.normal(0, 0.05, N) for p in phase], axis=-1)
                theta = 0.5 * label                 # class-dependent gravity direction
                gravity = np.array([np.cos(theta), np.sin(theta), 0.2])
                gyro = rng.normal(0, 0.1 * label, (N, 3))
                TOT.append(body + gravity); BODY.append(body); GYRO.append(gyro); ys.append(label); subs.append(subj)
    for name, arr in (("total_acc", np.asarray(TOT)), ("body_acc", np.asarray(BODY)), ("body_gyro", np.asarray(GYRO))):
        for i, axis in enumerate("xyz"):
            np.savetxt(sig / f"{name}_{axis}_{split}.txt", arr[:, :, i])
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


@pytest.fixture
def bigger_uci_root(tmp_path):
    """Synthetic data with more volunteers, so 5-fold grouped cross-validation is possible."""
    rng = np.random.default_rng(1)
    root = tmp_path / "UCI HAR Dataset"
    _write_split(root, "train", subjects=list(range(1, 9)), per_class=4, rng=rng)
    _write_split(root, "test", subjects=[9, 10, 11], per_class=4, rng=rng)
    return tmp_path
