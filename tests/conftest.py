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



# ---------------------------------------------------------------- synthetic CWRU-style data (NOT real data)
def _fake_cwru_signal(rng, kind, size, load, n, fs=48000.0):
    x = rng.normal(0, 0.05 * (1 + 0.1 * load), n)
    if kind != "normal":
        f_res = {"IR": 3000.0, "B": 5000.0, "OR": 7000.0}[kind]
        rate = {"IR": 160.0, "B": 100.0, "OR": 220.0}[kind]
        amp = 0.5 if size < 0.01 else 1.0
        tau = np.arange(int(0.004 * fs)) / fs
        burst = amp * np.exp(-tau / 0.0008) * np.sin(2 * np.pi * f_res * tau) * (1 + 0.1 * load)
        for start in range(0, n - len(burst), int(fs / rate)):
            x[start:start + len(burst)] += burst
    return x


@pytest.fixture
def fake_cwru(tmp_path):
    """(data_dir, manifest): 28 small .mat files named and keyed like CWRU's, with matching sizes and SHA-256 values."""
    import hashlib
    from scipy.io import savemat
    rng = np.random.default_rng(0)
    data = tmp_path / "cwru"
    data.mkdir()
    manifest, number = [], 1000
    combos = [("normal", 0.0)] + [(k, s) for k in ("IR", "B", "OR") for s in (0.007, 0.014)]
    for kind, size in combos:
        for load in range(4):
            number += 1
            x = _fake_cwru_signal(rng, kind, size, load, 72000)
            path = data / f"{number}.mat"
            savemat(path, {f"X{number:03d}_DE_time": x[:, None], f"X{number:03d}RPM": np.array([[1797 - 20 * load]])})
            cls = "normal" if kind == "normal" else f"{kind}{int(round(size * 1000)):03d}"
            manifest.append({"number": number, "cls": cls, "kind": kind, "size_in": size, "load_hp": load,
                             "rpm": 1797 - 20 * load, "bytes": path.stat().st_size,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return data, manifest
