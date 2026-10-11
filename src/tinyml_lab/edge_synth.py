"""SYNTHETIC desk-gesture recordings in the capture-file format. NOT real data.

Used only to prove the pipeline works before real sessions exist, and as test input. Every file carries
'# synthetic=1'; the loader refuses to mix them with real recordings, and the README writer refuses to publish
results computed from them.

What differs between synthetic sessions, because it is what differs between real ones: the way the board sits in the
hand (rotation), how hard and how fast the gesture is made, and the noise level.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .edge_gestures import GESTURES

GRAVITY_MG = 1042.0


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def gesture_motion(label: str, t: np.ndarray, rng, amp: float, fscale: float) -> np.ndarray:
    """Motion acceleration (mg) in the hand's frame, shape (len(t), 3). Gravity is added separately."""
    m = np.zeros((len(t), 3))
    phase = rng.uniform(0, 2 * np.pi)
    if label == "rest":
        return m
    if label in ("shake_lr", "shake_ud"):
        f = 3.0 * fscale * rng.uniform(0.9, 1.1)
        a = 700.0 * amp * rng.uniform(0.85, 1.15)
        m[:, 0 if label == "shake_lr" else 2] = a * np.sin(2 * np.pi * f * t + phase)
        return m
    if label == "circle":
        f = 1.0 * fscale * rng.uniform(0.8, 1.3)
        a = 350.0 * amp * rng.uniform(0.7, 1.3)
        m[:, 0] = a * np.sin(2 * np.pi * f * t + phase)
        m[:, 1] = a * np.cos(2 * np.pi * f * t + phase)
        return m
    if label == "tilt":
        theta = np.deg2rad(35.0 * amp * rng.uniform(0.8, 1.2)) * np.sin(2 * np.pi * 0.2 * fscale * t + phase)
        g = np.array([0.0, 0.0, GRAVITY_MG])
        return np.stack([_rot_x(a) @ g - g for a in theta])
    raise ValueError(f"no synthetic model for the gesture '{label}'")


def write_synthetic_gestures(out_dir, sessions: int = 3, seconds: float = 45.0, seed: int = 0, labels=GESTURES,
                             rate_hz: float = 51.22, orientation_sd_deg: float = 20.0,
                             recording_tilt_sd_deg: float = 0.0) -> list[Path]:
    """Write one capture-format file per (session, gesture). Returns the paths.

    orientation_sd_deg: how much the whole sitting's board orientation varies between sessions.
    recording_tilt_sd_deg: extra tilt that differs for every single recording. With a large value, how the board happened to
    be held identifies the recording (a shortcut that gravity-direction features can exploit and motion features cannot).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for si in range(1, sessions + 1):
        session = f"s{si}"
        rot = _rot_y(np.deg2rad(rng.normal(0, orientation_sd_deg))) @ _rot_x(np.deg2rad(rng.normal(0, orientation_sd_deg)))
        amp, fscale, noise_sd = rng.uniform(0.8, 1.2), rng.uniform(0.9, 1.1), rng.uniform(3.0, 6.0)
        for label in labels:
            n = int(seconds * rate_hz)
            t = np.arange(n) / rate_hz + rng.uniform(-0.0007, 0.0007, n)   # real timestamps jitter by about a millisecond
            motion = gesture_motion(label, t, rng, amp, fscale)
            tilt = (_rot_y(np.deg2rad(rng.normal(0, recording_tilt_sd_deg))) @ _rot_x(np.deg2rad(rng.normal(0, recording_tilt_sd_deg)))
                    if recording_tilt_sd_deg > 0 else np.eye(3))
            a = (np.array([0.0, 0.0, GRAVITY_MG]) + motion) @ (rot @ tilt).T + rng.normal(0, noise_sd, (n, 3))
            a = np.clip(np.rint(a), -2000, 2000).astype(int)
            t = t - t[0]
            lines = ["# SYNTHETIC desk-gesture recording (NOT real data), made by tinyml_lab.edge_synth",
                     "# synthetic=1", f"# label={label}", "# subject=synthetic", f"# session={session}", "# note=synthetic",
                     "# odr_hz=50 fs_g=2 mode=hr12 units=mg tick_hz=32768 timer=xtal32k",
                     f"# measured_rate_hz={rate_hz:.3f} samples={n} duration_s={t[-1]:.3f} gaps=0 missing_est=0 bad_lines=0",
                     "# verdict=OK", "t_s,x_mg,y_mg,z_mg"]
            lines += [f"{tt:.6f},{x},{y},{z}" for tt, (x, y, z) in zip(t, a)]
            path = out_dir / f"{session}_{label}_synthetic.csv"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            paths.append(path)
    return paths
