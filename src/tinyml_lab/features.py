"""Window features that are cheap enough to compute on a microcontroller.

The same features can be computed on a PC (here) and on a device (C, CMSIS-DSP),
so a model trained on the PC sees the same inputs on the board.
"""
import numpy as np


def feature_names(channels, n_bands=4):
    base = ["mean", "std", "rms", "min", "max", "dom_freq"]
    base += [f"band{b}" for b in range(n_bands)]
    return [f"{ch}_{name}" for ch in channels for name in base]


def window_features(window, fs, n_bands=4):
    """Features of one window.

    window: array (n_samples, n_channels); fs: sampling rate in Hz.
    Per channel: mean, std, rms, min, max, dominant frequency (DC excluded) and the
    fraction of spectral energy in `n_bands` equal-width bands from 0 to fs/2.
    Returns a 1-D float array of length n_channels * (6 + n_bands).
    """
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 2:
        raise ValueError("window must be (n_samples, n_channels)")
    n = window.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    out = []
    for c in range(window.shape[1]):
        x = window[:, c]
        mean = x.mean()
        xc = x - mean
        spec = np.abs(np.fft.rfft(xc)) ** 2
        spec[0] = 0.0  # drop DC
        total = spec.sum()
        dom = freqs[int(np.argmax(spec))] if total > 0 else 0.0
        edges = np.linspace(0, len(spec), n_bands + 1).astype(int)
        bands = [(spec[edges[b]:edges[b + 1]].sum() / total) if total > 0 else 0.0
                 for b in range(n_bands)]
        out += [mean, x.std(), np.sqrt(np.mean(x ** 2)), x.min(), x.max(), dom] + bands
    return np.asarray(out, dtype=np.float64)


def batch_features(windows, fs, n_bands=4):
    """windows: (N, n_samples, n_channels) -> (N, n_features)."""
    return np.vstack([window_features(w, fs, n_bands) for w in windows])
