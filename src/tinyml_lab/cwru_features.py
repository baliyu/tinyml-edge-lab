"""Simple window features for vibration signals: five time-domain statistics and log-spaced spectral band fractions.

Band edges are defined as fractions of the Nyquist frequency, so the same definition works at 48 kHz and at 400 Hz.
"""
import numpy as np

TIME_NAMES = ["rms", "peak", "crest", "kurtosis", "skewness"]


def feature_names(n_bands=12):
    """Names of the features batch_features returns: the time-domain names, then band0_fraction, band1_fraction, ... for n_bands bands."""
    return TIME_NAMES + [f"band{b}_fraction" for b in range(n_bands)]


def band_edges(fs, n_bands=12, lo_frac=0.004, hi_frac=0.9):
    """Edges in Hz of n_bands log-spaced bands from lo_frac to hi_frac of the Nyquist frequency fs/2: n_bands + 1 values."""
    return (fs / 2.0) * np.geomspace(lo_frac, hi_frac, n_bands + 1)


def batch_features(windows, fs, n_bands=12):
    """windows: (N, L) -> (N, 5 + n_bands)."""
    w = np.asarray(windows, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError("windows must be 2-D: (N, L)")
    n, length = w.shape
    eps = 1e-12
    rms = np.sqrt(np.mean(w ** 2, axis=1))
    peak = np.max(np.abs(w), axis=1)
    centred = w - w.mean(axis=1, keepdims=True)
    std = np.sqrt(np.mean(centred ** 2, axis=1)) + eps
    kurt = np.mean(centred ** 4, axis=1) / std ** 4
    skew = np.mean(centred ** 3, axis=1) / std ** 3
    spec = np.abs(np.fft.rfft(centred * np.hanning(length), axis=1)) ** 2
    spec[:, 0] = 0.0
    freqs = np.fft.rfftfreq(length, d=1.0 / fs)
    edges = band_edges(fs, n_bands)
    total = spec.sum(axis=1) + eps
    bands = np.stack([spec[:, (freqs >= edges[b]) & (freqs < edges[b + 1])].sum(axis=1) / total
                      for b in range(n_bands)], axis=1)
    return np.column_stack([rms, peak, peak / (rms + eps), kurt, skew, bands])
