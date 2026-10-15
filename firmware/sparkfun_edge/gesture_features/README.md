# Gesture features in C

The desk-gesture features, as C code that can run on the SparkFun Edge: `edge_features.c` and `edge_features.h`.

**Status: written and checked on the PC against the Python features, on synthetic and constructed windows and on real
recordings (see "Checked on real recordings"). It runs on the board as part of the gesture demo (`../gesture_demo/`).**

It is a port of `tinyml_lab.features.window_features` and `tinyml_lab.edge_gestures.feature_matrix`. The Python code is the
reference: a model is trained on the Python features, so the board must compute the same numbers.

## What it computes
One window of 128 samples (2.56 s at 50 Hz), x, y, z interleaved in milli-g, into one of two feature sets:
* `ef_raw_features`: 40 values (x, y, z and the vector magnitude, as measured), the order of `feature_names_for("raw")`.
* `ef_motion_features`: 37 values (each window's mean removed first, so orientation drops out), the order of
  `feature_names_for("motion")`.

Per channel: mean, standard deviation, rms, minimum, maximum, dominant frequency (DC excluded) and the share of spectral energy
in four equal bands from 0 to 25 Hz. The spectrum comes from a small in-place radix-2 FFT (128 points, about 450 butterflies per
channel) that reads its twiddle factors from the one 128-entry cosine table; there is no FFT library. A direct transform came first and was
replaced: it took 8,192 multiply-adds per channel, which measured about 16 ms for the four channels on the Edge, against a sample every 19.6 ms.

```c
#include "edge_features.h"
static ef_real window[EF_WINDOW * 3];        /* x0,y0,z0,x1,y1,z1,... in milli-g at 50 Hz */
ef_real f[EF_MOTION_FEATURES];
ef_motion_features(window, f);
/* int gesture = gesture_tree_predict(f);       the tree from results/gesture_tree.c, trained on features in this order */
```

## What it costs on the Edge (Cortex-M4F, `-Os`, measured)
* Flash: **1,504 bytes** in all (about 990 bytes of code and the 512-byte cosine table). No RAM is held between calls.
* Stack: about 3.5 KB at the deepest point (the FFT's scratch array is on the stack).
* Dependencies: none, if built with `-fno-math-errno`, which lets the compiler use the processor's square-root instruction.
  Without that flag the link needs `sqrtf` from the maths library.

## How it was checked
`tests/test_edge_features_c.py` compiles the C file with the host compiler and compares it with Python:
* Built as **double**, it matches Python to about 1e-15 of a feature's size, for both feature sets: the algorithm is the same.
* Built as **float**, what the board runs, it differs by about 1e-5 of a feature's size on synthetic windows, and no window
  picked a different dominant frequency bin.
* A trained tree and forest gave the **same answer** on the float C features as on the Python features for every window of
  synthetic data, including windows from a session the model never saw. That is a result about synthetic data only.
* The ARM build is checked for size, stack and missing symbols. The tests were also checked by deliberately breaking the C code
  in a dozen ways, and again in eleven ways when the direct transform was replaced by the FFT; each break was caught.

To repeat the comparison on your own recordings (needs `gcc`):
```
python scripts/check_edge_features.py                 # reads data/own/edge_gestures, writes results/edge_features_check.json
```

<!-- real-check:start -->
## Checked on real recordings
`python scripts/check_edge_features.py` on 465 windows from 15 real recordings (sessions s1, s2, s3). Errors are measured against each feature's own size.

| Features | C as double vs Python | C as float vs Python | Different dominant bin (float) |
|---|---|---|---|
| raw | 1.0e-15 (x_band2) | 2.0e-05 (mag_band2) | 0 of 465 windows |
| motion | 1.1e-15 (y_band0) | 3.1e-05 (mag_band1) | 0 of 465 windows |

Windows on which a trained model answered differently on the float C features than on the Python features:

* raw, tree (depth 6): 0 of 465 training windows; 0 of 465 windows from a session it never saw
* raw, forest (100 trees): 0 of 465 training windows; 0 of 465 windows from a session it never saw
* motion, tree (depth 6): 0 of 465 training windows; 0 of 465 windows from a session it never saw
* motion, forest (100 trees): 0 of 465 training windows; 0 of 465 windows from a session it never saw

The unseen-session figures are the ones that matter.
<!-- real-check:end -->

## Two cases where the feature itself is arbitrary
* **A channel with no variation** (a perfectly constant magnitude, a stuck sensor) has only rounding dust for a spectrum. Python
  reads the dust as a spectrum of arbitrary shape, depending on how numpy happens to add the numbers up. The C code treats a
  spectrum below 1e-9 of the channel's energy as none and reports zeros. Real sensor noise (a few milli-g) is far above that.
  Python was left unchanged because changing it would touch published numbers.
* **A flat spectrum** (an impulse has equal energy in every bin): which bin is "largest" is decided by rounding, in Python as
  much as in C, so the dominant frequency of such a window can differ between the two.

## Not done yet
<!-- live:features -->Done since, in `../gesture_demo/`: the 128-sample window is filled from the sensor, the model (a 100-tree forest) is called, the firmware is flashed, and its live accuracy was measured in 1 sitting (live1): 0.967 on 180 windows, one person and one board. Still not done: more sittings, other people and other boards.<!-- /live:features -->
