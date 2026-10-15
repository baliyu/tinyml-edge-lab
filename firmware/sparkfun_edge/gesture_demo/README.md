# Gesture chain for the SparkFun Edge: tested on the PC, not yet on the board

The whole path from a sensor sample to a gesture name, in plain C99 with no malloc and no hardware access: resample the sensor's
slightly uneven samples onto an exact 50 Hz grid, cut 128-sample windows, compute the features
(`../gesture_features/edge_features.c`), and ask the trained model (`gesture_model.c`) which gesture it is.

**Status: written and checked on the PC against the Python chain. Not flashed yet. The model was trained on every recording, so it
has no held-out accuracy; the on-device accuracy will come from a session recorded after the model was trained.**

## What is here
| File | What it is |
|---|---|
| `gesture_pipeline.h`, `gesture_pipeline.c` | The chain. The board calls `gp_push()` once per sensor sample. |
| `gesture_model.c`, `gesture_model.h`, `gesture_model.json` | The trained model as C, written by `python scripts/export_gesture_model.py` (default: a random forest on the raw features). The JSON lists the feature order, the gesture each number means and what the model was trained on. Generated: do not edit by hand. |

## The contract
- **One call per sample:** `gp_push(&g, ticks, x_mg, y_mg, z_mg)`. `ticks` is the free-running 32-bit counter the logger already uses (it may
  wrap); x, y, z are integer milli-g. It returns 1 when that sample completed a window, and `gp_class(&g)` is then the gesture number, an
  index into `GESTURE_CLASS_NAMES`.
- **Time is an integer.** It is kept in units of 1/(50 x tick_hz) second, so one grid step is exactly `tick_hz` units. The 50 Hz grid does
  not drift or round, for the 3 MHz internal oscillator and for the 32.768 kHz crystal alike (the recordings so far used the 3 MHz one).
- **Resampling** is linear interpolation onto the grid 0, 1/50 s, 2/50 s ... from the first sample, as `tinyml_lab.edge_gestures.resample_to_grid`.
- **Windows** are 128 samples (2.56 s), a new one every 64 samples (50 % overlap).
- **A hole longer than 12 grid steps (0.24 s)**, or time running backwards, is not interpolated over: the half-built window is dropped and
  a new one starts at the next sample (`gp_restarts()` counts them). A shorter hole is interpolated over, as the Python code does.
- **A repeated timestamp** keeps the first sample (`gp_ignored()` counts the rest).
- **`skip`** drops that many grid samples once, at start-up, not after a gap.
- State is one `gp_state` (about 1.8 KB). Nothing else is allocated.

## How it is tested
- `tests/test_gesture_pipeline_c.py` builds the chain with the host gcc and compares it with the Python chain on synthetic recordings at both
  tick rates: built as double it must equal Python to machine precision, built as float (what the board runs) closely. It also checks known
  answers that do not depend on the Python code (an exactly linear ramp must land exactly on the grid and the windows must hop by 64), the
  32-bit counter wrapping, short and long gaps, repeated and backwards timestamps, and the ARM build's size.
- The tests were also checked by breaking the C code in 15 deliberate ways; each break was caught.
- `python scripts/check_gesture_pipeline.py` does the same comparison on your own recordings, feeding them in sample by sample as the board
  would receive them.

<!-- pipeline-check:start -->
## Checked on real recordings
`python scripts/check_gesture_pipeline.py` fed 15 recordings (465 windows; sessions s1, s2, s3; timestamps from the 3 MHz internal oscillator) through the C chain sample by sample, as the board would receive them, and compared it with the Python chain on the same samples. The model is the forest on raw features (3036 nodes), trained on every session.

| Build | Windows where the gesture differs from Python | Worst feature difference (of a feature's size) | Windows with a different dominant bin |
|---|---|---|---|
| as double (proves the algorithm) | 0 of 465 | 6.5e-13 | 0 |
| as float (what the board runs) | 0 of 465 | 2.3e-05 | 0 |

Board build (arm-none-eabi-gcc, Cortex-M4F, -Os): flash 75,725 B (the chain 1,770 B, the model 73,955 B), state RAM 1,768 B, deepest stack about 3,096 B. These sizes come from the compiler; nothing has run on the board.

The model in this folder is identical to a fresh export from these recordings.

This compares the two chains on recordings the model was trained on. It shows that the C code does what the Python code does. It says nothing about accuracy on new sessions.
<!-- pipeline-check:end -->

## Not done yet
- Flashing it: a `main.c` that reads the accelerometer (the logger's code does), calls `gp_push()` and prints the gesture.
- The live accuracy: a new session, recorded after the model was trained, with the gestures done in a known order. One person, one board.
- Time per window on the chip. The sizes above come from the compiler; nothing has run on the board.
