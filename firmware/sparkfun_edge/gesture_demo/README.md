# Gesture chain for the SparkFun Edge: <!-- live:title -->runs on the board, live accuracy 0.967 (1 sitting, one person, one board)<!-- /live:title -->

The whole path from a sensor sample to a gesture name, in plain C99 with no malloc and no hardware access: resample the sensor's
slightly uneven samples onto an exact 50 Hz grid, cut 128-sample windows, compute the features
(`../gesture_features/edge_features.c`), and ask the trained model (`gesture_model.c`) which gesture it is.

**Status: the chain and the firmware's main loop are checked on the PC, and the firmware has been flashed and run on a SparkFun Edge (see "Measured on the board"). <!-- live:status -->The model was trained on every recording, so it has no held-out accuracy of its own; its live accuracy was measured in live1, which is not among its training sessions: 0.967 on 180 windows in 15 blocks, one person and one board (see "Live accuracy on the board" below).<!-- /live:status -->**

## What is here
| File | What it is |
|---|---|
| `gesture_pipeline.h`, `gesture_pipeline.c` | The chain. The board calls `gp_push()` once per sensor sample. |
| `gesture_model.c`, `gesture_model.h`, `gesture_model.json` | The trained model as C, written by `python scripts/export_gesture_model.py` (default: a random forest on the raw features). The JSON lists the feature order, the gesture each number means and what the model was trained on. Generated: do not edit by hand. |
| `main.c` | The firmware for the SparkFun Edge. It sets up the accelerometer exactly as the logger does, calls `gp_push()` for every sample and prints one line per window. |
| `host_sim/` | A fake board (an accelerometer that replays a file, a timer, a UART) so that `main.c` itself runs on the PC in the tests. A test aid, not firmware. |

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

## The firmware, and how to run it
`main.c` is derived from the logger (`../imu_logger/main.c`). The accelerometer is set up identically (50 Hz, +-2 g, 12-bit), so the samples look
like the recordings the model was trained on. The timer is fixed to the internal 3 MHz oscillator, which timed every recording. It prints
`G,window,t_ticks,gesture,compute_us` every 1.28 s (64 samples of the 50 Hz grid; `compute_us` is the time the chain took for the sample that completed the window) and
a `# stats` line every 20 windows with the counts of sensor overruns, failed reads, restarts and repeated timestamps.

Build (WSL; needs the SDK from `edge_build.sh` and the model from `python scripts/export_gesture_model.py`):

    bash firmware/sparkfun_edge/scripts/edge_demo_build.sh

It derives SparkFun's Makefile at build time (that file is not copied into this repo), adds the chain's source files, builds with `-Os`, stages
`gesture_asb.bin` and prints the flash command, the same method as for the logger. Watch it (Windows PowerShell, no other serial program open):

    py tools\edge_gesture_monitor.py --port COM4 --seconds 60

The monitor shows each gesture as it arrives and points out lost lines, sensor overruns and slow windows.

## How it is tested
- `tests/test_gesture_pipeline_c.py` builds the chain with the host gcc and compares it with the Python chain on synthetic recordings at both
  tick rates: built as double it must equal Python to machine precision, built as float (what the board runs) closely. It also checks known
  answers that do not depend on the Python code (an exactly linear ramp must land exactly on the grid and the windows must hop by 64), the
  32-bit counter wrapping, short and long gaps, repeated and backwards timestamps, and the ARM build's size.
- The tests were also checked by breaking the C code in 15 deliberate ways; each break was caught.
- `python scripts/check_gesture_pipeline.py` does the same comparison on your own recordings, feeding them in sample by sample as the board
  would receive them.
- `tests/test_gesture_demo_main.py` runs the firmware's real `main.c` on the PC against the fake board in `host_sim/`: its start-up text, one line per
  window, the error counters, how the sensor and the timer are set up, and exactly what it hands to the chain. It was checked by breaking `main.c` in
  16 ways; each break was caught. It says nothing about the chip's own I2C, timer or UART. `tests/test_edge_gesture_monitor.py` tests the monitor.

<!-- pipeline-check:start -->
## Checked on real recordings
`python scripts/check_gesture_pipeline.py` fed 15 recordings (465 windows; sessions s1, s2, s3; timestamps from the 3 MHz internal oscillator) through the C chain sample by sample, as the board would receive them, and compared it with the Python chain on the same samples. The model is the forest on raw features (3036 nodes), trained on every session.

| Build | Windows where the gesture differs from Python | Worst feature difference (of a feature's size) | Windows with a different dominant bin |
|---|---|---|---|
| as double (proves the algorithm) | 0 of 465 | 6.5e-13 | 0 |
| as float (what the board runs) | 0 of 465 | 2.3e-05 | 0 |

Board build (arm-none-eabi-gcc, Cortex-M4F, -Os): flash 75,965 B (the chain 2,010 B, the model 73,955 B), state RAM 1,768 B, deepest stack about 3,648 B. These sizes come from the compiler.

The model in this folder is identical to a fresh export from these recordings.

This compares the two chains on recordings the model was trained on. It shows that the C code does what the Python code does. It says nothing about accuracy on new sessions.
<!-- pipeline-check:end -->

<!-- board-run:start -->
## Measured on the board
`python scripts/record_board_run.py` read the monitor's own log files (`tools/edge_gesture_monitor.py --log`). One person, one SparkFun Edge, short runs by hand: they show that the chain runs on the chip and how long a window takes, not how accurate it is.

| Build | Windows | Mean time per window | Slowest |
|---|---|---|---|
| FFT features | 69 | 3.8 ms | 3.9 ms |
| direct-transform features (first build) | 68 | 16.1 ms | 16.2 ms |

Time per window is what `gp_push()` took for the sample that completed the window (resampling, features and the model), read from the chip's 3 MHz timer. A sensor sample arrives about every 19.6 ms (the roughly 51 Hz the recordings measured), so the fastest build here spends 19 % of one sample period on a window, and a window comes every 1.28 s.
<!-- board-run:end -->

<!-- live-accuracy:start -->
## Live accuracy on the board
`python scripts/evaluate_live_accuracy.py` scored the model running on the board against gestures done in a known order (test session: live1; `tools/live_accuracy_test.py`). Each block was one gesture for 20 s, the gestures in a shuffled order, every gesture the same number of times (15 blocks completed). A window counted only if all of its 2.56 s lay inside a block, after a 2 s lead-in and 0.5 s before the block ended; the board's own prediction for it was compared with the gesture shown.

The model was trained on sessions s1, s2, s3 only; the test session is not among them (model file SHA-256 `bbbee3fdb422`; it is assumed that the board runs an image built from that file).

**180 windows counted in 15 blocks: accuracy 0.967, macro-F1 0.967** (chance is 0.200).

95 % intervals from resampling whole blocks within each gesture (2000 resamples): accuracy 0.922 to 1.000, macro-F1 0.921 to 1.000. They are wide because there are few blocks per gesture.

| Gesture | Windows | Recall | Precision |
|---|---|---|---|
| rest | 36 | 1.000 | 1.000 |
| shake lr | 36 | 1.000 | 0.973 |
| shake ud | 36 | 0.972 | 1.000 |
| circle | 36 | 0.889 | 0.970 |
| tilt | 36 | 0.972 | 0.897 |

The weakest gesture is circle (recall 0.889).

![Live confusion matrix](results/figures/live_confusion.png)

For comparison, the same model type (forest, raw features) scored macro-F1 0.918 offline when a whole session was held out (465 windows, `results/gestures.json`). That figure lies outside the live interval above.

What this does and does not show:
- One person and one board. A different person, or the board held differently, may do better or worse.
- 15 blocks are few, and windows within a block overlap by half, so they are not independent; that is why the interval resamples blocks.
- The PC's arrival time of each line stands in for the moment its window ended; the serial link adds some milliseconds, which the lead-in and end margin are meant to absorb (not measured).
- Windows dropped: 152 in pauses, lead-ins or end margins, 0 that arrived late and could not be placed in time, 0 in a block that was not finished.
- Session live1: the board reported: the sensor overwrote 1 samples before they were read.
<!-- live-accuracy:end -->

## Not done yet
- A long run: the runs so far lasted about a minute each, so whether sensor samples are ever missed over minutes or hours is not measured. The sizes above come from the compiler.
- <!-- live:notdone -->More live sittings: so far 1 sitting (live1), by one person on one board. Another day, another person or another board would show whether 0.967 holds; none of these has been tested.<!-- /live:notdone -->
