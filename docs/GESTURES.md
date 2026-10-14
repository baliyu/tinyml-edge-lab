# Desk gestures with the SparkFun Edge: design notes

Everything here runs on a PC. Three real recording sessions (12, 13 and 14 October 2026; one person, one board) have been recorded and
evaluated: the results are in the README and in `results/gestures.md`. The synthetic data described below is used only for
tests and dry runs; it proves the code, not that gestures can be recognised from this sensor.

## Why gestures
The Edge is tethered to the PC by the serial adapter, so it cannot record walking or stairs. It can record what a hand does
with a board at a desk. Human-activity data (walking, sitting, stairs) needs a battery-powered logger: that is the Nano 33 BLE
with a microSD card on the roadmap.

## The gestures
| Name | What to do |
|---|---|
| `rest` | Lay the board flat on the table and do not touch it |
| `shake_lr` | Hold the board flat in your hand and shake it left and right, about 3 shakes per second |
| `shake_ud` | The same, up and down |
| `circle` | Slow circles in the air, about one per second |
| `tilt` | Tilt forward and back slowly, one cycle about every 5 seconds |

Other labels work too (any name the recording tool accepts), but each label must appear in at least two sessions.

## How to record (Windows PowerShell, repo root, no serial terminal open)
```
py tools\record_session.py --port COM4 --session s1
```
It walks through the five gestures, 45 s each, counts down, tells you whether each recording was clean, and lets you redo a
fumbled one (type `r`). Do not click inside the console while it records.

Rules that keep the data honest:
1. **Hold the board the same way every time** (for example flat in your palm, header pins pointing away from you).
2. **Record at least three sessions on different occasions**: another day, or at least after a break and picking the board up
   again. A session is one sitting. The evaluation holds a whole session out, so sessions recorded back to back in one
   sitting would pretend to be independent when they are not.
3. Do not cherry-pick: keep every recording that the tool did not flag. Redo a gesture only when you made a mistake doing it.

## How the data is prepared
* Each recording is resampled onto an exact 50 Hz grid from its real timestamps (the accelerometer ran at about 51.2 Hz for a
  configured 50 Hz, with about a millisecond of timing jitter). 2 s are trimmed at both ends (getting started, stopping).
* Windows of 128 samples (2.56 s) every 64 samples, as for the UCI data.
* Features: the repo's window features (mean, std, rms, min, max, dominant frequency, four spectral band fractions) on x, y,
  z and the vector magnitude.
  * `raw` keeps them as measured, including the window means, which carry the board's orientation.
  * `motion` removes each window's mean first, and the magnitude is the size of what is left, so orientation drops out to
    first order. If a model does better on `raw` than on `motion` across sessions, suspect that it learned how the board was
    held instead of what the hand did.

## How it is evaluated
1. **Random split of windows: leaky.** Windows cut from one recording are neighbours, and the sitting's orientation, speed and
   noise leak across the split. Reported only to expose the problem.
2. **Leave one session out: the honest test.** Train on all other sessions, test on the one held out, for each session in turn.
   Per-session accuracy is reported, so an average cannot hide a bad session.
Settings are fixed in advance (tree depth 6, forest of 100 trees) and not tuned. Always compare with the chance level and with
"always answer the commonest gesture", which the results file reports.

## What the result cannot tell you
* All recordings come from one person and one board. Another person, another way of holding it, or another board is a
  different test, and this one says little about it.
* Windows overlap by half, so neighbouring windows are not independent, and the number of independent sessions is small. A
  difference of a few points between models should not be read as a difference.

## Dry run on synthetic data
```
python scripts/make_synthetic_gestures.py
python scripts/run_gestures.py --data data/synthetic/edge_gestures --out results/synthetic
```
Synthetic files carry `# synthetic=1`. The loader refuses to mix them with real recordings, the README writer refuses their
results, and `results/synthetic/` and `data/synthetic/` are ignored by git.

## Running a model on the board (partly done)
`python scripts/run_gestures.py --export-c` writes `results/gesture_tree.c` (float thresholds, checked against scikit-learn)
and `results/gesture_model.json` (feature order and the gesture each number means). That tree is trained on every session, so
it has no held-out score of its own.

The features exist in C: `firmware/sparkfun_edge/gesture_features/` (about 1.3 KB of flash), checked against the Python
features by `tests/test_edge_features_c.py` on synthetic and constructed windows. `python scripts/check_edge_features.py`
repeats the comparison on your own recordings, including how often a trained model answers differently on the C features.
Still missing: filling a 128-sample window from the sensor on the board, calling the tree, flashing it, and the on-device
accuracy check on a held-out session.
