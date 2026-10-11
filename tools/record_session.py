#!/usr/bin/env python3
"""Record one desk-gesture session from the SparkFun Edge: every gesture in turn, same session name, one command.

    py tools\\record_session.py --port COM4 --session s1

For each gesture it shows what to do, waits for Enter, counts down, records, and tells you whether the recording was
clean. If you fumbled one, type r to record that gesture again. Files go to data/own/edge_gestures/ and carry the
session name, which the evaluation uses to keep whole sittings apart (a model is tested on a sitting it never saw).

Rules that keep the data honest:
  * Hold the board the SAME way every time (for example flat in your palm, header pins pointing away from you).
  * Record at least three sessions, on different occasions (different day or at least after a break and re-picking up the board).
  * Do not click inside the console while it records: selecting text pauses a Windows console and loses samples.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edge_capture as ec  # noqa: E402

# Keep in step with tinyml_lab.edge_gestures.GESTURES (a test checks that they match).
GESTURES = (
    ("rest", "Lay the board flat on the table and do not touch it."),
    ("shake_lr", "Hold the board flat in your hand and shake it LEFT and RIGHT, about 3 shakes per second."),
    ("shake_ud", "Hold the board flat in your hand and shake it UP and DOWN, about 3 shakes per second."),
    ("circle", "Hold the board flat and move it in slow circles in the air, about one circle per second."),
    ("tilt", "Hold the board and tilt it slowly forward and back, one slow cycle about every 5 seconds."),
)
DEFAULT_SECONDS = 45


def record_one(ser, *, label, seconds, session, subject, note, port, baud, out_dir, clock=time.monotonic,
               progress=None, now=None):
    """Record one gesture and save it. Returns the dict from edge_capture.save_capture."""
    ser.reset_input_buffer()
    result = ec.capture(ser, seconds, clock=clock, progress=progress)
    return ec.save_capture(result, label=label, subject=subject, session=session, note=note, port=port, baud=baud,
                           out_dir=out_dir, now=now)


def run_session(ser, *, session, subject, labels, seconds, note, port, baud, out_dir, input_fn=input,
                sleep_fn=time.sleep, out=print, clock=time.monotonic, progress=None, now_fn=None):
    """Record every label in turn. Returns {label: save_capture result} for the recordings that were kept."""
    help_text = dict(GESTURES)
    kept = {}
    for label in labels:
        while True:
            out(f"\n=== {label} ===")
            out(help_text.get(label, "Do the gesture you want to label with this name."))
            input_fn(f"Press Enter when you are ready to record '{label}' for {seconds:.0f} s ... ")
            for k in (3, 2, 1):
                out(f"  starting in {k} ...")
                sleep_fn(1)
            out("  RECORDING. Do not click in this window.")
            r = record_one(ser, label=label, seconds=seconds, session=session, subject=subject, note=note, port=port,
                           baud=baud, out_dir=out_dir, clock=clock, progress=progress, now=now_fn() if now_fn else None)
            if not r["ok"]:
                out("  " + r["reason"])
                again = input_fn("  Nothing was saved. Press Enter to try again, or type s to skip this gesture: ")
                if again.strip().lower() == "s":
                    break
                continue
            sm = r["summary"]
            out(f"  saved {sm['n']} samples; measured {sm['rate_hz']:.2f} Hz; gaps {sm['gaps']}; "
                f"verdict {'OK' if not r['problems'] else 'PROBLEMS'}")
            for p in r["problems"]:
                out("    - " + p)
            answer = input_fn("  Press Enter to keep it, or type r to record this gesture again: ")
            if answer.strip().lower() == "r":
                Path(r["path"]).unlink(missing_ok=True)
                Path(r["resampled_path"]).unlink(missing_ok=True)
                continue
            kept[label] = r
            break
    bad = [lab for lab, r in kept.items() if r["problems"]]
    out(f"\nSession '{session}': {len(kept)} of {len(labels)} gestures kept; "
        f"{len(bad)} with problems" + (f" ({', '.join(bad)})" if bad else "") + ".")
    return kept


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", required=True, help="serial port, e.g. COM4")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--session", required=True, help="name of this sitting, e.g. s1, s2, s3")
    ap.add_argument("--subject", default="me", help="who is holding the board (default me)")
    ap.add_argument("--seconds", type=float, default=DEFAULT_SECONDS, help=f"seconds per gesture (default {DEFAULT_SECONDS})")
    ap.add_argument("--labels", default=",".join(g for g, _ in GESTURES), help="comma-separated gestures to record")
    ap.add_argument("--note", default="", help="free text, e.g. how the board was held")
    ap.add_argument("--out-dir", default="data/own/edge_gestures")
    a = ap.parse_args(argv)
    labels = [ec.safe_label(x) for x in a.labels.split(",") if x.strip()]
    session = ec.safe_label(a.session)
    try:
        import serial  # pyserial
    except ImportError:
        print("pyserial is missing. Run: py -m pip install pyserial", file=sys.stderr)
        return 2
    try:
        ser = serial.Serial(a.port, a.baud, timeout=0.5)
    except Exception as e:  # noqa: BLE001
        print(f"Could not open {a.port}: {e}\nIs another program (miniterm, PuTTY) using it?", file=sys.stderr)
        return 2
    print(f"Session '{session}': {len(labels)} gestures x {a.seconds:.0f} s on {a.port}. Hold the board the same way every time.")
    with ser:
        run_session(ser, session=session, subject=a.subject, labels=labels, seconds=a.seconds, note=a.note, port=a.port,
                    baud=a.baud, out_dir=a.out_dir, progress=lambda n: print(f"    {n} samples", end="\r", flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
