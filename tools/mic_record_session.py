#!/usr/bin/env python3
"""Record one microphone session from the SparkFun Edge: quiet, speech and knock in turn, same session name, one command.
Built with AI assistance (Claude).

    py tools\\mic_record_session.py --port COM4 --session s1

For each class it shows what to do, waits for Enter, counts down, records, and says whether the recording was clean.
If you fumbled one, type r to record that class again. Files go to data/own/mic/ and carry the session name, which the
evaluation uses to keep whole sittings apart (a model is tested on a sitting it never saw).

Rules that keep the data honest:
  * Put the board in the SAME place every time (flat on the desk, microphone side up), and speak from about 30 cm away.
  * Record at least three sessions, on different occasions (a different day, or after a long break).
  * Do not click inside the console while it records: selecting text pauses a Windows console and loses lines.
  * Knock: a light fingertip tap was almost invisible to the microphone (first trial), so the class is a firm knuckle knock.
    Some windows between knocks still look like quiet, and some knock windows clip (a few samples hit 16383).
  * Speech: keep talking the whole time (read aloud or count). Pauses between words still count as speech windows, so
    the 'speech' label is noisier than the others. Say so when you report results.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mic_capture as mc  # noqa: E402

CLASSES = (
    ("quiet", "Stay quiet. Do not touch the desk or the board. Normal room noise is fine."),
    ("speech", "Talk in your normal voice, about 30 cm from the board. Keep talking the whole time (read aloud or count)."),
    ("knock", "Knock on the desk next to the board with one knuckle, firmly but not hard enough to hurt, about 4 knocks per second, steadily the whole time."),
)
DEFAULT_SECONDS = 45


def record_one(ser, *, label, seconds, session, subject, note, port, baud, out_dir, clock=time.monotonic,
               progress=None, now=None):
    """Record one class and save it. Returns the dict from mic_capture.save_capture."""
    ser.reset_input_buffer()
    result = mc.capture(ser, seconds, clock=clock, progress=progress)
    return mc.save_capture(result, label=label, subject=subject, session=session, note=note, port=port, baud=baud,
                           out_dir=out_dir, now=now)


def run_session(ser, *, session, subject, seconds, port, baud, out_dir, note="", ask=input, say=print,
                sleep=time.sleep, clock=time.monotonic, classes=CLASSES):
    """Walk through every class. Returns {label: result of the accepted recording}."""
    done = {}
    for label, how in classes:
        while True:
            say(f"\n=== {label} ({seconds:.0f} s) ===\n{how}")
            ask("Press Enter when you are ready (the recording starts after a 3-second countdown)... ")
            for n in (3, 2, 1):
                say(f"  {n}")
                sleep(1)
            say("  recording...")
            r = record_one(ser, label=label, seconds=seconds, session=session, subject=subject, note=note, port=port,
                           baud=baud, out_dir=out_dir, clock=clock)
            if not r["ok"]:
                say("  " + r["reason"])
            else:
                c = r["counts"]
                say(f"  saved {r['windows']} windows ({r['valid']} valid), median rate {c['median_rate']:.0f} Hz, "
                    f"missing {c['gaps']}, valid windows that clipped {c['clip_windows']}")
                say("  verdict: " + ("OK" if not r["problems"] else "PROBLEMS"))
                for p in r["problems"]:
                    say("   - " + p)
            if r["ok"] and not r["problems"]:
                again = ask("Type r to record this one again, or press Enter to continue: ").strip().lower() == "r"
            else:
                again = ask("Type r to record this one again (recommended), or press Enter to keep it and continue: ").strip().lower() == "r"
            if not again:
                done[label] = r
                break
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--session", required=True, help="one sitting, e.g. s1, s2, s3")
    ap.add_argument("--subject", default="s1")
    ap.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    ap.add_argument("--note", default="")
    ap.add_argument("--out-dir", default="data/own/mic")
    a = ap.parse_args(argv)
    session = mc.safe_label(a.session)
    try:
        import serial  # pyserial
    except ImportError:
        print("pyserial is missing. Run: py -m pip install pyserial", file=sys.stderr)
        return 2
    try:
        ser = serial.Serial(a.port, a.baud, timeout=0.5)
    except Exception as e:  # noqa: BLE001
        print(f"Could not open {a.port}: {e}\nIs another program (PuTTY) using it?", file=sys.stderr)
        return 2
    print(f"Session '{session}'. Do not click inside this window while it records.")
    with ser:
        done = run_session(ser, session=session, subject=a.subject, seconds=a.seconds, port=a.port, baud=a.baud,
                           out_dir=a.out_dir, note=a.note)
    print("\nSession finished:")
    for label, r in done.items():
        print(f"  {label}: " + (f"{r['valid']} valid windows -> {r['path']}" if r["ok"] else "NOT SAVED"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
