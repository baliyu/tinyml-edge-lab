"""Capture one benchmark run from the Nano over USB serial.

AI-assisted: drafted with Claude, checked by the author.
Use on Windows: set PYTHONPATH=src, then
python scripts/board_capture.py --port COM6 --out results/board_int8_run1.txt
Needs pyserial. Stops at the DONE line, an ERROR line or the timeout.
Use --rerun only when the sketch has already finished a run.
"""
import argparse
import time
from pathlib import Path


def read_run(ser, timeout_s=60.0, send_r=False, clock=time.time):
    lines = []
    if send_r:
        ser.write(b"r")
    start = clock()
    while clock() - start < timeout_s:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("ascii", errors="replace").strip()
        lines.append(line)
        if line.startswith("DONE,") or line.startswith("ERROR,"):
            break
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    import serial

    with serial.Serial(args.port, 115200, timeout=0.5) as ser:
        ser.dtr = True
        lines = read_run(ser, args.timeout, args.rerun)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(len(lines), "lines written to", out)
    print("last line:", lines[-1] if lines else "(none)")
    return 0 if lines and lines[-1].startswith("DONE,") else 1
