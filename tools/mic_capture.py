#!/usr/bin/env python3
"""Capture the SparkFun Edge microphone summary lines (firmware/sparkfun_edge/mic_logger) into a labelled CSV file.
Built with AI assistance (Claude).

The firmware prints one line per 0.25 s window: "w,mean,rms,peak,zcr,hf,clip,rate_hz", plus '#' comment lines.
This tool keeps the lines, marks which windows are trustworthy (valid=1), and says plainly what it found.

    py tools\\mic_capture.py --port COM4 --label quiet --session s1 --seconds 60

Rules for valid=0: the first START_DROP windows after a board reset (start-up junk seen in the first test), and any
window whose rate_hz is more than RATE_TOL away from the median rate of this recording (a gap or lost samples).
Nothing is deleted: every line received is saved, and the valid column says what to use.
It needs pyserial (py -m pip install pyserial). Do not click inside the console window while it records.
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

LINE_RE = re.compile(r"^(\d{1,9}),(\d{1,5}),(\d{1,5}),(\d{1,5}),(\d{1,5}),(\d{1,5}),(\d{1,5}),(\d{1,6})$")
FIELDS = ("w", "mean", "rms", "peak", "zcr", "hf", "clip", "rate_hz")
START_DROP = 2        # windows after a board reset that are not trusted (the first test showed odd lines 1 and 2)
RATE_TOL = 0.02       # a window is valid if its rate_hz is within 2 % of the recording's median
MAX_COUNT = 16383     # mean, peak and rms of 14-bit samples cannot exceed this; anything larger is a corrupted line
WINDOW_N = 2000


def parse_line(line: str):
    """('row', dict), ('meta', dict), ('comment', text) or None for garbage."""
    line = line.strip()
    if not line:
        return None
    if line.startswith("#"):
        kv = dict(re.findall(r"(\w+)=(\S+)", line))
        if line.startswith("# mic_logger") and "window" in kv:
            try:
                return ("meta", {"window": int(kv["window"]), "fs_target": int(kv.get("fs_target", 0))})
            except ValueError:
                return None
        return ("comment", line)
    m = LINE_RE.match(line)
    if not m:
        return None
    row = dict(zip(FIELDS, (int(g) for g in m.groups())))
    if max(row["mean"], row["rms"], row["peak"]) > MAX_COUNT or row["clip"] > WINDOW_N:
        return None
    return ("row", row)


def capture(ser, seconds: float, clock=time.monotonic, progress=None) -> dict:
    """Read lines for `seconds` of wall time. A repeated window number is a stale duplicate and is ignored; a number
    that goes backwards means the board restarted (counted, and the start-up windows are distrusted again)."""
    rows, meta, bad, dups, restarts, stale = [], None, 0, 0, 0, 0
    end = clock() + seconds
    last_w, last_report = None, clock()
    while clock() < end:
        raw = ser.readline()
        if not raw:
            continue
        text = raw.decode("ascii", errors="replace") if isinstance(raw, bytes) else raw
        item = parse_line(text)
        if item is None:
            if text.strip() and not text.strip().startswith("# w,"):
                if rows:
                    bad += 1
                else:
                    stale += 1   # before the first good window: old bytes Windows handed over when the port was opened
            continue
        if item[0] == "meta":
            meta = item[1]
        elif item[0] == "row":
            w = item[1]["w"]
            if last_w is not None and w == last_w:
                dups += 1
                continue
            if last_w is not None and w < last_w:
                restarts += 1
            last_w = w
            rows.append(item[1])
        if progress and clock() - last_report >= 1.0:
            progress(len(rows))
            last_report = clock()
    return {"rows": rows, "meta": meta, "bad_lines": bad, "duplicates": dups, "restarts": restarts, "stale_start": stale}


def mark_valid(rows: list[dict]) -> dict:
    """Add 'valid' (0/1) and 'why' to each row, in place. Returns counts."""
    if not rows:
        return {"median_rate": 0, "gaps": 0, "invalid": 0, "clip_windows": 0}
    med = statistics.median(r["rate_hz"] for r in rows)
    gaps = 0
    prev_w = None
    since_reset = 0
    for r in rows:
        if prev_w is None or r["w"] < prev_w:
            since_reset = 0
        since_reset += 1
        why = ""
        if since_reset <= START_DROP:
            why = "start-up"
        elif med and abs(r["rate_hz"] - med) > RATE_TOL * med:
            why = "rate"
        if prev_w is not None and r["w"] > prev_w + 1:
            gaps += r["w"] - prev_w - 1   # whole windows never arrived
            if not why:
                why = "after-gap"
        r["valid"] = 0 if why else 1
        r["why"] = why
        prev_w = r["w"]
    return {"median_rate": med, "gaps": gaps, "invalid": sum(1 for r in rows if not r["valid"]),
            "clip_windows": sum(1 for r in rows if r["valid"] and r["clip"] > 0)}


def safe_label(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    if not s:
        raise ValueError("the label must contain at least one letter or digit")
    return s


def save_capture(result: dict, *, label: str, subject: str, session: str, note: str, port: str, baud: int,
                 out_dir, now=None) -> dict:
    rows = result["rows"]
    if not rows:
        return {"ok": False, "reason": "No usable lines received. Tap RST on the Edge, check the port, and close PuTTY."}
    counts = mark_valid(rows)
    valid = sum(r["valid"] for r in rows)
    problems = []
    if result["restarts"]:
        problems.append(f"the board restarted {result['restarts']} time(s) during the recording")
    if counts["gaps"]:
        problems.append(f"{counts['gaps']} whole window(s) never arrived")
    if result["bad_lines"] > max(2, len(rows) // 100):
        problems.append(f"{result['bad_lines']} unreadable lines")
    if valid < 0.8 * len(rows):
        problems.append(f"only {valid} of {len(rows)} windows are valid")
    now = now or datetime.now()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (f"{session}_" if session else "") + f"{label}_{now.strftime('%Y%m%d_%H%M%S')}"
    path = out_dir / f"{stem}.csv"
    n = 2
    while path.exists():                      # never overwrite an earlier recording
        path = out_dir / f"{stem}_{n}.csv"
        n += 1
    head = ["# SparkFun Edge microphone summary capture (firmware mic_logger)",
            f"# label={label}", f"# subject={subject}", f"# session={session}", f"# note={note}",
            f"# captured_at={now.isoformat(timespec='seconds')}", f"# port={port} baud={baud}",
            f"# window_n={WINDOW_N} fs_target=8000 window_s=0.25",
            f"# windows={len(rows)} valid={valid} median_rate_hz={counts['median_rate']:.0f} gaps={counts['gaps']} "
            f"duplicates_ignored={result['duplicates']} bad_lines={result['bad_lines']} restarts={result['restarts']} "
            f"stale_start_lines={result['stale_start']} header_seen={int(result['meta'] is not None)}",
            "# verdict=" + ("OK" if not problems else "PROBLEMS: " + "; ".join(problems)),
            "w,mean,rms,peak,zcr,hf,clip,rate_hz,valid,why"]
    body = [",".join(str(r[k]) for k in FIELDS) + f",{r['valid']},{r['why']}" for r in rows]
    path.write_text("\n".join(head + body) + "\n", encoding="utf-8")
    return {"ok": True, "path": path, "windows": len(rows), "valid": valid, "counts": counts, "problems": problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--label", required=True, help="quiet, speech or tap")
    ap.add_argument("--subject", default="s1")
    ap.add_argument("--session", default="", help="one sitting; same name for every class recorded in it")
    ap.add_argument("--note", default="")
    ap.add_argument("--out-dir", default="data/own/mic")
    a = ap.parse_args(argv)
    label = safe_label(a.label)
    session = safe_label(a.session) if a.session else ""
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
    print(f"Recording '{label}' for {a.seconds:.0f} s on {a.port}. Do not click in this window.", flush=True)
    with ser:
        ser.reset_input_buffer()
        result = capture(ser, a.seconds, progress=lambda n: print(f"  {n} windows", end="\r", flush=True))
    print()
    r = save_capture(result, label=label, subject=a.subject, session=session, note=a.note, port=a.port, baud=a.baud,
                     out_dir=a.out_dir)
    if not r["ok"]:
        print(r["reason"], file=sys.stderr)
        return 1
    c = r["counts"]
    print(f"Saved {r['windows']} windows ({r['valid']} valid) to {r['path']}")
    print(f"Median rate {c['median_rate']:.0f} Hz; windows missing {c['gaps']}; valid windows that clipped {c['clip_windows']}")
    if result["stale_start"]:
        print(f"Note: ignored {result['stale_start']} unreadable line(s) before the first good window "
              f"(old bytes Windows handed over when the port was opened).")
    print("Verdict: " + ("OK" if not r["problems"] else "PROBLEMS"))
    for p in r["problems"]:
        print("  - " + p)
    return 0 if not r["problems"] else 3


if __name__ == "__main__":
    sys.exit(main())
