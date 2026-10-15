#!/usr/bin/env python3
"""Record what monitor logs say about the gesture chain running on the Edge.

Usage: python scripts/record_board_run.py --run "FFT features=data/own/edge_live/second_run.csv"
                                         [--run "direct-transform features (first build)=data/own/edge_live/first_run.csv"]
                                         [--out results] [--readme firmware/sparkfun_edge/gesture_demo/README.md]
Reads the CSV files written by `tools/edge_gesture_monitor.py --log` (pc_time,window,t_ticks,gesture,compute_us), drops repeated rows
(an old monitor also logged the stale lines Windows hands over when it opens the port), and works out the time per window, the interval
between windows and the gesture counts. Writes <out>/board_run.json and the "Measured on the board" section of the demo README, from
those numbers. It measures timing on the chip, not accuracy: for that the gestures must be done in a known order.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

TICK_HZ = 3_000_000                   # the firmware times everything with the 3 MHz oscillator
WRAP = 1 << 32
WINDOW_S = 64 / 50                    # a window is 64 samples of the 50 Hz grid: 1.28 s by construction, not a measurement
SENSOR_PERIOD_MS = 19.6               # the sensor's own sample period: the recordings' headers measured 50.9 to 51.1 Hz, so about 51 Hz
MIN_WINDOWS = 10
START, END = "<!-- board-run:start -->", "<!-- board-run:end -->"
DEFAULT_README = Path(__file__).resolve().parents[1] / "firmware" / "sparkfun_edge" / "gesture_demo" / "README.md"


def read_log(path) -> list[dict]:
    """Windows from a monitor log, each (window, t_ticks) once, in file order."""
    rows, seen = [], set()
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                row = {"window": int(r["window"]), "t_ticks": int(r["t_ticks"]), "gesture": r["gesture"], "compute_us": int(r["compute_us"])}
            except (KeyError, TypeError, ValueError):
                continue
            key = (row["window"], row["t_ticks"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def summarise(rows: list[dict], label: str = "", file: str = "") -> dict:
    if len(rows) < MIN_WINDOWS:
        raise ValueError(f"only {len(rows)} windows in {file or label}: record at least {MIN_WINDOWS} (about {MIN_WINDOWS * WINDOW_S:.0f} s)")
    us = [r["compute_us"] for r in rows]
    gaps = [((b["t_ticks"] - a["t_ticks"]) % WRAP) / TICK_HZ for a, b in zip(rows, rows[1:])]
    consecutive = sum(1 for a, b in zip(rows, rows[1:]) if b["window"] == a["window"] + 1)
    return {"label": label, "file": file, "windows": len(rows), "mean_ms": statistics.fmean(us) / 1000, "median_ms": statistics.median(us) / 1000,
            "min_ms": min(us) / 1000, "max_ms": max(us) / 1000, "duration_s": sum(gaps), "median_interval_s": statistics.median(gaps),
            "windows_not_consecutive": len(rows) - 1 - consecutive,
            "gesture_counts": dict(Counter(r["gesture"] for r in rows))}


def section(runs: list[dict]) -> str:
    rows = [f"| {r['label']} | {r['windows']} | {r['mean_ms']:.1f} ms | {r['max_ms']:.1f} ms |" for r in runs]
    best = min(runs, key=lambda r: r["mean_ms"])
    return "\n".join([START, "## Measured on the board",
        "`python scripts/record_board_run.py` read the monitor's own log files (`tools/edge_gesture_monitor.py --log`). One person, one SparkFun Edge, short runs "
        "by hand: they show that the chain runs on the chip and how long a window takes, not how accurate it is.", "",
        "| Build | Windows | Mean time per window | Slowest |", "|---|---|---|---|", *rows, "",
        "Time per window is what `gp_push()` took for the sample that completed the window (resampling, features and the model), read from the chip's "
        f"3 MHz timer. A sensor sample arrives about every {SENSOR_PERIOD_MS} ms (the roughly 51 Hz the recordings measured), so the fastest build here "
        f"spends {100 * best['mean_ms'] / SENSOR_PERIOD_MS:.0f} % of one sample period on a window, and a window comes every {WINDOW_S:.2f} s.", END])


def update_readme(path: Path, text: str) -> None:
    s = path.read_text(encoding="utf-8")
    if START in s and END in s:
        a, b = s.index(START), s.index(END) + len(END)
        s = s[:a] + text + s[b:]
    else:
        marker = "\n## Not done yet\n"
        if s.count(marker) != 1:
            raise SystemExit(f"STOP: cannot find the place for the section in {path}.")
        s = s.replace(marker, "\n" + text + "\n" + marker)
    path.write_text(s, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, metavar='"LABEL=FILE"', help="a labelled monitor log; repeat for more than one")
    ap.add_argument("--out", default="results")
    ap.add_argument("--readme", default=str(DEFAULT_README))
    a = ap.parse_args(argv)
    runs = []
    for spec in a.run:
        label, sep, file = spec.partition("=")
        if not sep or not label.strip() or not file.strip():
            print(f'Each --run must look like "LABEL=FILE", got {spec!r}.', file=sys.stderr)
            return 2
        if not Path(file).is_file():
            print(f"{file} does not exist.", file=sys.stderr)
            return 2
        try:
            runs.append(summarise(read_log(file), label.strip(), file.strip()))
        except ValueError as e:
            print(e, file=sys.stderr)
            return 1
    for r in runs:
        print(f"{r['label']}: {r['windows']} windows over {r['duration_s']:.0f} s, {r['mean_ms']:.1f} ms per window on average (slowest {r['max_ms']:.1f} ms), "
              f"gestures {r['gesture_counts']}")
        if r["windows_not_consecutive"]:
            print(f"  note: {r['windows_not_consecutive']} places where the window numbers are not consecutive (lines lost on the way, or a reset)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "board_run.json").write_text(json.dumps({"tick_hz": TICK_HZ, "runs": runs,
        "note": "Timing on the chip from short runs by hand. It says nothing about accuracy: the gestures were not done in a known order."}, indent=2), encoding="utf-8")
    update_readme(Path(a.readme), section(runs))
    print(f"wrote {out / 'board_run.json'} and updated {a.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
