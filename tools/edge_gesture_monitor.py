#!/usr/bin/env python3
r"""Watch the SparkFun Edge's gesture demo over a serial port.

Usage (Windows PowerShell, repo root, no other serial program open):
    py tools\edge_gesture_monitor.py --port COM4 [--seconds 60] [--log data\\own\\edge_live\\run.csv]
    py tools\edge_gesture_monitor.py --port COM4 --seconds 90 --quiet    # a dot per window and a short progress line, then the summary
    py tools\edge_gesture_monitor.py --replay saved_output.txt        # no board: replay a saved text file of the board's output

Shows every "G,window,t_ticks,gesture,compute_us" line the firmware prints (one per 64 samples of the 50 Hz grid, every 1.28 s), a running
count per gesture, and a summary at the end that points out anything wrong: lost lines, sensor overruns, a window that took
too long, a restart after a gap, an error from the firmware. When Windows opens a port it can hand over a stale buffer that repeats the
last line hundreds of times; exact repeats are dropped and counted, not shown. It needs pyserial for a real port (py -m pip install pyserial).
It does not judge accuracy: that needs the gestures to be done in a known order (a separate step).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

TRAINING_TICK_HZ = 3_000_000          # every training recording was timed by the 3 MHz oscillator
SAMPLE_PERIOD_US = 19_600             # one sensor sample at ~51 Hz: a window that takes longer than this delays the next sample
SLOW_US = 15_000                      # warn well before that

_G = re.compile(r"^G,(\d+),(\d+),([A-Za-z0-9_]+),(\d+)$")
_STATS = re.compile(r"^# stats (.*)$")
_CONFIG = re.compile(r"^# odr_hz=(\d+) tick_hz=(\d+) timer=(\S+) window=(\d+) step=(\d+) features=(\d+) classes=(\S+)$")
_HELLO = re.compile(r"^# sparkfun_edge gesture demo v(\d+), whoami 0x([0-9a-fA-F]{2})$")


def parse_line(text: str):
    """('window', {...}) | ('stats', {...}) | ('config', {...}) | ('hello', {...}) | ('error', text) | ('comment', text) | None for anything else."""
    line = text.strip()
    if not line:
        return None
    m = _G.match(line)
    if m:
        return ("window", {"window": int(m.group(1)), "t_ticks": int(m.group(2)), "gesture": m.group(3), "compute_us": int(m.group(4))})
    m = _STATS.match(line)
    if m:
        kv = dict(p.split("=", 1) for p in m.group(1).split() if "=" in p)
        if kv and all(v.isdigit() for v in kv.values()):
            return ("stats", {k: int(v) for k, v in kv.items()})
        return None
    m = _CONFIG.match(line)
    if m:
        return ("config", {"odr_hz": int(m.group(1)), "tick_hz": int(m.group(2)), "timer": m.group(3), "window": int(m.group(4)), "step": int(m.group(5)),
                           "features": int(m.group(6)), "classes": m.group(7).split(",")})
    m = _HELLO.match(line)
    if m:
        return ("hello", {"version": int(m.group(1)), "whoami": int(m.group(2), 16)})
    if line.startswith("# error"):
        return ("error", line)
    if line.startswith("#"):
        return ("comment", line)
    return None


class Monitor:
    """Collects what the firmware printed and works out what, if anything, looks wrong."""

    def __init__(self):
        self.windows, self.counts, self.config, self.stats, self.errors = [], Counter(), None, None, []
        self.bad_lines, self.skipped_windows, self.last_window = 0, 0, None
        self.stale_lines, self._in_stale, self._last_line = 0, False, None
        self.stale_fragments, self._bad_since_good, self._bad_before_last = 0, 0, 0

    def feed(self, text: str):
        item = parse_line(text)
        if item is None:
            if text.strip() and not text.strip().startswith("#"):
                if self._in_stale:                                       # a cut-off line inside the stale flood: debris of the flood, not corruption
                    self.stale_fragments += 1
                else:
                    self.bad_lines += 1
                    self._bad_since_good += 1
            return None
        kind, data = item
        if kind == "window":
            if (data["window"], data["t_ticks"]) == self._last_line:     # the same line again: stale data, never a new window
                self.stale_lines += 1
                if not self._in_stale:                                   # the cut-off lines just before the flood (and before its first copy) belonged to it
                    moved = self._bad_since_good + self._bad_before_last
                    self.bad_lines -= moved
                    self.stale_fragments += moved
                    self._bad_since_good = self._bad_before_last = 0
                self._in_stale = True
                return None
            if self._in_stale:                                           # the stale flood replaced real lines: do not count them as lost
                self.last_window, self._in_stale = None, False
            self._last_line = (data["window"], data["t_ticks"])
            self._bad_before_last, self._bad_since_good = self._bad_since_good, 0
            if self.last_window is not None and data["window"] != self.last_window + 1:
                self.skipped_windows += max(0, data["window"] - self.last_window - 1)
            self.last_window = data["window"]
            self.windows.append(data)
            self.counts[data["gesture"]] += 1
        elif kind == "config":
            self.config = data
        elif kind == "stats":
            self.stats = data
        elif kind == "error":
            self.errors.append(data)
        return item

    def problems(self) -> list[str]:
        out = []
        if self.errors:
            out.append(f"the firmware reported an error: {self.errors[0]}")
        if self.config is None:
            out.append("the settings line was never seen: reset the board (RST) with this program already open to catch it")
        elif self.config["tick_hz"] != TRAINING_TICK_HZ:
            out.append(f"the board's timer runs at {self.config['tick_hz']} Hz, not the {TRAINING_TICK_HZ} Hz of the training recordings")
        if self.skipped_windows or self.bad_lines:
            out.append(f"{self.skipped_windows} window lines and {self.bad_lines} unreadable lines were lost on the way (serial corruption or a closed port)")
        slow = max((w["compute_us"] for w in self.windows), default=0)
        if slow >= SLOW_US:
            out.append(f"one window took {slow / 1000:.1f} ms; a sensor sample arrives every {SAMPLE_PERIOD_US / 1000:.1f} ms, so samples may be missed")
        if self.stats:
            if self.stats.get("overruns"):
                out.append(f"the sensor overwrote {self.stats['overruns']} samples before they were read")
            if self.stats.get("i2c_err"):
                out.append(f"{self.stats['i2c_err']} accelerometer reads failed")
            if self.stats.get("restarts"):
                out.append(f"a gap restarted the window {self.stats['restarts']} times")
            if self.stats.get("ignored"):
                out.append(f"{self.stats['ignored']} samples repeated a timestamp and were ignored")
        return out

    def summary(self) -> dict:
        n = len(self.windows)
        cu = [w["compute_us"] for w in self.windows]
        return {"windows": n, "counts": dict(self.counts), "max_compute_us": max(cu, default=0), "mean_compute_us": (sum(cu) / n) if n else 0.0,
                "stale_lines": self.stale_lines, "stale_fragments": self.stale_fragments, "problems": self.problems()}


def show(item, started: float):
    kind, d = item
    if kind == "window":
        print(f"  window {d['window']:>4}  {d['gesture']:<10} ({d['compute_us'] / 1000:.1f} ms)")
    elif kind in ("hello", "config"):
        print("  board:", d if kind == "config" else f"gesture demo v{d['version']}, sensor id 0x{d['whoami']:02x}")
    elif kind == "stats":
        print("  stats:", " ".join(f"{k}={v}" for k, v in d.items()))
    elif kind == "error":
        print(" ", d)


def print_summary(mon: Monitor) -> None:
    s = mon.summary()
    print(f"\n{s['windows']} windows.")
    for g, c in sorted(s["counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {g:<10} {c:>4}  ({100 * c / s['windows']:.0f} %)")
    if s["stale_lines"]:
        print(f"(ignored {s['stale_lines']} repeated lines and {s['stale_fragments']} cut-off lines at the start: old data that Windows hands over when it opens the port)")
    if s["windows"]:
        print(f"time per window: mean {s['mean_compute_us'] / 1000:.1f} ms, slowest {s['max_compute_us'] / 1000:.1f} ms")
    if s["problems"]:
        print("\nThings to look at:")
        for p in s["problems"]:
            print("  -", p)
    else:
        print("\nNothing looked wrong.")


def lines_from_port(ser, seconds: float | None, clock=time.monotonic):
    end = None if seconds is None else clock() + seconds
    while end is None or clock() < end:
        raw = ser.readline()
        if raw:
            yield raw.decode("ascii", errors="replace") if isinstance(raw, bytes) else raw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", help="serial port, e.g. COM4")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=None, help="stop after this long (default: until Ctrl+C)")
    ap.add_argument("--replay", help="read the board's output from this text file instead of a serial port")
    ap.add_argument("--quiet", action="store_true", help="print a dot per window and a progress line every 10 windows instead of one line per window")
    ap.add_argument("--log", help="write every window to this CSV file (pc_time,window,t_ticks,gesture,compute_us)")
    a = ap.parse_args(argv)
    if bool(a.port) == bool(a.replay):
        print("Give exactly one of --port or --replay.", file=sys.stderr)
        return 2
    mon, started = Monitor(), time.monotonic()
    log, writer = None, None
    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        log = open(a.log, "w", newline="", encoding="utf-8")
        writer = csv.writer(log)
        writer.writerow(["pc_time", "window", "t_ticks", "gesture", "compute_us"])
    try:
        if a.replay:
            if not Path(a.replay).is_file():
                print(f"{a.replay} does not exist.", file=sys.stderr)
                return 2
            source, closer = Path(a.replay).read_text(encoding="utf-8", errors="replace").splitlines(), None
        else:
            try:
                import serial  # pyserial
            except ImportError:
                print("pyserial is missing. Run: py -m pip install pyserial", file=sys.stderr)
                return 2
            try:
                closer = serial.Serial(a.port, a.baud, timeout=0.5)
            except Exception as e:  # noqa: BLE001
                print(f"Could not open {a.port}: {e}\nIs another program (miniterm, PuTTY) using it?", file=sys.stderr)
                return 2
            closer.reset_input_buffer()
            print(f"Listening on {a.port} at {a.baud} baud. Press RST on the board to see its settings line. Ctrl+C to stop.", flush=True)
            source = lines_from_port(closer, a.seconds)
        try:
            for text in source:
                item = mon.feed(text)
                if item and a.quiet:
                    if item[0] == "window":
                        print(".", end="", flush=True)
                        if len(mon.windows) % 10 == 0:
                            print(f" {len(mon.windows)} windows: " + ", ".join(f"{g} {c}" for g, c in mon.counts.most_common()), flush=True)
                    elif item[0] in ("hello", "config", "error"):
                        print(); show(item, started)
                elif item:
                    show(item, started)
                if item:
                    if item[0] == "window" and writer:
                        d = item[1]
                        writer.writerow([datetime.now().isoformat(timespec="milliseconds"), d["window"], d["t_ticks"], d["gesture"], d["compute_us"]])
                        log.flush()
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
            if closer is not None:
                closer.close()
    finally:
        if log:
            log.close()
    print_summary(mon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
