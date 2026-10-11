#!/usr/bin/env python3
"""Capture the SparkFun Edge's CSV accelerometer stream from a serial port into a labelled CSV file.

The firmware (firmware/sparkfun_edge/imu_logger/main.c) prints one line per sample, "t_ticks,x_mg,y_mg,z_mg", plus
'#' comment lines with the settings and, every 10 seconds, statistics including the sample rate it measured itself.
This tool keeps the samples, unwraps the 32-bit tick counter, measures the rate from the timestamps, counts gaps, and
says plainly whether the capture is trustworthy.

Run it with Windows Python (the Edge is on a Windows COM port):
    py tools\\edge_capture.py --port COM4 --label walking --seconds 60
It needs pyserial (py -m pip install pyserial). Do not click inside the console window while it runs: selecting text
pauses a Windows console and the adapter's buffer can overflow, which loses samples.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

WRAP = 1 << 32
SAMPLE_RE = re.compile(r"^(\d{1,10}),(-?\d{1,5}),(-?\d{1,5}),(-?\d{1,5})$")
KV_RE = re.compile(r"(\w+)=(\S+)")
MAX_ABS_MG = 4096          # the sensor is set to +-2 g (about 2048 mg); anything far outside is a corrupted line


def parse_line(line: str):
    """Return ('sample', ticks, x, y, z), ('meta', dict), ('stats', dict), ('comment', text) or None for garbage."""
    line = line.strip()
    if not line:
        return None
    if line.startswith("#"):
        kv = dict(KV_RE.findall(line))
        if line.startswith("# stats") and {"samples", "rate_hz", "tick_hz"} <= kv.keys():
            try:
                return ("stats", {"samples": int(kv["samples"]), "overruns": int(kv.get("overruns", 0)),
                                  "i2c_err": int(kv.get("i2c_err", 0)), "rate_hz": float(kv["rate_hz"]),
                                  "tick_hz": int(kv["tick_hz"]), "timer": kv.get("timer", "")})
            except ValueError:
                return None
        if "odr_hz" in kv and "tick_hz" in kv:
            try:
                return ("meta", {"odr_hz": int(kv["odr_hz"]), "tick_hz": int(kv["tick_hz"]), "timer": kv.get("timer", "")})
            except ValueError:
                return None
        return ("comment", line)
    m = SAMPLE_RE.match(line)
    if not m:
        return None
    ticks, x, y, z = (int(g) for g in m.groups())
    if ticks >= WRAP or max(abs(x), abs(y), abs(z)) > MAX_ABS_MG:
        return None
    return ("sample", ticks, x, y, z)


def unwrap_ticks(ticks: list[int]) -> list[int]:
    """Make a 32-bit wrapping counter monotonic. Samples are far apart compared with the wrap, so a step is the
    difference modulo 2**32."""
    out, total, prev = [], 0, None
    for t in ticks:
        if prev is not None:
            total += (t - prev) % WRAP
        out.append(total)
        prev = t
    return out


def summarise(unwrapped: list[int], tick_hz: int, odr_hz: int) -> dict:
    """Rate, spacing and gaps measured from the timestamps."""
    n = len(unwrapped)
    if n < 2:
        return {"n": n, "duration_s": 0.0, "rate_hz": 0.0, "max_dt_s": 0.0, "min_dt_s": 0.0, "gaps": 0, "missing_est": 0}
    dts = [(b - a) / tick_hz for a, b in zip(unwrapped, unwrapped[1:])]
    duration = (unwrapped[-1] - unwrapped[0]) / tick_hz
    nominal = 1.0 / odr_hz
    gap_dts = [d for d in dts if d > 1.5 * nominal]
    return {"n": n, "duration_s": duration, "rate_hz": (n - 1) / duration if duration > 0 else 0.0,
            "max_dt_s": max(dts), "min_dt_s": min(dts), "gaps": len(gap_dts),
            "missing_est": int(sum(round(d / nominal) - 1 for d in gap_dts))}


def change_during_capture(stats: list[dict], key: str):
    """How much a cumulative firmware counter grew between the first and last statistics line of this capture, or None
    when fewer than two statistics lines arrived (the counter is cumulative since power-up, so one line cannot say)."""
    if len(stats) < 2:
        return None
    return stats[-1][key] - stats[0][key]


def rate_note(summary: dict, odr_hz: int) -> str:
    """Information, not a fault: this sensor's output rate comes from an internal oscillator with no published accuracy."""
    if summary["n"] < 2 or abs(summary["rate_hz"] - odr_hz) <= 0.005 * odr_hz:
        return ""
    return (f"the sensor ran at {summary['rate_hz']:.2f} Hz, not exactly {odr_hz} Hz (its output rate comes from an internal "
            f"oscillator with no published accuracy); the saved timestamps are real, so resample to {odr_hz} Hz before "
            f"comparing with other {odr_hz} Hz data")


def verdict(summary: dict, odr_hz: int, stats: list[dict], bad_lines: int, tolerance: float = 0.05) -> list[str]:
    """Plain-language problems; an empty list means nothing suspicious was found."""
    problems = []
    if summary["n"] < 2:
        return ["fewer than two samples were captured"]
    if abs(summary["rate_hz"] - odr_hz) > tolerance * odr_hz:
        problems.append(f"measured rate {summary['rate_hz']:.2f} Hz differs from the configured {odr_hz} Hz by more than {tolerance:.0%}")
    if summary["gaps"]:
        problems.append(f"{summary['gaps']} gap(s) in the timestamps, about {summary['missing_est']} sample(s) missing")
    new_overruns = change_during_capture(stats, "overruns")
    if new_overruns:
        problems.append(f"the sensor reported {new_overruns} new overrun(s) during the capture: the firmware read samples too late")
    new_i2c = change_during_capture(stats, "i2c_err")
    if new_i2c:
        problems.append(f"the firmware counted {new_i2c} new I2C error(s) during the capture")
    if bad_lines > max(2, summary["n"] // 200):
        problems.append(f"{bad_lines} unreadable lines (serial corruption or dropped bytes)")
    return problems


def resample_uniform(t_s: list[float], values: list[tuple], hz: float):
    """Linear interpolation onto an exact grid (0, 1/hz, 2/hz, ...) within the recorded time span."""
    if len(t_s) < 2:
        return [], []
    n = int(t_s[-1] * hz) + 1
    grid, out, j = [], [], 0
    for k in range(n):
        t = k / hz
        while j < len(t_s) - 2 and t_s[j + 1] < t:
            j += 1
        t0, t1 = t_s[j], t_s[j + 1]
        w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        grid.append(t)
        out.append(tuple(a + w * (b - a) for a, b in zip(values[j], values[j + 1])))
    return grid, out


def safe_label(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    if not s:
        raise ValueError("the label must contain at least one letter or digit")
    return s


def capture(ser, seconds: float, clock=time.monotonic, progress=None) -> dict:
    """Read lines from `ser` (anything with readline()) for `seconds` of wall time."""
    samples, stats, meta, bad = [], [], None, 0
    end = clock() + seconds
    last_report = clock()
    while clock() < end:
        raw = ser.readline()
        if not raw:
            continue
        text = raw.decode("ascii", errors="replace") if isinstance(raw, bytes) else raw
        item = parse_line(text)
        if item is None:
            stripped = text.strip()
            if stripped and not stripped.startswith("t_ticks"):      # the CSV header is expected, anything else is not
                bad += 1
            continue
        kind = item[0]
        if kind == "sample":
            samples.append(item[1:])
        elif kind == "meta":
            meta = item[1]
        elif kind == "stats":
            stats.append(item[1])
        if progress and clock() - last_report >= 1.0:
            progress(len(samples))
            last_report = clock()
    return {"samples": samples, "stats": stats, "meta": meta, "bad_lines": bad}


def write_csv(path: Path, result: dict, label: str, subject: str, note: str, port: str, baud: int, odr_hz: int,
              tick_hz: int, timer: str, summary: dict, problems: list[str]) -> None:
    ticks = unwrap_ticks([s[0] for s in result["samples"]])
    t0 = ticks[0]
    lines = ["# SparkFun Edge LIS2DH12 capture",
             f"# label={label}", f"# subject={subject}", f"# note={note}",
             f"# captured_at={datetime.now().isoformat(timespec='seconds')}", f"# port={port} baud={baud}",
             f"# odr_hz={odr_hz} fs_g=2 mode=hr12 units=mg tick_hz={tick_hz} timer={timer}",
             f"# measured_rate_hz={summary['rate_hz']:.3f} samples={summary['n']} duration_s={summary['duration_s']:.3f} "
             f"gaps={summary['gaps']} missing_est={summary['missing_est']} bad_lines={result['bad_lines']}",
             "# verdict=" + ("OK" if not problems else "PROBLEMS: " + "; ".join(problems)),
             "# rate_note=" + (rate_note(summary, odr_hz) or "none"),
             "t_s,x_mg,y_mg,z_mg"]
    for t, (_, x, y, z) in zip(ticks, result["samples"]):
        lines.append(f"{(t - t0) / tick_hz:.6f},{x},{y},{z}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_resampled(path: Path, result: dict, tick_hz: int, hz: int, label: str, subject: str) -> int:
    ticks = unwrap_ticks([x[0] for x in result["samples"]])
    t = [(v - ticks[0]) / tick_hz for v in ticks]
    grid, out = resample_uniform(t, [x[1:] for x in result["samples"]], hz)
    lines = [f"# resampled to exactly {hz} Hz by linear interpolation from the real timestamps", f"# label={label}", f"# subject={subject}",
             "t_s,x_mg,y_mg,z_mg"] + [f"{g:.6f},{v[0]:.1f},{v[1]:.1f},{v[2]:.1f}" for g, v in zip(grid, out)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(grid)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", required=True, help="serial port, e.g. COM4")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=60.0, help="how long to record (default 60)")
    ap.add_argument("--label", required=True, help="what you are doing, e.g. walking, sitting, standing, laying")
    ap.add_argument("--subject", default="s1", help="who is wearing it (default s1)")
    ap.add_argument("--note", default="", help="free text, e.g. where the board is fixed")
    ap.add_argument("--out-dir", default="data/own/edge")
    a = ap.parse_args(argv)
    label = safe_label(a.label)
    if a.seconds < 12:
        print("Note: shorter than 12 s, so the firmware's first statistics line may not arrive; that is only needed "
              "when the settings line was missed.", file=sys.stderr)
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
    print(f"Recording '{label}' for {a.seconds:.0f} s on {a.port}. Do not click in this window.", flush=True)
    with ser:
        ser.reset_input_buffer()
        result = capture(ser, a.seconds, progress=lambda n: print(f"  {n} samples", end="\r", flush=True))
    print()
    info = result["meta"] or (({"odr_hz": 0, "tick_hz": result["stats"][-1]["tick_hz"], "timer": result["stats"][-1]["timer"]})
                              if result["stats"] else None)
    if not result["samples"] or info is None:
        print("No usable data: no samples, or no settings line seen. Tap RST on the Edge just before starting, or record "
              "for at least 12 s so a statistics line arrives.", file=sys.stderr)
        return 1
    odr_hz = info["odr_hz"] or 50
    unwrapped = unwrap_ticks([s[0] for s in result["samples"]])
    summary = summarise(unwrapped, info["tick_hz"], odr_hz)
    problems = verdict(summary, odr_hz, result["stats"], result["bad_lines"])
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    write_csv(path, result, label, a.subject, a.note, a.port, a.baud, odr_hz, info["tick_hz"], info["timer"], summary, problems)
    n_rs = write_resampled(path.with_name(path.stem + f"_{odr_hz}hz.csv"), result, info["tick_hz"], odr_hz, label, a.subject)
    print(f"Saved {summary['n']} samples ({summary['duration_s']:.1f} s) to {path}")
    print(f"Also saved {n_rs} samples resampled to exactly {odr_hz} Hz: {path.stem}_{odr_hz}hz.csv")
    print(f"Measured rate {summary['rate_hz']:.3f} Hz (configured {odr_hz} Hz); gaps {summary['gaps']}; bad lines {result['bad_lines']}")
    note = rate_note(summary, odr_hz)
    if note:
        print("Note: " + note)
    print("Verdict: " + ("OK" if not problems else "PROBLEMS"))
    for p in problems:
        print("  - " + p)
    return 0 if not problems else 3


if __name__ == "__main__":
    sys.exit(main())
