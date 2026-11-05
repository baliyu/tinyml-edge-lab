#!/usr/bin/env python3
"""Decide, by a rule fixed before any recording, whether the laptop's power state changes the SparkFun Edge microphone's quiet level. Built with AI assistance (Claude).

    python3 tools/mic_quiet_ab.py --sessions ab1,ab2

It reads the quiet recordings of the named sessions (data/own/mic/<session>_quiet_*.csv, made by mic_capture.py) and uses the tags in each recording's note:
power=plugged or power=battery, block=<n> (one sitting, recorded in the order plugged, battery, battery, plugged) and order=<n>. For each block it compares the median rms of the plugged
recordings with that of the battery recordings, checks that two recordings of the same condition agree, and applies the rule in docs/QUIET_AB_PLAN.md. The rule is not tuned afterwards.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

# The decision rule, written in docs/QUIET_AB_PLAN.md before the first recording was made. A test keeps the numbers there and here equal.
SUPPORT_RATIO = 1.20          # plugged / battery median rms at or above this in every block: supported (if the recordings agree with themselves)
NULL_LOW, NULL_HIGH = 0.90, 1.10   # inside this range in every block: not supported
MAX_SPREAD = 0.10             # two recordings of one condition in one block may differ by at most this fraction of their mean
MIN_VALID = 280               # valid windows (0.25 s each) needed for a recording to count: 70 s of a 90 s recording
MIN_BLOCKS = 2                # one block cannot support or reject anything
FEATURES = ("mean", "rms", "peak", "zcr", "hf")
POWERS = ("plugged", "battery")


def parse_note(text: str) -> dict:
    """{'power': 'plugged', 'block': '1', ...} from a note such as 'power=plugged block=1 order=1'."""
    return dict(re.findall(r"(\w+)=(\S+)", text or ""))


def read_recording(path: Path) -> dict:
    """One recording of mic_capture.py: its header values, the tags in its note, and the median of every feature over its valid windows."""
    head, rows = {}, []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith("#"):
            key, sep, value = line[1:].strip().partition("=")
            if sep:
                head.setdefault(key.strip(), value.strip())
        else:
            data_start = i
            break
    if data_start is not None:
        for r in csv.DictReader(lines[data_start:]):
            rows.append(r)
    valid = [r for r in rows if r.get("valid") == "1"]
    medians = {f: statistics.median(float(r[f]) for r in valid) for f in FEATURES} if valid else {}
    return {"path": Path(path), "label": head.get("label", ""), "session": head.get("session", ""), "captured_at": head.get("captured_at", ""),
            "verdict": head.get("verdict", ""), "tags": parse_note(head.get("note", "")), "n_rows": len(rows), "n_valid": len(valid), "medians": medians}


def load(directory: Path, sessions: list[str]) -> list[dict]:
    recs = []
    for s in sessions:
        for p in sorted(Path(directory).glob(f"{s}_quiet_*.csv")):
            recs.append(read_recording(p))
    return sorted(recs, key=lambda r: (r["session"], r["captured_at"], str(r["path"])))


def usable(rec: dict):
    """(True, '') or (False, the reason this recording is left out)."""
    t = rec["tags"]
    if rec["label"] != "quiet":
        return False, "not a quiet recording"
    if not rec["verdict"].startswith("OK"):
        return False, "the capture tool reported problems: " + rec["verdict"]
    if rec["n_valid"] < MIN_VALID:
        return False, f"only {rec['n_valid']} valid windows (need {MIN_VALID})"
    if t.get("power") not in POWERS:
        return False, "the note has no power=plugged or power=battery tag"
    if not t.get("block", "").isdigit() or not t.get("order", "").isdigit():
        return False, "the note has no block=<n> or order=<n> tag"
    return True, ""


def group_blocks(recs: list[dict]):
    """({block number: {'plugged': [rms medians], 'battery': [rms medians]}}, [(path name, reason) for recordings left out])."""
    blocks, left_out = {}, []
    for r in recs:
        ok, why = usable(r)
        if not ok:
            left_out.append((r["path"].name, why))
            continue
        b = int(r["tags"]["block"])
        blocks.setdefault(b, {p: [] for p in POWERS})[r["tags"]["power"]].append(r["medians"]["rms"])
    return blocks, left_out


def spread(values: list[float]) -> float:
    """The largest difference between recordings of one condition, as a fraction of their mean."""
    return (max(values) - min(values)) / statistics.mean(values)


def block_stats(block: dict) -> dict:
    p, b = block["plugged"], block["battery"]
    complete = len(p) >= 2 and len(b) >= 2
    out = {"complete": complete, "plugged_rms": statistics.mean(p) if p else None, "battery_rms": statistics.mean(b) if b else None}
    if complete:
        out.update(ratio=out["plugged_rms"] / out["battery_rms"], plugged_spread=spread(p), battery_spread=spread(b))
    return out


def decide(blocks: dict):
    """(verdict, reasons) by the rule in docs/QUIET_AB_PLAN.md: supported, not supported, inconclusive, or not enough blocks."""
    stats = {n: block_stats(b) for n, b in sorted(blocks.items())}
    complete = {n: s for n, s in stats.items() if s["complete"]}
    if len(complete) < MIN_BLOCKS:
        return "not enough blocks", [f"{len(complete)} complete block(s); the rule needs {MIN_BLOCKS}, each with two plugged and two battery recordings"]
    reasons = []
    noisy = [n for n, s in complete.items() if max(s["plugged_spread"], s["battery_spread"]) > MAX_SPREAD]
    if noisy:
        return "inconclusive", [f"block {n}: recordings of one condition differ by more than {MAX_SPREAD:.0%}" for n in noisy]
    ratios = [s["ratio"] for s in complete.values()]
    if all(r >= SUPPORT_RATIO for r in ratios):
        return "supported", [f"plugged is at least {SUPPORT_RATIO:.2f} times battery in every block"]
    if all(NULL_LOW <= r <= NULL_HIGH for r in ratios):
        return "not supported", [f"plugged and battery are within {NULL_LOW:.2f} to {NULL_HIGH:.2f} of each other in every block"]
    return "inconclusive", [f"block ratios {', '.join(f'{r:.2f}' for r in ratios)} are neither all at least {SUPPORT_RATIO:.2f} nor all within {NULL_LOW:.2f} to {NULL_HIGH:.2f}"]


def report(recs: list[dict]) -> tuple[str, str]:
    """(text, verdict)."""
    blocks, left_out = group_blocks(recs)
    verdict, reasons = decide(blocks)
    lines = [f"{len(recs)} quiet recording(s) read, {len(recs) - len(left_out)} used", ""]
    lines.append("block order power     rms   peak     hf   mean  valid windows")
    for r in recs:
        if usable(r)[0]:
            m, t = r["medians"], r["tags"]
            lines.append(f"{int(t['block']):5d} {int(t['order']):5d} {t['power']:8s} {m['rms']:5.0f} {m['peak']:6.0f} {m['hf']:6.0f} {m['mean']:6.0f}  {r['n_valid']:4d}")
    for name, why in left_out:
        lines.append(f"left out: {name}: {why}")
    lines.append("")
    for n, b in sorted(blocks.items()):
        s = block_stats(b)
        if s["complete"]:
            lines.append(f"block {n}: plugged/battery rms ratio {s['ratio']:.2f} (plugged {s['plugged_rms']:.0f}, battery {s['battery_rms']:.0f}); "
                         f"spread within a condition: plugged {s['plugged_spread']:.1%}, battery {s['battery_spread']:.1%}")
        else:
            lines.append(f"block {n}: incomplete ({len(b['plugged'])} plugged, {len(b['battery'])} battery recordings)")
    lines += ["", f"VERDICT: {verdict}"] + [f"  {r}" for r in reasons]
    lines.append("This is one person, one board, one room and two sittings; the verdict is about the rule, not about every setup.")
    return "\n".join(lines), verdict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sessions", required=True, help="comma-separated session names, e.g. ab1,ab2")
    ap.add_argument("--dir", default="data/own/mic", help="folder with the recordings")
    args = ap.parse_args(argv)
    sessions = [s for s in args.sessions.split(",") if s]
    recs = load(Path(args.dir), sessions)
    if not recs:
        print(f"no quiet recordings found for {', '.join(sessions)} in {args.dir}")
        return 1
    text, _ = report(recs)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
