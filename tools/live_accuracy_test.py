#!/usr/bin/env python3
r"""Test the gesture model live on the SparkFun Edge: do gestures in a known order while the board predicts, then score it.

Usage (Windows PowerShell, repo root, the gesture demo running on the board, no other serial program open):
    py tools\live_accuracy_test.py --port COM4 --session live1
    py tools\live_accuracy_test.py --preview               # show the order and the timing, no board needed
    py tools\live_accuracy_test.py --port COM4 --session live2 --rounds 2      # shorter: 10 blocks, about 5 minutes

It prints what to do and counts you in, one block per gesture (3 rounds of the 5 gestures in a shuffled order by default, 20 s each, 8 s to get ready
in between). The board predicts a gesture for every window; the tool logs each prediction with the time it arrived, then scores the windows that lie
fully inside a block after a 2 s lead-in (see tinyml_lab/live_accuracy.py). It never asks for a key press while the test runs, because a pause would let the serial
buffer fill and deliver late data. Files go to data/own/edge_live/<session>_live.csv and .json and are never overwritten.

Rules that keep it honest:
  * Use a session that was NOT in the model's training data (a new sitting; the model in the board was trained on s1, s2, s3). Never add this session to training.
  * Hold the board as in the recordings (flat in your palm, header pins away from you). Do not press RST. Do not click inside the console while it runs.
  * Keep every run, even a poor one: the evaluation counts all the runs you give it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import edge_capture as ec  # noqa: E402
import edge_gesture_monitor as egm  # noqa: E402
from record_session import GESTURES  # noqa: E402
from tinyml_lab import live_accuracy as la  # noqa: E402

MODEL_FILE = Path("firmware") / "sparkfun_edge" / "gesture_demo" / "gesture_model.c"
MODEL_JSON = Path("firmware") / "sparkfun_edge" / "gesture_demo" / "gesture_model.json"
NEVER_SEEN = "the settings line was never seen"        # a hint of the monitor, not a problem of this test


def build_events(schedule, pause_s, instructions):
    """(seconds from the start, text to print) for the whole test, in time order: what to do next, the count-in, GO and STOP."""
    events, n = [], len(schedule)
    for b in schedule:
        announce = b["start_s"] - pause_s
        if b["index"] > 0:
            events.append((announce, "  STOP. Relax."))
        events.append((announce, f"\nBlock {b['index'] + 1} of {n}: {b['target']}\n  {instructions.get(b['target'], 'Do the gesture you were trained on.')}"))
        events += [(b["start_s"] - k, f"  {k} ...") for k in (3, 2, 1)]
        events.append((b["start_s"], f"  GO: {b['target']}. Keep going until STOP."))
    events.append((schedule[-1]["end_s"], "  STOP. That was the last block."))
    return sorted(events, key=lambda e: e[0])


def _text(raw):
    return raw.decode("ascii", errors="replace") if isinstance(raw, bytes) else raw


def run_test(ser, schedule, *, pause_s, instructions, clock=None, out=print, wait_s=20.0, tail_s=3.0):
    """Run the timed test on an open serial port. Returns a dict: windows, monitor, interrupted, completed_s (seconds of the schedule that were run), error.

    The schedule's clock starts when the first window arrives, so the board is known to be alive. Nothing here waits for a key.
    """
    clock = clock or time.monotonic
    ser.reset_input_buffer()
    mon, windows = egm.Monitor(), []
    begin, t0 = clock(), None
    try:
        while t0 is None:
            if clock() - begin > wait_s:
                return {"error": f"no gesture window arrived from the board in {wait_s:.0f} s: is it flashed, powered and on this port? (press RST once, then start again)",
                        "windows": [], "monitor": mon, "interrupted": False, "completed_s": 0.0}
            raw = ser.readline()
            item = mon.feed(_text(raw)) if raw else None
            if item and item[0] == "window":
                t0 = clock()
                windows.append({"t_s": 0.0, "window": item[1]["window"], "t_ticks": item[1]["t_ticks"], "predicted": item[1]["gesture"], "compute_us": item[1]["compute_us"]})
    except KeyboardInterrupt:
        return {"error": "stopped before the test started: nothing was saved", "windows": [], "monitor": mon, "interrupted": True, "completed_s": 0.0}
    events, shown, end_at, interrupted, now = build_events(schedule, pause_s, instructions), 0, schedule[-1]["end_s"] + tail_s, False, 0.0
    try:
        while True:
            now = clock() - t0
            while shown < len(events) and events[shown][0] <= now:
                out(events[shown][1])
                shown += 1
            if now >= end_at:
                break
            raw = ser.readline()
            item = mon.feed(_text(raw)) if raw else None
            if item and item[0] == "window":
                d = item[1]
                windows.append({"t_s": clock() - t0, "window": d["window"], "t_ticks": d["t_ticks"], "predicted": d["gesture"], "compute_us": d["compute_us"]})
            elif item and item[0] == "error":
                out(item[1])
    except KeyboardInterrupt:
        interrupted, now = True, clock() - t0
        out("\nstopped")
    return {"error": None, "windows": windows, "monitor": mon, "interrupted": interrupted, "completed_s": min(now, schedule[-1]["end_s"])}


def model_info(root=ROOT):
    """Which model file this repository holds (its SHA-256 and what it was trained on), so a result says which model it is about."""
    f, j = Path(root) / MODEL_FILE, Path(root) / MODEL_JSON
    if not f.is_file():
        return None
    info = {"file": MODEL_FILE.as_posix(), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "trained_on": None,
            "assumption": "the image on the board was built from this file; the tool cannot check that"}
    if j.is_file():
        info["trained_on"] = json.loads(j.read_text(encoding="utf-8")).get("trained_on")
    return info


def build_meta(result, schedule, *, session, port, baud, seed, rounds, block_s, pause_s, started_at, model):
    """Everything about a run except the windows themselves."""
    mon = result["monitor"]
    problems = [p for p in mon.problems() if NEVER_SEEN not in p]
    if mon.config and mon.config["classes"] != list(la.LABELS):
        problems.append(f"the board's gestures {mon.config['classes']} differ from the test's {list(la.LABELS)}")
    return {"session": session, "started_at": started_at, "port": port, "baud": baud, "seed": seed, "labels": list(la.LABELS), "rounds": rounds,
            "block_s": block_s, "pause_s": pause_s, "lead_in_s": la.LEAD_IN_S, "tail_guard_s": la.TAIL_GUARD_S, "schedule": schedule,
            "completed_s": result["completed_s"], "interrupted": result["interrupted"], "windows_received": len(result["windows"]),
            "stale_lines": mon.stale_lines, "stale_fragments": mon.stale_fragments, "config_seen": mon.config is not None, "board_config": mon.config,
            "monitor_problems": problems, "model": model,
            "note": "One person, one board. The gesture shown on screen is the target; each prediction is the board's own."}


def preview(schedule, out=print):
    out(f"{len(schedule)} blocks, {schedule[0]['end_s'] - schedule[0]['start_s']:.0f} s each, {schedule[0]['start_s']:.0f} s to get ready before each; "
        f"about {schedule[-1]['end_s'] / 60:.1f} minutes in all.")
    for b in schedule:
        out(f"  block {b['index'] + 1:>2}  round {b['round'] + 1}  {b['target']:<9} from {b['start_s']:>5.0f} s to {b['end_s']:>5.0f} s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", help="serial port, e.g. COM4")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--session", help="name of this test, e.g. live1 (must not be a training session such as s1, s2, s3)")
    ap.add_argument("--rounds", type=int, default=la.DEFAULT_ROUNDS, help=f"passes through all gestures (default {la.DEFAULT_ROUNDS})")
    ap.add_argument("--block-seconds", type=float, default=la.DEFAULT_BLOCK_S)
    ap.add_argument("--pause-seconds", type=float, default=la.DEFAULT_PAUSE_S)
    ap.add_argument("--seed", type=int, default=None, help="order of the gestures (default: from the clock, and saved)")
    ap.add_argument("--out-dir", default="data/own/edge_live")
    ap.add_argument("--preview", action="store_true", help="show the schedule and exit; no board needed")
    a = ap.parse_args(argv)
    seed = a.seed if a.seed is not None else int(time.time()) % 100000
    try:
        schedule = la.make_schedule(la.LABELS, a.rounds, seed, a.block_seconds, a.pause_seconds)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    if a.preview:
        preview(schedule)
        return 0
    if not a.port or not a.session:
        print("Give --port and --session (or --preview to look at the schedule).", file=sys.stderr)
        return 2
    session = ec.safe_label(a.session)
    model = model_info()
    trained = ((model or {}).get("trained_on") or {}).get("sessions", [])
    if session in trained:
        print(f"Session '{session}' was used to train the model: a test on it would prove nothing. Use a new name (and a new sitting).", file=sys.stderr)
        return 2
    csv_path, json_path = la.run_paths(a.out_dir, session)
    if csv_path.exists() or json_path.exists():
        print(f"{csv_path.name} or {json_path.name} already exists in {a.out_dir}: use another --session name (a run is never overwritten).", file=sys.stderr)
        return 2
    try:
        import serial  # pyserial
    except ImportError:
        print("pyserial is missing. Run: py -m pip install pyserial", file=sys.stderr)
        return 2
    try:
        ser = serial.Serial(a.port, a.baud, timeout=0.2)
    except Exception as e:  # noqa: BLE001
        print(f"Could not open {a.port}: {e}\nIs another program (miniterm, PuTTY) using it?", file=sys.stderr)
        return 2
    print(f"Live accuracy test '{session}': {len(schedule)} blocks of {a.block_seconds:.0f} s, about {schedule[-1]['end_s'] / 60:.1f} minutes. Order seed {seed} (saved).")
    print("Hold the board as in the recordings (flat in your palm, header pins away from you). Do not press RST. Do not click inside this window while it runs.")
    print("Each block: read what to do, wait for the count-in, do it until STOP. Ctrl+C stops early (completed blocks are kept).")
    started_at = datetime.now().isoformat(timespec="seconds")
    with ser:
        input("Press Enter to start ... ")
        result = run_test(ser, schedule, pause_s=a.pause_seconds, instructions=dict(GESTURES))
    if result["error"]:
        print(result["error"], file=sys.stderr)
        return 1
    meta = build_meta(result, schedule, session=session, port=a.port, baud=a.baud, seed=seed, rounds=a.rounds, block_s=a.block_seconds,
                      pause_s=a.pause_seconds, started_at=started_at, model=model)
    csv_path, json_path = la.write_run(a.out_dir, meta, result["windows"])
    rows, dropped, problems = la.score_run(meta, result["windows"])
    print(f"\nSaved {csv_path} and {json_path.name}: {len(result['windows'])} windows from the board, {len(rows)} counted "
          f"({dropped['outside_scored_zone']} in pauses, lead-ins or end margins, {dropped['delayed']} dropped as late).")
    if rows:
        try:
            print("\n" + la.summary_text(la.evaluate(rows, la.LABELS, seed=seed)))
        except ValueError as e:
            print(e, file=sys.stderr)
    else:
        print("No window could be counted.")
    if problems:
        print("\nThings to look at:")
        for p in problems:
            print("  -", p)
    print(f"\nNext (WSL, repo root): python scripts/evaluate_live_accuracy.py --run data/own/edge_live/{session}_live.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
