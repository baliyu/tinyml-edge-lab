"""Live accuracy of the gesture model running on the SparkFun Edge: the board's own predictions scored against gestures done in a known order.

How a test works: tools/live_accuracy_test.py shows a schedule of blocks (one gesture per block, every gesture the same number of times, the order
shuffled from a recorded seed) and logs every prediction the board prints with the PC time it arrived. This module builds the schedule, decides which
windows count, and scores them.

A window counts only if all of its 2.56 s lie inside a block, after a 2 s lead-in (the 2 s the training recordings trim at their start) and ending 0.5 s
before the block ends, and only if it arrived on time: a window delivered late in a burst (for example after a console was paused) cannot be placed
in time, so it is dropped and counted, never guessed. The PC's arrival time stands in for the moment the window ended; the serial link adds
some milliseconds, which the lead-in and the end margin are there to absorb (an assumption, not a measurement).

The confidence interval resamples whole blocks, inside each gesture, because windows from one block overlap and are not independent.
Standard library only, so the Windows recording tool can import it without numpy.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

TICK_HZ = 3_000_000                  # the firmware times everything with the 3 MHz oscillator
WRAP = 1 << 32
WINDOW_S = 2.56                      # 128 samples of the 50 Hz grid
STEP_S = 1.28                        # a new window every 64 samples
LEAD_IN_S = 2.0                      # at the start of a block: getting started (the training recordings trim the same 2 s)
TAIL_GUARD_S = 0.5                   # at the end of a block: room for the PC clock being a little behind the board's
TIMELY_TOLERANCE_S = 0.25            # a window whose PC spacing differs from the board's by more than this was delayed
DEFAULT_BLOCK_S = 20.0
DEFAULT_PAUSE_S = 8.0
DEFAULT_ROUNDS = 3
MIN_BLOCK_S = 10.0
LABELS = ("rest", "shake_lr", "shake_ud", "circle", "tilt")
LOG_HEADER = "t_s,window,t_ticks,predicted,compute_us"
N_BOOT = 2000
CI_LEVEL = 0.95


# ---------------------------------------------------------------- the schedule
def make_schedule(labels=LABELS, rounds=DEFAULT_ROUNDS, seed=1, block_s=DEFAULT_BLOCK_S, pause_s=DEFAULT_PAUSE_S):
    """Blocks of one gesture each: `rounds` shuffled passes through the gestures, never the same gesture twice in a row.

    Returns a list of dicts (index, round, target, start_s, end_s), in seconds from the start of the test. The same seed gives the same order.
    """
    labels = list(labels)
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("need at least two different gestures")
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if block_s < MIN_BLOCK_S:
        raise ValueError(f"a block must last at least {MIN_BLOCK_S:.0f} s to hold any scored window, got {block_s}")
    if pause_s < 4:
        raise ValueError("the pause between blocks must be at least 4 s (time to read the next gesture and get ready)")
    rng = random.Random(seed)
    order = []
    for r in range(rounds):
        perm = list(labels)
        rng.shuffle(perm)
        if order and perm[0] == order[-1][1]:
            perm[0], perm[1] = perm[1], perm[0]
        order += [(r, g) for g in perm]
    return [{"index": i, "round": r, "target": g, "start_s": pause_s + i * (block_s + pause_s), "end_s": pause_s + i * (block_s + pause_s) + block_s}
            for i, (r, g) in enumerate(order)]


def min_windows_per_block(block_s=DEFAULT_BLOCK_S, lead_in_s=LEAD_IN_S, tail_guard_s=TAIL_GUARD_S):
    """The fewest windows that always fit in a block's scored zone, wherever the 1.28 s window grid happens to fall."""
    return max(0, int((block_s - lead_in_s - tail_guard_s - WINDOW_S) / STEP_S))


# ---------------------------------------------------------------- which windows count
def timely_flags(windows, tolerance_s=TIMELY_TOLERANCE_S):
    """For each window (dicts with t_s, the PC arrival time, and t_ticks), whether it arrived on time.

    A window is on time when the PC time since the previous window matches the board's own time between the two (from t_ticks, wrap-safe).
    A burst of windows delivered together after a stall fails this. The first window has nothing to compare with and is not counted.
    """
    flags = []
    for i, w in enumerate(windows):
        if i == 0:
            flags.append(False)
            continue
        p = windows[i - 1]
        board_gap = ((w["t_ticks"] - p["t_ticks"]) % WRAP) / TICK_HZ
        flags.append(abs((w["t_s"] - p["t_s"]) - board_gap) <= tolerance_s)
    return flags


def assign_windows(windows, blocks, completed_s=None, lead_in_s=LEAD_IN_S, tail_guard_s=TAIL_GUARD_S, tag=""):
    """Decide which windows count. Returns (rows, dropped).

    rows: one dict per counted window (block, target, predicted, window, t_s); `tag` is put in front of the block number so runs can be pooled.
    dropped: counts of windows that did not count and why: outside_scored_zone (in a pause, a lead-in or an end margin), unfinished_block
    (the test was stopped inside that block) and delayed (it would have counted, but arrived late and cannot be placed in time).
    """
    flags = timely_flags(windows)
    rows, dropped = [], {"outside_scored_zone": 0, "unfinished_block": 0, "delayed": 0}
    for w, on_time in zip(windows, flags):
        end, start = w["t_s"], w["t_s"] - WINDOW_S
        hit = next((b for b in blocks if start >= b["start_s"] + lead_in_s and end <= b["end_s"] - tail_guard_s), None)
        if hit is None:
            dropped["outside_scored_zone"] += 1
        elif completed_s is not None and hit["end_s"] > completed_s:
            dropped["unfinished_block"] += 1
        elif not on_time:
            dropped["delayed"] += 1
        else:
            rows.append({"block": f"{tag}{hit['index']}", "target": hit["target"], "predicted": w["predicted"], "window": w["window"], "t_s": w["t_s"]})
    return rows, dropped


# ---------------------------------------------------------------- scoring
def _metrics(cm):
    """Accuracy, macro-F1 and per-class recall / precision / F1 from a confusion matrix (rows: true, columns: predicted)."""
    n = sum(map(sum, cm))
    k = len(cm)
    support = [sum(cm[i]) for i in range(k)]
    predicted = [sum(cm[i][j] for i in range(k)) for j in range(k)]
    tp = [cm[i][i] for i in range(k)]
    per = []
    for i in range(k):
        per.append({"support": support[i], "recall": tp[i] / support[i] if support[i] else None,
                    "precision": tp[i] / predicted[i] if predicted[i] else None,
                    "f1": 2 * tp[i] / (support[i] + predicted[i]) if (support[i] + predicted[i]) else None})
    seen = [p["f1"] for p in per if p["support"]]
    return {"accuracy": sum(tp) / n if n else None, "macro_f1": sum(seen) / len(seen) if seen else None, "per_class": per}


def _percentile(sorted_values, q):
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _block_bootstrap(counts_by_block, labels, seed, n_boot, level):
    """Resample whole blocks within each gesture. Returns (interval dict, None) or (None, reason it could not be computed)."""
    strata = {}
    for target, counts in counts_by_block.values():
        strata.setdefault(target, []).append(counts)
    thin = sorted(labels[t] for t, v in strata.items() if len(v) < 2)
    if thin:
        return None, "needs at least 2 blocks of every gesture that was tested (only one for " + ", ".join(thin) + ")"
    k = len(labels)
    rng = random.Random(seed)
    accs, f1s = [], []
    for _ in range(n_boot):
        cm = [[0] * k for _ in range(k)]
        for t, blocks in strata.items():
            for _ in blocks:
                c = blocks[rng.randrange(len(blocks))]
                for j in range(k):
                    cm[t][j] += c[j]
        m = _metrics(cm)
        accs.append(m["accuracy"])
        f1s.append(m["macro_f1"])
    accs.sort()
    f1s.sort()
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    return {"level": level, "method": "resampling whole blocks within each gesture", "n_boot": n_boot, "seed": seed,
            "accuracy": [_percentile(accs, lo_q), _percentile(accs, hi_q)],
            "macro_f1": [_percentile(f1s, lo_q), _percentile(f1s, hi_q)]}, None


def evaluate(rows, labels=LABELS, seed=1, n_boot=N_BOOT):
    """Score the counted windows: confusion matrix, accuracy, macro-F1, per-gesture recall and precision, per-block results and an interval.

    rows come from assign_windows. Raises ValueError if there are none, or if a gesture name is not in `labels`. The interval resamples
    blocks, so it is None (with a reason in `ci_unavailable`) unless every tested gesture has at least two blocks.
    """
    labels = list(labels)
    index = {g: i for i, g in enumerate(labels)}
    if not rows:
        raise ValueError("no windows were scored: nothing to evaluate")
    odd = sorted({r[key] for r in rows for key in ("target", "predicted")} - set(labels))
    if odd:
        raise ValueError(f"gesture name(s) {odd} are not among the test's gestures {labels}: was the board flashed with a different model?")
    k = len(labels)
    cm = [[0] * k for _ in range(k)]
    blocks = {}
    for r in rows:
        cm[index[r["target"]]][index[r["predicted"]]] += 1
        t, counts = blocks.setdefault(r["block"], (index[r["target"]], [0] * k))
        counts[index[r["predicted"]]] += 1
    m = _metrics(cm)
    n = sum(map(sum, cm))
    ci, why = _block_bootstrap(blocks, labels, seed, n_boot, CI_LEVEL)
    per_block = [{"block": b, "target": labels[t], "windows": sum(c), "correct": c[t]} for b, (t, c) in blocks.items()]
    return {"labels": labels, "n_windows": n, "n_blocks": len(blocks), "confusion_matrix": cm, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "chance_accuracy": max(sum(r) for r in cm) / n, "per_class": {g: m["per_class"][i] for i, g in enumerate(labels)},
            "per_block": per_block, "ci": ci, "ci_unavailable": why}


# ---------------------------------------------------------------- run files
def run_paths(out_dir, session):
    """(csv path, json path) of a run: <out_dir>/<session>_live.csv and .json."""
    out = Path(out_dir)
    return out / f"{session}_live.csv", out / f"{session}_live.json"


def write_run(out_dir, meta, windows):
    """Save a run (the windows as CSV, everything else as JSON). Refuses to overwrite: every attempt is kept, none is replaced by a better one."""
    csv_path, json_path = run_paths(out_dir, meta["session"])
    if csv_path.exists() or json_path.exists():
        raise FileExistsError(f"{csv_path.name} or {json_path.name} already exists in {out_dir}: use another --session name (a run is never overwritten)")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write(LOG_HEADER + "\n")
        w = csv.writer(f, lineterminator="\n")
        for x in windows:
            w.writerow([f"{x['t_s']:.3f}", x["window"], x["t_ticks"], x["predicted"], x["compute_us"]])
    json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def read_run(json_path):
    """Load a run saved by write_run: (meta, windows). Unreadable CSV rows are skipped."""
    json_path = Path(json_path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    windows = []
    with open(json_path.with_suffix(".csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                windows.append({"t_s": float(r["t_s"]), "window": int(r["window"]), "t_ticks": int(r["t_ticks"]), "predicted": r["predicted"],
                                "compute_us": int(r["compute_us"])})
            except (KeyError, TypeError, ValueError):
                continue
    return meta, windows


def score_run(meta, windows):
    """Place one run's windows in its blocks. Returns (rows, dropped, problems); problems are things that make the run less trustworthy."""
    tag = f"{meta['session']}#"
    lead, tail = meta.get("lead_in_s", LEAD_IN_S), meta.get("tail_guard_s", TAIL_GUARD_S)
    completed = meta.get("completed_s")
    rows, dropped = assign_windows(windows, meta["schedule"], completed, lead, tail, tag=tag)
    problems = []
    if meta.get("interrupted"):
        done = sum(1 for b in meta["schedule"] if completed is not None and b["end_s"] <= completed)
        problems.append(f"the test was stopped early: {done} of {len(meta['schedule'])} blocks were completed; only those count")
    block_s = meta["schedule"][0]["end_s"] - meta["schedule"][0]["start_s"]
    need = min_windows_per_block(block_s, lead, tail)
    got = {}
    for r in rows:
        got[r["block"]] = got.get(r["block"], 0) + 1
    for b in meta["schedule"]:
        if completed is not None and b["end_s"] > completed:
            continue
        n = got.get(f"{tag}{b['index']}", 0)
        if n < need:
            problems.append(f"block {b['index'] + 1} ({b['target']}) has {n} counted windows, fewer than the {need} that always fit: lines were lost or delayed")
    if dropped["delayed"]:
        problems.append(f"{dropped['delayed']} windows that would have counted arrived late and were dropped")
    problems += [f"the board reported: {p}" for p in meta.get("monitor_problems", [])]
    return rows, dropped, problems


def summary_text(res):
    """The result as plain text for a terminal (ASCII only, so a Windows console prints it)."""
    f = lambda x: "n/a" if x is None else f"{x:.3f}"
    ci = res["ci"]
    lines = [f"{res['n_windows']} windows counted in {res['n_blocks']} blocks (always guessing the commonest gesture would score {res['chance_accuracy']:.3f}).",
             f"accuracy {f(res['accuracy'])}" + (f"  (95 % interval {ci['accuracy'][0]:.3f} to {ci['accuracy'][1]:.3f})" if ci else ""),
             f"macro-F1 {f(res['macro_f1'])}" + (f"  (95 % interval {ci['macro_f1'][0]:.3f} to {ci['macro_f1'][1]:.3f})" if ci else "")]
    if not ci:
        lines.append(f"no interval: {res['ci_unavailable']}")
    lines += ["", "recall per gesture: " + ", ".join(f"{g} {f(v['recall'])} (of {v['support']})" for g, v in res["per_class"].items() if v["support"]), "",
              "rows are the gesture done, columns what the board said:"]
    w = max(len(g) for g in res["labels"]) + 2
    lines.append(" " * w + "".join(f"{g[:8]:>9}" for g in res["labels"]))
    lines += [f"{g:<{w}}" + "".join(f"{c:>9}" for c in row) for g, row in zip(res["labels"], res["confusion_matrix"])]
    return "\n".join(lines)
