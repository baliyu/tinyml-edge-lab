"""Pool live runs, write the live-accuracy results and README block (every number generated from the runs, none typed in)."""
from __future__ import annotations

import json
from pathlib import Path

from . import live_accuracy as la

START, END = "<!-- live-accuracy:start -->", "<!-- live-accuracy:end -->"
HEADING = "## Live accuracy on the board"


def pool_runs(runs):
    """Score several runs together. `runs` is a list of (meta, windows). Returns the report dict (what results/live_accuracy.json holds).

    Refuses (ValueError) to pool runs that cannot be pooled: the same session twice, a session the model was trained on, different
    gesture sets, or different model files.
    """
    if not runs:
        raise ValueError("no runs given")
    sessions = [m["session"] for m, _ in runs]
    if len(set(sessions)) != len(sessions):
        raise ValueError(f"the same session is given twice: {sorted(s for s in set(sessions) if sessions.count(s) > 1)}")
    labels = runs[0][0]["labels"]
    if any(m["labels"] != labels for m, _ in runs):
        raise ValueError("the runs do not use the same gestures")
    models = {(m.get("model") or {}).get("sha256") for m, _ in runs}
    if len(models) > 1:
        raise ValueError("the runs were made with different model files: scores of different models cannot be pooled")
    model = runs[0][0].get("model")
    trained = ((model or {}).get("trained_on") or {}).get("sessions", [])
    leaked = sorted(set(sessions) & set(trained))
    if leaked:
        raise ValueError(f"session(s) {leaked} were used to train the model: a live score on them would prove nothing")
    rows, dropped, summary = [], {"outside_scored_zone": 0, "unfinished_block": 0, "delayed": 0}, []
    for meta, windows in runs:
        r, d, problems = la.score_run(meta, windows)
        rows += r
        for k in dropped:
            dropped[k] += d[k]
        done = sum(1 for b in meta["schedule"] if b["end_s"] <= meta.get("completed_s", 0.0))
        summary.append({"session": meta["session"], "started_at": meta.get("started_at"), "seed": meta.get("seed"), "blocks": len(meta["schedule"]),
                        "completed_blocks": done, "windows_received": len(windows), "counted": len(r), "dropped": d, "problems": problems})
    first = runs[0][0]
    protocol = {"block_s": first["block_s"], "pause_s": first["pause_s"], "lead_in_s": first.get("lead_in_s", la.LEAD_IN_S),
                "tail_guard_s": first.get("tail_guard_s", la.TAIL_GUARD_S), "window_s": la.WINDOW_S, "labels": labels}
    return {"model": model, "protocol": protocol, "runs": summary, "dropped_total": dropped, "result": la.evaluate(rows, labels, seed=first.get("seed", 1)),
            "note": "One person, one board. Windows overlap by half; the interval resamples whole blocks."}


def offline_reference(results_dir="results"):
    """The offline leave-one-session-out score of the deployed model (the forest on raw features) from gestures.json, or None."""
    p = Path(results_dir) / "gestures.json"
    if not p.is_file():
        return None
    rep = json.loads(p.read_text(encoding="utf-8"))
    try:
        s = rep["protocols"]["leave_one_session_out"]["results"]["raw"]["forest"]
        return {"macro_f1": s["macro_f1"], "accuracy": s["accuracy"], "windows": rep["data"]["windows"], "sessions": rep["data"]["sessions"], "source": "results/gestures.json"}
    except KeyError:
        return None


def confusion_fig(result, path):
    """The live confusion matrix as a figure (rows: the gesture done, columns: what the board said)."""
    from .plots import confusion_matrix_fig
    confusion_matrix_fig(result["confusion_matrix"], result["labels"], path,
                         title=f"Live on the SparkFun Edge: {result['n_windows']} windows in {result['n_blocks']} blocks\n(each row sums to 100 % of that gesture)",
                         xlabel="gesture the board said", ylabel="gesture done")


def _f(x):
    return "n/a" if x is None else f"{x:.3f}"


def _iv(pair):
    return f"{pair[0]:.3f} to {pair[1]:.3f}"


def render_block(report, offline=None, figure="results/figures/live_confusion.png"):
    """The README section, written from the report (and the offline reference, if there is one)."""
    r, p, runs = report["result"], report["protocol"], report["runs"]
    model, ci = report.get("model"), r["ci"]
    sess = ", ".join(x["session"] for x in runs)
    trained = ((model or {}).get("trained_on") or {}).get("sessions")
    n_done = sum(x["completed_blocks"] for x in runs)
    lines = [START, HEADING,
             f"`python scripts/evaluate_live_accuracy.py` scored the model running on the board against gestures done in a known order (test session{'s' if len(runs) > 1 else ''}: {sess}; "
             f"`tools/live_accuracy_test.py`). Each block was one gesture for {p['block_s']:.0f} s, the gestures in a shuffled order, every gesture the same number of times "
             f"({n_done} blocks completed). A window counted only if all of its {p['window_s']:.2f} s lay inside a block, after a {p['lead_in_s']:.0f} s lead-in and "
             f"{p['tail_guard_s']:g} s before the block ended; the board's own prediction for it was compared with the gesture shown.",
             ""]
    if trained:
        lines += [f"The model was trained on sessions {', '.join(trained)} only; the test session{'s are' if len(runs) > 1 else ' is'} not among them "
                  f"(model file SHA-256 `{model['sha256'][:12]}`; it is assumed that the board runs an image built from that file).", ""]
    lines += [f"**{r['n_windows']} windows counted in {r['n_blocks']} blocks: accuracy {_f(r['accuracy'])}, macro-F1 {_f(r['macro_f1'])}** "
              f"(chance is {r['chance_accuracy']:.3f}).", ""]
    if ci:
        lines += [f"{int(ci['level'] * 100)} % intervals from resampling whole blocks within each gesture ({ci['n_boot']} resamples): "
                  f"accuracy {_iv(ci['accuracy'])}, macro-F1 {_iv(ci['macro_f1'])}. They are wide because there are few blocks per gesture.", ""]
    else:
        lines += [f"No interval is given: {r['ci_unavailable']}.", ""]
    lines += ["| Gesture | Windows | Recall | Precision |", "|---|---|---|---|"]
    lines += [f"| {g.replace('_', ' ')} | {v['support']} | {_f(v['recall'])} | {_f(v['precision'])} |" for g, v in r["per_class"].items() if v["support"]]
    worst = min(((g, v["recall"]) for g, v in r["per_class"].items() if v["recall"] is not None), key=lambda kv: kv[1])
    lines += ["", f"The weakest gesture is {worst[0].replace('_', ' ')} (recall {_f(worst[1])}).", "", f"![Live confusion matrix]({figure})", ""]
    if offline:
        inside = ci and ci["macro_f1"][0] <= offline["macro_f1"] <= ci["macro_f1"][1]
        lines += [f"For comparison, the same model type (forest, raw features) scored macro-F1 {_f(offline['macro_f1'])} offline when a whole session was held out "
                  f"({offline['windows']} windows, `{offline['source']}`)."
                  + (f" That figure lies {'inside' if inside else 'outside'} the live interval above." if ci else ""), ""]
    lines += ["What this does and does not show:",
              "- One person and one board. A different person, or the board held differently, may do better or worse.",
              f"- {r['n_blocks']} blocks are few, and windows within a block overlap by half, so they are not independent; that is why the interval resamples blocks.",
              "- The PC's arrival time of each line stands in for the moment its window ended; the serial link adds some milliseconds, which the lead-in and end margin are meant to absorb (not measured).",
              f"- Windows dropped: {report['dropped_total']['outside_scored_zone']} in pauses, lead-ins or end margins, {report['dropped_total']['delayed']} that arrived late and could not be placed in time, "
              f"{report['dropped_total']['unfinished_block']} in a block that was not finished."]
    for x in runs:
        for prob in x["problems"]:
            lines.append(f"- Session {x['session']}: {prob}.")
    lines.append(END)
    return "\n".join(lines)


def update_readme(path, block):
    """Put the block into the demo README: replace it if it is there, else insert it before '## Not done yet'. Keeps the file's line endings."""
    p = Path(path)
    raw = p.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    s = raw.replace("\r\n", "\n")
    if START in s and END in s:
        a, b = s.index(START), s.index(END) + len(END)
        s = s[:a] + block + s[b:]
    else:
        marker = "\n## Not done yet\n"
        if s.count(marker) != 1:
            raise SystemExit(f"STOP: cannot find the place for the section in {path}.")
        s = s.replace(marker, "\n" + block + "\n" + marker)
    p.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))
