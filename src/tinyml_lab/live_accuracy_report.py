"""Pool live runs, write the live-accuracy results and README block (every number generated from the runs, none typed in)."""
from __future__ import annotations

import json
import re
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


# ---------------------------------------------------------------- the other places that said "accuracy not yet measured"
DEMO = "firmware/sparkfun_edge/gesture_demo/README.md"
MODEL_JSON = "firmware/sparkfun_edge/gesture_demo/gesture_model.json"
# key: (file under the repository root, regular expression for the original sentence that the first run replaces)
CLAIM_PLACES = {
    "title": (DEMO, r"runs on the board, accuracy not yet measured"),
    "status": (DEMO, r"The model was trained on every\s+recording, so it has no held-out accuracy; the on-device accuracy will come from a session recorded after the model was trained\."),
    "notdone": (DEMO, r"The live accuracy: a new session, recorded after the model was trained, with the gestures done in a known order\. One person, one board\."),
    "features": ("firmware/sparkfun_edge/gesture_features/README.md",
                 r"Filling the 128-sample window from the sensor on the board, calling the tree, flashing it, and the on-device accuracy check on a\s+held-out session\."),
    "board": ("firmware/sparkfun_edge/README.md", r"No machine learning has run on this board yet\."),
    "roadmap": ("README.md", r"Check on-device accuracy against the PC on a held-out set\."),
}


def _join(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def claim_texts(report, model=None):
    """The sentences that replace the old 'accuracy not yet measured' wording, written from the report ({key: text}). `model` is gesture_model.json, if known."""
    r, sessions = report["result"], [x["session"] for x in report["runs"]]
    k, acc, sess = len(sessions), f"{report['result']['accuracy']:.3f}", _join(sessions)
    sit = f"{k} sitting" + ("" if k == 1 else "s")
    kind = f"a {model['trees']}-tree {model['model']}" if model and model.get("trees") and model.get("model") else "a model"
    return {
        "title": f"runs on the board, live accuracy {acc} ({sit}, one person, one board)",
        "status": (f"The model was trained on every recording, so it has no held-out accuracy of its own; its live accuracy was measured in {sess}, "
                   f"which {'is' if k == 1 else 'are'} not among its training sessions: {acc} on {r['n_windows']} windows in {r['n_blocks']} blocks, one person and one board "
                   f"(see \"Live accuracy on the board\" below)."),
        "notdone": (f"More live sittings: so far {sit} ({sess}), by one person on one board. Another day, another person or another board would show "
                    f"whether {acc} holds; none of these has been tested."),
        "features": (f"Done since, in `../gesture_demo/`: the 128-sample window is filled from the sensor, the model ({kind}) is called, the firmware is flashed, "
                     f"and its live accuracy was measured in {sit} ({sess}): {acc} on {r['n_windows']} windows, one person and one board. "
                     "Still not done: more sittings, other people and other boards."),
        "board": (f"A gesture model ({kind}) has since run on this board, see `gesture_demo/README.md`: its live accuracy was {acc} in {sit}, "
                  "one person and one board."),
        "roadmap": (f"On-device accuracy: measured live in {sit} (one person, one board): {acc} over {r['n_windows']} windows. More sittings, other people and "
                    "other boards are still to do; see `firmware/sparkfun_edge/gesture_demo/README.md`."),
    }


def _swap(s, key, original, inner):
    """Put `inner` between <!-- live:key --> markers: replace the marked text if it is there, else the original sentence (which must occur exactly once)."""
    op, cl = f"<!-- live:{key} -->", f"<!-- /live:{key} -->"
    new = op + inner + cl
    region = re.compile(re.escape(op) + r".*?" + re.escape(cl), re.S)
    if region.search(s):
        if len(region.findall(s)) != 1:
            raise ValueError(f"the marker for '{key}' occurs more than once")
        return region.sub(lambda m: new, s)
    if len(re.findall(original, s)) != 1:
        raise ValueError(f"cannot find the one sentence to replace for '{key}'")
    return re.sub(original, lambda m: new, s)


def update_claims(root, report):
    """Rewrite the six places that said the on-device accuracy was not measured, from the report. All or nothing: if any place cannot be found, nothing is written (ValueError)."""
    root = Path(root)
    mj = root / MODEL_JSON
    texts = claim_texts(report, json.loads(mj.read_text(encoding="utf-8")) if mj.is_file() else None)
    files = {}
    for key, (rel, original) in CLAIM_PLACES.items():
        if rel not in files:
            raw = (root / rel).read_bytes().decode("utf-8")
            files[rel] = ("\r\n" in raw, raw.replace("\r\n", "\n"))
        crlf, s = files[rel]
        try:
            files[rel] = (crlf, _swap(s, key, original, texts[key]))
        except ValueError as e:
            raise ValueError(f"{rel}: {e}") from None
    for rel, (crlf, s) in files.items():
        (root / rel).write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))
