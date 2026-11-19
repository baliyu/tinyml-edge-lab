"""Parse and analyse the Nano's benchmark output.

AI-assisted: drafted with Claude, checked by the author.
Lines: START,<model>,<count> then S,<i>,<label>,<pred>,<cycles> then
DONE,<model>,<arena_used>,<count>,<correct>,<model_bytes>.
"""
import hashlib
import json
import statistics
from pathlib import Path


def parse_log(text):
    start = None
    done = None
    samples = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("START,"):
            _, model, count = line.split(",")
            start = (model, int(count))
        elif line.startswith("S,"):
            _, i, label, pred, cycles = line.split(",")
            samples.append((int(i), int(label), int(pred), int(cycles)))
        elif line.startswith("DONE,"):
            _, model, arena, count, correct, model_bytes = line.split(",")
            done = {
                "model": model,
                "arena": int(arena),
                "count": int(count),
                "correct": int(correct),
                "model_bytes": int(model_bytes),
            }
        elif line.startswith("ERROR,"):
            raise ValueError("board reported " + line)
    if start is None or done is None:
        raise ValueError("log is incomplete")
    if len(samples) != start[1] or done["count"] != start[1]:
        raise ValueError("sample count does not match the header")
    if [s[0] for s in samples] != list(range(start[1])):
        raise ValueError("sample numbers are not 0 to count-1 in order")
    if sum(1 for s in samples if s[1] == s[2]) != done["correct"]:
        raise ValueError("board correct count does not match its own lines")
    return {
        "model": start[0],
        "count": start[1],
        "arena_used_bytes": done["arena"],
        "model_bytes": done["model_bytes"],
        "correct": done["correct"],
        "samples": samples,
    }


def summarise_cycles(parsed):
    cycles = [s[3] for s in parsed["samples"]]
    return {
        "min": min(cycles),
        "median": statistics.median(cycles),
        "mean": round(statistics.mean(cycles), 1),
        "max": max(cycles),
        "first": cycles[0],
    }


def compare(parsed, pc_pred, y_test):
    labels = [s[1] for s in parsed["samples"]]
    if labels != list(y_test):
        raise ValueError("labels on the board differ from the PC test labels")
    board_pred = [s[2] for s in parsed["samples"]]
    agree = sum(1 for b, p in zip(board_pred, pc_pred) if b == p)
    return {
        "agree_with_pc": agree,
        "disagree_with_pc": parsed["count"] - agree,
        "board_accuracy": parsed["correct"] / parsed["count"],
    }


def analyse_log(log_path, npz_path, out_json):
    import numpy as np

    text = Path(log_path).read_text()
    parsed = parse_log(text)
    data = np.load(npz_path)
    result = {
        "log_file": Path(log_path).name,
        "log_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "model": parsed["model"],
        "count": parsed["count"],
        "arena_used_bytes": parsed["arena_used_bytes"],
        "model_bytes": parsed["model_bytes"],
        "cycles": summarise_cycles(parsed),
        "comparison_to": parsed["model"],
    }
    result.update(compare(parsed, data[parsed["model"]].tolist(), data["y_test"].tolist()))
    Path(out_json).write_text(json.dumps(result, indent=2) + "\n")
    return result


def cli(argv, root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    log = Path(argv[1])
    out = root / "results" / (log.stem + ".json")
    result = analyse_log(log, root / "results" / "model1_pc_predictions.npz", out)
    cyc = result["cycles"]
    print("model", result["model"], "samples", result["count"])
    print("board accuracy", round(result["board_accuracy"], 4))
    print("agree with PC", result["agree_with_pc"], "of", result["count"])
    print("cycles min/median/max", cyc["min"], cyc["median"], cyc["max"])
    print("arena used bytes", result["arena_used_bytes"])
    print("wrote", out)
    return 0
