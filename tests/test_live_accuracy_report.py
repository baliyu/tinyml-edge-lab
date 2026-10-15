import importlib.util
import json
import sys
from pathlib import Path

import pytest
from live_helpers import synthetic_run

from tinyml_lab import live_accuracy as la
from tinyml_lab import live_accuracy_report as rep

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_live_accuracy", ROOT / "scripts" / "evaluate_live_accuracy.py")
script = importlib.util.module_from_spec(spec)
sys.modules["evaluate_live_accuracy"] = script
spec.loader.exec_module(script)

README = "# Demo\n\nintro\n\n## Measured\ntext\n\n## Not done yet\n- a long run\n"


def two_runs(**kw):
    """Two finished runs of a board that is wrong in two blocks of the first run (guaranteed wrong: it names the next gesture in the list)."""
    sched = la.make_schedule(la.LABELS, 3, 1, 20.0, 8.0)
    wrong = {i: la.LABELS[(la.LABELS.index(sched[i]["target"]) + 1) % len(la.LABELS)] for i in (2, 6)}
    return [synthetic_run("live1", seed=1, wrong=wrong, **kw), synthetic_run("live2", seed=2, **kw)]


# ---------------------------------------------------------------- pooling
def test_two_runs_are_pooled_and_every_run_is_accounted_for():
    r = rep.pool_runs(two_runs())
    one = la.evaluate(sum((la.score_run(m, w)[0] for m, w in two_runs()), []), la.LABELS, n_boot=10)
    assert one["accuracy"] < 1.0
    assert r["result"]["n_windows"] == one["n_windows"] and r["result"]["n_blocks"] == 30 and r["result"]["confusion_matrix"] == one["confusion_matrix"]
    assert [x["session"] for x in r["runs"]] == ["live1", "live2"] and all(x["completed_blocks"] == 15 for x in r["runs"])
    assert sum(x["counted"] for x in r["runs"]) == r["result"]["n_windows"] and r["protocol"]["lead_in_s"] == 2.0
    assert r["dropped_total"]["outside_scored_zone"] > 0 and r["model"]["sha256"] == "ab" * 32


def test_blocks_of_different_runs_stay_different_blocks():
    r = rep.pool_runs(two_runs())
    assert len({x["block"] for x in r["result"]["per_block"]}) == 30


@pytest.mark.parametrize("make, message", [
    (lambda: [synthetic_run("live1"), synthetic_run("live1", seed=2)], "given twice"),
    (lambda: [synthetic_run("s2")], "used to train the model"),
    (lambda: [synthetic_run("live1"), synthetic_run("live2", model_sha="cd" * 32)], "different model files"),
    (lambda: [], "no runs"),
])
def test_runs_that_cannot_honestly_be_pooled_are_refused(make, message):
    with pytest.raises(ValueError, match=message):
        rep.pool_runs(make())


def test_runs_with_different_gesture_sets_are_refused():
    a, b = synthetic_run("live1"), synthetic_run("live2", seed=2)
    b[0]["labels"] = ["rest", "wave"]
    with pytest.raises(ValueError, match="same gestures"):
        rep.pool_runs([a, b])


def test_the_offline_reference_is_the_deployed_models_own_score(tmp_path):
    assert rep.offline_reference(tmp_path) is None
    (tmp_path / "gestures.json").write_text(json.dumps({"data": {"windows": 465, "sessions": ["s1", "s2", "s3"]}, "protocols": {"leave_one_session_out": {"results": {
        "raw": {"forest": {"macro_f1": 0.9178, "accuracy": 0.9183}, "tree": {"macro_f1": 0.5, "accuracy": 0.5}}, "motion": {"forest": {"macro_f1": 0.99, "accuracy": 0.99}}}}}}))
    ref = rep.offline_reference(tmp_path)
    assert ref["macro_f1"] == 0.9178 and ref["windows"] == 465


# ---------------------------------------------------------------- the README block
def test_the_block_is_written_from_the_report_with_no_typed_in_numbers():
    r = rep.pool_runs(two_runs())
    ref = {"macro_f1": 0.918, "accuracy": 0.92, "windows": 465, "sessions": ["s1", "s2", "s3"], "source": "results/gestures.json"}
    block = rep.render_block(r, ref)
    res = r["result"]
    assert block.startswith(rep.START) and block.endswith(rep.END) and rep.HEADING in block
    assert f"{res['n_windows']} windows counted in {res['n_blocks']} blocks: accuracy {res['accuracy']:.3f}, macro-F1 {res['macro_f1']:.3f}" in block
    assert f"{res['ci']['accuracy'][0]:.3f} to {res['ci']['accuracy'][1]:.3f}" in block and "test sessions: live1, live2" in block
    assert "trained on sessions s1, s2, s3 only" in block and "abababababab" in block and "macro-F1 0.918 offline" in block and "465 windows" in block
    for g, v in res["per_class"].items():
        assert f"| {g.replace('_', ' ')} | {v['support']} | {v['recall']:.3f} | {v['precision']:.3f} |" in block
    assert "One person and one board" in block and "not independent" in block and "not measured" in block
    weakest = min(((g, v["recall"]) for g, v in res["per_class"].items()), key=lambda kv: kv[1])
    assert f"The weakest gesture is {weakest[0].replace('_', ' ')} (recall {weakest[1]:.3f})" in block and "figures/live_confusion.png" in block


def test_the_block_says_whether_the_offline_score_lies_inside_the_live_interval_and_leaves_it_out_when_there_is_none():
    r = rep.pool_runs(two_runs())
    lo, hi = r["result"]["ci"]["macro_f1"]
    inside = rep.render_block(r, {"macro_f1": (lo + hi) / 2, "windows": 465, "source": "x"})
    outside = rep.render_block(r, {"macro_f1": hi + 0.05, "windows": 465, "source": "x"})
    assert "lies inside the live interval" in inside and "lies outside the live interval" in outside
    assert "offline" not in rep.render_block(r, None)


def test_without_an_interval_the_block_says_why():
    meta, windows = synthetic_run("live1", rounds=1)
    r = rep.pool_runs([(meta, windows)])
    block = rep.render_block(r)
    assert r["result"]["ci"] is None and "No interval is given" in block and "at least 2 blocks" in block


def test_problems_of_a_run_appear_in_the_block():
    meta, windows = synthetic_run("live1")
    meta["monitor_problems"] = ["the sensor overwrote 1 samples before they were read"]
    assert "Session live1: the board reported: the sensor overwrote 1 samples before they were read." in rep.render_block(rep.pool_runs([(meta, windows), synthetic_run("live2", seed=2)]))


def test_the_readme_gets_the_block_before_not_done_yet_then_the_block_is_replaced_not_added(tmp_path):
    p = tmp_path / "README.md"
    p.write_text(README)
    rep.update_readme(p, rep.START + "\nfirst\n" + rep.END)
    s = p.read_text()
    assert s.index("## Measured") < s.index(rep.START) < s.index("## Not done yet") and s.count(rep.START) == 1
    rep.update_readme(p, rep.START + "\nsecond\n" + rep.END)
    s = p.read_text()
    assert "second" in s and "first" not in s and s.count(rep.START) == 1 and s.endswith("- a long run\n")


def test_windows_line_endings_survive(tmp_path):
    p = tmp_path / "README.md"
    p.write_bytes(README.replace("\n", "\r\n").encode())
    rep.update_readme(p, rep.START + "\nx\n" + rep.END)
    raw = p.read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")


def test_a_readme_without_the_expected_place_is_not_touched(tmp_path):
    p = tmp_path / "README.md"
    p.write_text("# nothing\n")
    with pytest.raises(SystemExit):
        rep.update_readme(p, rep.START + rep.END)
    assert p.read_text() == "# nothing\n"


# ---------------------------------------------------------------- the script
def saved_runs(tmp_path, runs):
    files = []
    for meta, windows in runs:
        files += ["--run", str(la.write_run(tmp_path / "live", meta, windows)[1])]
    return files


def test_the_script_writes_results_figure_and_readme_and_a_second_run_changes_nothing(tmp_path, capsys):
    args = saved_runs(tmp_path, two_runs())
    readme = tmp_path / "README.md"
    readme.write_text(README)
    out = tmp_path / "results"
    assert script.main([*args, "--out", str(out), "--readme", str(readme)]) == 0
    first = (readme.read_text(), (out / "live_accuracy.json").read_text())
    saved = json.loads(first[1])
    assert saved["result"]["n_blocks"] == 30 and saved["offline_reference"] is None and saved["model"]["sha256"] == "ab" * 32
    assert (out / "figures" / "live_confusion.png").stat().st_size > 1000 and rep.START in first[0]
    assert "accuracy" in capsys.readouterr().out
    assert script.main([*args, "--out", str(out), "--readme", str(readme)]) == 0
    assert (readme.read_text(), (out / "live_accuracy.json").read_text()) == first


def test_the_script_can_leave_the_readme_alone(tmp_path):
    args = saved_runs(tmp_path, [synthetic_run("live1"), synthetic_run("live2", seed=2)])
    readme = tmp_path / "README.md"
    readme.write_text(README)
    assert script.main([*args, "--out", str(tmp_path / "r"), "--readme", str(readme), "--no-readme"]) == 0
    assert readme.read_text() == README and (tmp_path / "r" / "live_accuracy.json").is_file()


def test_the_script_refuses_a_training_session_and_a_missing_file(tmp_path, capsys):
    args = saved_runs(tmp_path, [synthetic_run("s1")])
    assert script.main([*args, "--out", str(tmp_path / "r"), "--no-readme"]) == 1
    assert "STOP" in capsys.readouterr().err and not (tmp_path / "r").exists()
    assert script.main(["--run", str(tmp_path / "nope.json"), "--out", str(tmp_path / "r")]) == 2


def test_the_script_uses_the_offline_score_when_the_results_folder_has_one(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    (out / "gestures.json").write_text(json.dumps({"data": {"windows": 465, "sessions": []}, "protocols": {"leave_one_session_out": {"results": {"raw": {"forest": {"macro_f1": 0.918, "accuracy": 0.92}}}}}}))
    args = saved_runs(tmp_path, two_runs())
    readme = tmp_path / "README.md"
    readme.write_text(README)
    assert script.main([*args, "--out", str(out), "--readme", str(readme)]) == 0
    assert "macro-F1 0.918 offline" in readme.read_text() and json.loads((out / "live_accuracy.json").read_text())["offline_reference"]["windows"] == 465
