import importlib.util
import json
import sys
import time as real_time
import types
from pathlib import Path

import pytest
from live_helpers import Clock, FakeBoard, expected_scored, performed

from tinyml_lab import live_accuracy as la

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("live_accuracy_test_tool", ROOT / "tools" / "live_accuracy_test.py")
lat = importlib.util.module_from_spec(spec)
sys.modules["live_accuracy_test_tool"] = lat
spec.loader.exec_module(lat)

INSTR = {g: f"how to do {g}" for g in la.LABELS}


def run(predict_for, schedule=None, **board_kw):
    """Run the tool's test loop against a fake board. Returns (result, printed lines, schedule, board)."""
    schedule = schedule or la.make_schedule(seed=2)
    clock = Clock()
    board = FakeBoard(clock, predict_for(schedule), **board_kw)
    shown = []
    result = lat.run_test(board, schedule, pause_s=8.0, instructions=INSTR, clock=clock, out=shown.append)
    return result, shown, schedule, board


def scored(result, schedule, seed=2, tail=None):
    meta = lat.build_meta(result, schedule, session="live1", port="COM4", baud=115200, seed=seed, rounds=3, block_s=20.0, pause_s=8.0,
                          started_at="2026-10-15T12:00:00", model=None)
    return meta, la.score_run(meta, result["windows"])


# ---------------------------------------------------------------- the timeline
def test_the_events_come_in_time_order_with_a_count_in_and_a_go_for_every_block():
    s = la.make_schedule(seed=2)
    ev = lat.build_events(s, 8.0, INSTR)
    assert [t for t, _ in ev] == sorted(t for t, _ in ev)
    for b in s:
        texts = [(t, x) for t, x in ev if b["start_s"] - 3 <= t <= b["start_s"]]
        assert [x.strip() for _, x in texts] == ["3 ...", "2 ...", "1 ...", f"GO: {b['target']}. Keep going until STOP."]
        assert [t for t, _ in texts] == [b["start_s"] - 3, b["start_s"] - 2, b["start_s"] - 1, b["start_s"]]
        assert any(f"Block {b['index'] + 1} of 15: {b['target']}" in x and f"how to do {b['target']}" in x and t == b["start_s"] - 8.0 for t, x in ev)
    assert ev[-1] == (s[-1]["end_s"], "  STOP. That was the last block.")
    assert sum("STOP. Relax." in x for _, x in ev) == 14


def test_every_gesture_has_a_real_instruction_text():
    from record_session import GESTURES
    assert [g for g, _ in GESTURES] == list(la.LABELS) and all(len(t) > 20 for _, t in GESTURES)


# ---------------------------------------------------------------- a whole test against a fake board
def test_a_perfect_board_is_scored_perfectly_on_exactly_the_windows_that_fit():
    result, shown, schedule, board = run(performed)
    assert result["error"] is None and not result["interrupted"] and result["completed_s"] == schedule[-1]["end_s"]
    meta, (rows, dropped, problems) = scored(result, schedule)
    res = la.evaluate(rows, la.LABELS, n_boot=100)
    exp = expected_scored(schedule)
    assert res["n_windows"] == sum(exp.values()) and res["accuracy"] == 1.0 and problems == [] and dropped["delayed"] == 0
    assert {int(x["block"].split("#")[1]): x["windows"] for x in res["per_block"]} == exp
    assert board.resets >= 1                                                                # the serial buffer was cleared before the test began
    assert [x for x in shown if x.startswith("  GO")] == [f"  GO: {b['target']}. Keep going until STOP." for b in schedule]


def test_the_tick_counter_wrapping_during_the_test_does_no_harm():
    """The fake board's counter wraps after about a second and a half of windows (ticks0 is just under 2**32); a wrapped window must not look late."""
    result, _, schedule, _ = run(performed)
    ticks = [w["t_ticks"] for w in result["windows"]]
    assert any(b < a for a, b in zip(ticks, ticks[1:]))
    assert scored(result, schedule)[1][2] == []


def test_a_board_that_gets_one_gesture_wrong_shows_it_in_that_gesture_only():
    def predict_for(schedule):
        good = performed(schedule)
        circle = [b for b in schedule if b["target"] == "circle"]
        wrong = circle[1]                                                                    # the second circle block: the board says tilt
        return lambda t: "tilt" if wrong["start_s"] <= t < wrong["end_s"] else good(t)
    result, _, schedule, _ = run(predict_for)
    _, (rows, _, _) = scored(result, schedule)
    res = la.evaluate(rows, la.LABELS, n_boot=100)
    wrong = [b for b in schedule if b["target"] == "circle"][1]
    exp = expected_scored(schedule)
    cm = dict(zip(la.LABELS, res["confusion_matrix"]))
    assert cm["circle"][la.LABELS.index("tilt")] == exp[wrong["index"]] and cm["circle"][la.LABELS.index("circle")] == sum(exp[b["index"]] for b in schedule if b["target"] == "circle") - exp[wrong["index"]]
    assert all(cm[g][la.LABELS.index(g)] == sum(cm[g]) for g in la.LABELS if g != "circle")
    assert res["per_class"]["circle"]["recall"] == pytest.approx(2 / 3, abs=0.03)


def test_a_stall_on_the_pc_drops_the_delayed_windows_instead_of_putting_them_in_the_wrong_block():
    """The PC stops reading from 40 s to 70 s (block 2 runs 36 to 56 s, block 3 from 64 s). Everything due meanwhile arrives together at 70 s, inside block 3's zone,
    carrying block 2's gesture. Placed by arrival time those windows would be wrong answers; the test must drop them."""
    result, _, schedule, _ = run(performed, stall=(40.0, 70.0))
    _, (rows, dropped, problems) = scored(result, schedule)
    assert dropped["delayed"] > 0 and all(r["predicted"] == r["target"] for r in rows)
    assert any("arrived late" in p for p in problems)
    assert any("block 2 (" in p and "fewer than the 11" in p for p in problems)                  # its whole scored zone fell inside the stall


def test_a_test_stopped_with_ctrl_c_keeps_the_blocks_it_finished():
    result, shown, schedule, _ = run(performed, interrupt_at=schedule_time(5, 12.0))
    assert result["interrupted"] and "stopped" in shown[-1]
    meta, (rows, _, problems) = scored(result, schedule)
    assert meta["interrupted"] and 5 * 28 + 8 < result["completed_s"] < 5 * 28 + 8 + 20
    assert {int(r["block"].split("#")[1]) for r in rows} == set(range(5)) and any("5 of 15 blocks" in p for p in problems)


def schedule_time(block, into):
    """Fake-clock seconds (after the first window) at which block number `block` (0-based) is `into` seconds old; the board starts 0.5 s before t0."""
    return 0.5 + 8.0 + block * 28.0 + into


def test_no_board_means_a_clear_error_and_no_test():
    result, shown, _, _ = run(performed, never=True)
    assert "no gesture window arrived" in result["error"] and result["windows"] == [] and shown == []


def test_ctrl_c_before_the_first_window_saves_nothing():
    result, _, _, _ = run(performed, interrupt_at=0.0)
    assert result["error"].startswith("stopped before the test started") and result["windows"] == []


def test_a_flood_of_stale_repeated_lines_is_not_counted_as_windows():
    clock, shown = Clock(), []
    schedule = la.make_schedule(seed=2)

    class Flooded(FakeBoard):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.flood = [b"G,99999,12345,rest,3800\r\n"] * 400

        def readline(self):
            return self.flood.pop() if self.flood else super().readline()
    board = Flooded(clock, performed(schedule))
    result = lat.run_test(board, schedule, pause_s=8.0, instructions=INSTR, clock=clock, out=shown.append)
    assert result["monitor"].stale_lines == 399 and sum(1 for w in result["windows"] if w["window"] == 99999) == 1


# ---------------------------------------------------------------- what is saved about a run
def test_the_run_records_how_it_was_made_and_what_the_board_reported():
    result, _, schedule, _ = run(performed)
    mon = result["monitor"]
    mon.feed("# stats windows=20 overruns=2 i2c_err=0 restarts=0 ignored=0 max_compute_us=3900\r\n")
    mon.feed("# odr_hz=50 tick_hz=3000000 timer=hfrc3mhz window=128 step=64 features=40 classes=rest,shake_lr,shake_ud,circle,tilt\r\n")
    meta, _ = scored(result, schedule)
    assert meta["schedule"] == schedule and meta["seed"] == 2 and meta["labels"] == list(la.LABELS) and meta["config_seen"] is True
    assert meta["monitor_problems"] == ["the sensor overwrote 2 samples before they were read"]
    json.dumps(meta)                                                                         # it can be saved


def test_a_missing_settings_line_is_not_reported_as_a_problem_but_a_different_set_of_gestures_is():
    result, _, schedule, _ = run(performed)
    meta, _ = scored(result, schedule)
    assert meta["monitor_problems"] == [] and meta["config_seen"] is False
    result["monitor"].feed("# odr_hz=50 tick_hz=3000000 timer=hfrc3mhz window=128 step=64 features=40 classes=rest,wave\r\n")
    meta, _ = scored(result, schedule)
    assert any("differ from the test's" in p for p in meta["monitor_problems"])


def test_the_model_file_is_identified_by_its_hash_and_its_training_sessions():
    import hashlib
    info = lat.model_info()
    real = hashlib.sha256((ROOT / "firmware" / "sparkfun_edge" / "gesture_demo" / "gesture_model.c").read_bytes()).hexdigest()
    assert info["sha256"] == real and info["trained_on"]["sessions"] == ["s1", "s2", "s3"] and "cannot check" in info["assumption"]
    assert lat.model_info(Path("/nonexistent")) is None


# ---------------------------------------------------------------- the command line
def fake_serial(monkeypatch, schedule, clock, **kw):
    """Replace pyserial and the clock so main() runs a whole test in no time."""
    board = FakeBoard(clock, performed(schedule), **kw)
    mod = types.ModuleType("serial")
    mod.Serial = lambda port, baud, timeout=None: board
    monkeypatch.setitem(sys.modules, "serial", mod)
    monkeypatch.setattr(lat, "time", types.SimpleNamespace(monotonic=clock, time=real_time.time))
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    return board


def test_preview_needs_no_board_and_prints_the_order(capsys):
    assert lat.main(["--preview", "--seed", "4"]) == 0
    out = capsys.readouterr().out
    assert "15 blocks, 20 s each, 8 s to get ready" in out and out.count("block ") == 15 and all(g in out for g in la.LABELS)
    assert [b["target"] for b in la.make_schedule(seed=4)][0] in out.splitlines()[1]


def test_a_whole_test_from_the_command_line_saves_scores_and_points_to_the_next_step(tmp_path, monkeypatch, capsys):
    clock = Clock()
    fake_serial(monkeypatch, la.make_schedule(seed=2), clock)
    assert lat.main(["--port", "COM4", "--session", "Live 1", "--seed", "2", "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    meta, windows = la.read_run(tmp_path / "live_1_live.json")
    assert meta["session"] == "live_1" and meta["seed"] == 2 and meta["completed_s"] == meta["schedule"][-1]["end_s"] and not meta["interrupted"]
    assert meta["model"]["trained_on"]["sessions"] == ["s1", "s2", "s3"]
    exp = sum(expected_scored(meta["schedule"]).values())
    assert f"{exp} windows counted in 15 blocks" in out and "accuracy 1.000" in out
    assert "python scripts/evaluate_live_accuracy.py --run data/own/edge_live/live_1_live.json" in out
    assert len(windows) > exp


def test_a_session_the_model_was_trained_on_is_refused_before_anything_is_opened(tmp_path, capsys):
    assert lat.main(["--port", "COM4", "--session", "S2", "--out-dir", str(tmp_path)]) == 2
    assert "used to train the model" in capsys.readouterr().err and list(tmp_path.iterdir()) == []


def test_an_existing_run_is_not_overwritten_and_this_is_known_before_the_test_starts(tmp_path, capsys):
    from live_helpers import synthetic_run
    la.write_run(tmp_path, *synthetic_run("live1"))
    assert lat.main(["--port", "COM4", "--session", "live1", "--out-dir", str(tmp_path)]) == 2
    assert "already exists" in capsys.readouterr().err


def test_a_missing_port_or_session_is_explained(capsys):
    assert lat.main([]) == 2 and "--port and --session" in capsys.readouterr().err


def test_a_board_that_says_nothing_gives_an_error_and_saves_no_file(tmp_path, monkeypatch, capsys):
    clock = Clock()
    fake_serial(monkeypatch, la.make_schedule(seed=2), clock, never=True)
    assert lat.main(["--port", "COM4", "--session", "live9", "--seed", "2", "--out-dir", str(tmp_path)]) == 1
    assert "no gesture window arrived" in capsys.readouterr().err and list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- docs/API.md describes these files
def test_the_documented_run_files_are_the_files_that_are_written():
    import re
    doc = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    sec = doc[doc.index("### The files of a live accuracy test"):]
    sec = sec[:sec.index("\n### Examples")]
    header, keys = re.findall(r"```text\n(.*?)```", sec, re.S)
    result, _, schedule, _ = run(performed)
    meta, _ = scored(result, schedule)
    assert header.strip() == la.LOG_HEADER
    assert sorted(re.split(r",\s*", keys.replace("\n", " ").strip())) == sorted(meta)
    assert all(f"`{k}`" in sec for k in schedule[0])
    assert all(f"`{k}`" in sec or k in sec for k in la.LOG_HEADER.split(","))
