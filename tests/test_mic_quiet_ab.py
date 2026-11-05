"""Tests for tools/mic_quiet_ab.py: reading recordings, the use-or-leave-out rules, the decision rule and its edges, and that the plan's numbers equal the code's."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mic_quiet_ab", ROOT / "tools" / "mic_quiet_ab.py")
ab = importlib.util.module_from_spec(spec)
sys.modules["mic_quiet_ab"] = ab
spec.loader.exec_module(ab)


def write_rec(d, session, power, block, order, rms, n=360, n_invalid=2, verdict="OK", label="quiet", note=None):
    note = note if note is not None else f"power={power} block={block} order={order}"
    stamp = f"2026110{block}_09{order:02d}00"
    head = ["# SparkFun Edge microphone summary capture (firmware mic_logger)", f"# label={label}", "# subject=s1", f"# session={session}", f"# note={note}",
            f"# captured_at=2026-11-0{block}T09:{order:02d}:00", "# port=COM4 baud=115200", f"# windows={n + n_invalid} valid={n}", f"# verdict={verdict}",
            "w,mean,rms,peak,zcr,hf,clip,rate_hz,valid,why"]
    rows = [f"{i},8000,9999,9999,9,9999,0,7880,0,start" for i in range(n_invalid)]
    rows += [f"{n_invalid + i},8000,{rms},{rms * 5},500,{rms * 0.8:.0f},0,7880,1," for i in range(n)]
    p = Path(d) / f"{session}_{label}_{stamp}.csv"
    p.write_text("\n".join(head + rows) + "\n", encoding="utf-8")
    return p


def block(d, session, n, plugged, battery, spread=0.0):
    """Four recordings in the order plugged, battery, battery, plugged; `spread` moves the second recording of each condition by that fraction."""
    write_rec(d, session, "plugged", n, 1, plugged)
    write_rec(d, session, "battery", n, 2, battery)
    write_rec(d, session, "battery", n, 3, round(battery * (1 + spread)))
    write_rec(d, session, "plugged", n, 4, round(plugged * (1 + spread)))


def verdict_of(d, sessions=("ab1", "ab2")):
    recs = ab.load(d, list(sessions))
    return ab.decide(ab.group_blocks(recs)[0])[0]


def test_the_note_tags_are_read_and_missing_ones_are_absent():
    assert ab.parse_note("power=plugged block=1 order=4 free text") == {"power": "plugged", "block": "1", "order": "4"}
    assert ab.parse_note("") == {} and ab.parse_note(None) == {}


def test_a_recording_is_read_with_its_header_tags_and_the_medians_of_its_valid_windows_only(tmp_path):
    r = ab.read_recording(write_rec(tmp_path, "ab1", "plugged", 1, 1, 140, n=300))
    assert r["label"] == "quiet" and r["session"] == "ab1" and r["tags"] == {"power": "plugged", "block": "1", "order": "1"}
    assert r["n_rows"] == 302 and r["n_valid"] == 300 and r["verdict"] == "OK"
    assert r["medians"]["rms"] == 140 and r["medians"]["peak"] == 700 and r["medians"]["mean"] == 8000        # the 9999 start-up windows are not counted


def test_load_takes_only_the_quiet_recordings_of_the_named_sessions_in_time_order(tmp_path):
    block(tmp_path, "ab1", 1, 140, 90)
    write_rec(tmp_path, "ab1", "plugged", 1, 5, 140, label="speech")
    write_rec(tmp_path, "other", "plugged", 1, 1, 140)
    recs = ab.load(tmp_path, ["ab1"])
    assert [r["tags"]["order"] for r in recs] == ["1", "2", "3", "4"] and {r["session"] for r in recs} == {"ab1"}


@pytest.mark.parametrize("kw,reason", [(dict(label="speech"), "not a quiet recording"), (dict(verdict="PROBLEMS: the board restarted 1 time(s)"), "capture tool reported problems"),
                                        (dict(n=100), "only 100 valid windows"), (dict(note="block=1 order=1"), "no power="), (dict(note="power=plugged order=1"), "no block="),
                                        (dict(note="power=plugged block=1"), "no block="), (dict(note="power=ac block=1 order=1"), "no power=")])
def test_a_recording_is_left_out_with_a_reason(tmp_path, kw, reason):
    args = dict(session="ab1", power="plugged", block=1, order=1, rms=100)
    args.update(kw)
    ok, why = ab.usable(ab.read_recording(write_rec(tmp_path, **args)))
    assert not ok and reason in why


def test_a_good_recording_is_used_and_exactly_the_minimum_number_of_valid_windows_is_enough(tmp_path):
    assert ab.usable(ab.read_recording(write_rec(tmp_path, "ab1", "plugged", 1, 1, 100, n=ab.MIN_VALID)))[0]
    assert not ab.usable(ab.read_recording(write_rec(tmp_path, "ab1", "plugged", 1, 2, 100, n=ab.MIN_VALID - 1)))[0]


def test_spread_is_the_largest_difference_as_a_fraction_of_the_mean():
    assert ab.spread([95, 105]) == pytest.approx(0.10) and ab.spread([100, 100]) == 0 and ab.spread([90, 100, 110]) == pytest.approx(0.20)


def test_block_stats_gives_the_ratio_and_the_spread_of_each_condition():
    s = ab.block_stats({"plugged": [140, 144], "battery": [90, 90]})
    assert s["complete"] and s["ratio"] == pytest.approx(142 / 90) and s["plugged_spread"] == pytest.approx(4 / 142) and s["battery_spread"] == 0
    assert not ab.block_stats({"plugged": [140], "battery": [90, 90]})["complete"]


def test_a_clear_and_repeatable_difference_in_two_blocks_is_supported(tmp_path):
    block(tmp_path, "ab1", 1, 140, 90, spread=0.03)
    block(tmp_path, "ab2", 2, 135, 95, spread=0.03)
    assert verdict_of(tmp_path) == "supported"


def test_no_difference_in_two_blocks_is_not_supported(tmp_path):
    block(tmp_path, "ab1", 1, 100, 98, spread=0.02)
    block(tmp_path, "ab2", 2, 97, 100, spread=0.02)
    assert verdict_of(tmp_path) == "not supported"


def test_blocks_that_disagree_are_inconclusive(tmp_path):
    block(tmp_path, "ab1", 1, 140, 90)
    block(tmp_path, "ab2", 2, 100, 100)
    assert verdict_of(tmp_path) == "inconclusive"


def test_a_ratio_between_the_two_thresholds_is_inconclusive(tmp_path):
    block(tmp_path, "ab1", 1, 115, 100)
    block(tmp_path, "ab2", 2, 115, 100)
    assert verdict_of(tmp_path) == "inconclusive"


def test_the_opposite_direction_is_inconclusive_because_only_plugged_higher_was_predicted(tmp_path):
    block(tmp_path, "ab1", 1, 90, 140)
    block(tmp_path, "ab2", 2, 95, 135)
    assert verdict_of(tmp_path) == "inconclusive"


def test_recordings_that_do_not_agree_with_themselves_make_the_result_inconclusive_whatever_the_ratio(tmp_path):
    block(tmp_path, "ab1", 1, 140, 90, spread=0.25)
    block(tmp_path, "ab2", 2, 140, 90, spread=0.0)
    v, reasons = ab.decide(ab.group_blocks(ab.load(tmp_path, ["ab1", "ab2"]))[0])
    assert v == "inconclusive" and "block 1" in reasons[0] and "10%" in reasons[0]


def test_one_block_or_an_incomplete_block_is_not_enough(tmp_path):
    block(tmp_path, "ab1", 1, 140, 90)
    assert verdict_of(tmp_path, ("ab1",)) == "not enough blocks"
    write_rec(tmp_path, "ab2", "plugged", 2, 1, 140)
    write_rec(tmp_path, "ab2", "battery", 2, 2, 90)
    assert verdict_of(tmp_path) == "not enough blocks"


def test_a_recording_left_out_can_make_its_block_incomplete(tmp_path):
    block(tmp_path, "ab1", 1, 140, 90)
    block(tmp_path, "ab2", 2, 140, 90)
    write_rec(tmp_path, "ab2", "battery", 2, 3, 90, n=100)                      # overwrites the third recording of block 2 with a short one
    assert verdict_of(tmp_path) == "not enough blocks"


@pytest.mark.parametrize("ratio_plugged,expected", [(120, "supported"), (119, "inconclusive")])
def test_a_ratio_of_exactly_the_support_threshold_counts_as_supported(tmp_path, ratio_plugged, expected):
    block(tmp_path, "ab1", 1, ratio_plugged, 100)
    block(tmp_path, "ab2", 2, ratio_plugged, 100)
    assert verdict_of(tmp_path) == expected


@pytest.mark.parametrize("plugged,expected", [(90, "not supported"), (110, "not supported"), (89, "inconclusive"), (111, "inconclusive")])
def test_the_edges_of_the_no_difference_range_are_included(tmp_path, plugged, expected):
    block(tmp_path, "ab1", 1, plugged, 100)
    block(tmp_path, "ab2", 2, plugged, 100)
    assert verdict_of(tmp_path) == expected


def test_a_spread_of_exactly_ten_percent_is_allowed_and_more_is_not(tmp_path):
    for session, block_no in (("ab1", 1), ("ab2", 2)):
        write_rec(tmp_path, session, "plugged", block_no, 1, 150)
        write_rec(tmp_path, session, "plugged", block_no, 4, 150)
        write_rec(tmp_path, session, "battery", block_no, 2, 95)
        write_rec(tmp_path, session, "battery", block_no, 3, 105)                # spread (105 - 95) / 100 = 0.10
    assert verdict_of(tmp_path) == "supported"
    write_rec(tmp_path, "ab1", "battery", 1, 3, 106)                              # spread 11 / 100.5 is just over 10%
    assert verdict_of(tmp_path) == "inconclusive"


def test_the_report_lists_every_recording_the_ratios_and_the_verdict_and_what_was_left_out(tmp_path):
    block(tmp_path, "ab1", 1, 140, 90)
    block(tmp_path, "ab2", 2, 140, 90)
    write_rec(tmp_path, "ab2", "plugged", 2, 5, 140, verdict="PROBLEMS: x")
    text, verdict = ab.report(ab.load(tmp_path, ["ab1", "ab2"]))
    assert verdict == "supported" and "VERDICT: supported" in text and "ratio 1.56" in text and "left out: ab2_quiet_" in text and "9 quiet recording(s) read, 8 used" in text
    assert text.count("plugged ") >= 4 and "one person, one board, one room" in text


def test_command_line_prints_the_verdict_and_reports_missing_recordings(tmp_path, capsys):
    assert ab.main(["--sessions", "ab1", "--dir", str(tmp_path)]) == 1
    assert "no quiet recordings found" in capsys.readouterr().out
    block(tmp_path, "ab1", 1, 140, 90)
    block(tmp_path, "ab2", 2, 140, 90)
    assert ab.main(["--sessions", "ab1,ab2", "--dir", str(tmp_path)]) == 0
    assert "VERDICT: supported" in capsys.readouterr().out


def test_the_numbers_in_the_plan_are_the_numbers_in_the_code():
    plan = (ROOT / "docs" / "QUIET_AB_PLAN.md").read_text(encoding="utf-8")
    assert f"{ab.SUPPORT_RATIO:.2f}" in plan and f"{ab.NULL_LOW:.2f} and {ab.NULL_HIGH:.2f}" in plan
    assert f"{ab.MAX_SPREAD:.0%}" in plan and f"{ab.MIN_VALID} valid windows" in plan and f"{ab.MIN_BLOCKS} complete blocks" in plan
    assert "plugged, battery, battery, plugged" in plan


def test_a_recording_is_summarised_by_the_median_so_a_few_loud_windows_do_not_move_it(tmp_path):
    p = write_rec(tmp_path, "ab1", "plugged", 1, 1, 100, n=300)
    extra = "\n".join(f"{400 + i},8000,5000,25000,500,4000,0,7880,1," for i in range(60)) + "\n"        # 60 loud valid windows, like an Xbox switching on
    p.write_text(p.read_text(encoding="utf-8") + extra, encoding="utf-8")
    r = ab.read_recording(p)
    assert r["n_valid"] == 360 and r["medians"]["rms"] == 100 and r["medians"]["hf"] == 80
