import pytest
from live_helpers import STEP, expected_scored, synthetic_run

from tinyml_lab import live_accuracy as la

LABELS2 = ["a", "b"]


def rows_from_cm(cm, labels, per_block=3):
    """Counted-window rows that produce exactly this confusion matrix, spread over `per_block` blocks per gesture."""
    rows = []
    for i, g in enumerate(labels):
        for j, p in enumerate(labels):
            for n in range(cm[i][j]):
                rows.append({"block": f"{g}{n % per_block}", "target": g, "predicted": p, "window": len(rows), "t_s": 0.0})
    return rows


# ---------------------------------------------------------------- the schedule
def test_every_gesture_appears_equally_often_and_the_times_are_laid_out_as_documented():
    s = la.make_schedule(rounds=3, seed=5, block_s=20, pause_s=8)
    assert len(s) == 15 and all(sum(b["target"] == g for b in s) == 3 for g in la.LABELS)
    assert [b["index"] for b in s] == list(range(15))
    assert [b["start_s"] for b in s][:3] == [8, 36, 64] and all(b["end_s"] - b["start_s"] == 20 for b in s)
    assert all(sorted(b["target"] for b in s if b["round"] == r) == sorted(la.LABELS) for r in range(3))


def test_the_same_seed_gives_the_same_order_and_another_seed_another_one():
    order = lambda seed: [b["target"] for b in la.make_schedule(seed=seed)]
    assert order(7) == order(7) and order(7) != order(8)


@pytest.mark.parametrize("seed", range(60))
def test_the_same_gesture_never_comes_twice_in_a_row(seed):
    t = [b["target"] for b in la.make_schedule(rounds=4, seed=seed)]
    assert all(a != b for a, b in zip(t, t[1:]))


@pytest.mark.parametrize("kw", [{"labels": ["a"]}, {"labels": ["a", "a"]}, {"rounds": 0}, {"block_s": 9.9}, {"pause_s": 3.9}])
def test_a_schedule_that_cannot_work_is_refused(kw):
    with pytest.raises(ValueError):
        la.make_schedule(**kw)


def test_the_documented_minimum_of_windows_per_block_is_the_true_minimum():
    """Counted by brute force over where the 1.28 s grid of window ends falls: the fewest ends inside the scored zone is the number the code promises."""
    zone_lo = la.LEAD_IN_S + la.WINDOW_S                      # the earliest end of a window that starts after the lead-in (4.56 s into a 20 s block)
    zone_hi = 20.0 - la.TAIL_GUARD_S
    counts = []
    for phase in range(128):                                  # the grid can sit anywhere: ends at phase/100 + k * 1.28
        ends = [phase / 100.0 + k * STEP for k in range(40)]
        counts.append(sum(1 for e in ends if zone_lo <= e <= zone_hi))
    assert min(counts) == la.min_windows_per_block(20.0) == 11 and max(counts) == 12


# ---------------------------------------------------------------- which windows count
def win(t, ticks=None, w=1, pred="rest"):
    return {"t_s": t, "t_ticks": int(t * la.TICK_HZ) % la.WRAP if ticks is None else ticks, "window": w, "predicted": pred, "compute_us": 3800}


def test_windows_on_a_steady_clock_are_on_time_and_the_first_is_not_counted():
    ws = [win(i * 1.28, w=i) for i in range(6)]
    assert la.timely_flags(ws) == [False, True, True, True, True, True]


def test_a_burst_after_a_stall_is_late_and_the_window_after_it_too_but_not_the_ones_after_that():
    ws = [win(1.28 * i, w=i) for i in range(4)]
    ws += [win(10.0, ticks=int(1.28 * k * la.TICK_HZ), w=k) for k in (4, 5, 6)]           # three windows delivered together at 10 s
    ws += [win(10.0 + 1.28 * j, ticks=int(1.28 * (7 + j) * la.TICK_HZ), w=7 + j) for j in range(3)]
    assert la.timely_flags(ws) == [False, True, True, True, False, False, False, False, True, True]


def test_a_missing_line_does_not_make_the_next_one_late():
    ws = [win(0.0), win(1.28), win(3.84, ticks=int(3.84 * la.TICK_HZ))]                   # window 3 never arrived; the board's own time says 2.56 s passed
    assert la.timely_flags(ws) == [False, True, True]


def test_the_32_bit_tick_counter_wrapping_is_not_a_late_window():
    t0 = la.WRAP - 2_000_000
    ws = [win(1.28 * i, ticks=(t0 + i * 3_840_000) % la.WRAP) for i in range(4)]
    assert la.timely_flags(ws) == [False, True, True, True]


def test_the_edges_of_the_scored_zone_are_where_the_lead_in_and_the_end_margin_put_them():
    """Windows 10 ms either side of each edge (10 ms, not exact: 2.56 is not exact in binary, so an exact edge could fall either way)."""
    block = [{"index": 0, "round": 0, "target": "rest", "start_s": 10.0, "end_s": 30.0}]
    lo = 10.0 + la.LEAD_IN_S + la.WINDOW_S                    # a window ending at 14.56 s starts exactly when the lead-in ends
    hi = 30.0 - la.TAIL_GUARD_S
    ends = [0.0, lo - 0.01, lo + 0.01, hi - 0.01, hi + 0.01]
    ws = [dict(win(t, w=i), t_ticks=int(round(t * la.TICK_HZ))) for i, t in enumerate(ends)]       # every window after the first is spaced like the board's clock
    rows, dropped = la.assign_windows(ws, block)
    assert [r["window"] for r in rows] == [2, 3] and dropped == {"outside_scored_zone": 3, "unfinished_block": 0, "delayed": 0}


def test_a_clean_run_counts_exactly_the_windows_a_brute_force_count_says():
    meta, windows = synthetic_run(seed=2)
    rows, dropped, problems = la.score_run(meta, windows)
    exp = expected_scored(meta["schedule"])
    got = {}
    for r in rows:
        got[int(r["block"].split("#")[1])] = got.get(int(r["block"].split("#")[1]), 0) + 1
    assert got == exp and sum(exp.values()) == len(rows) and problems == [] and dropped["delayed"] == 0
    assert dropped["outside_scored_zone"] == len(windows) - len(rows) and dropped["unfinished_block"] == 0
    assert all(r["predicted"] == r["target"] for r in rows)


def test_nothing_in_a_pause_or_a_lead_in_is_scored_whatever_the_board_says():
    """The board says 'tilt' all the time; every window that does get scored must end inside some block's scored zone."""
    meta, windows = synthetic_run(seed=2)
    rows, _, _ = la.score_run(meta, [dict(w, predicted="tilt") for w in windows])
    zone = [(b["start_s"] + la.LEAD_IN_S + la.WINDOW_S, b["end_s"] - la.TAIL_GUARD_S) for b in meta["schedule"]]
    assert rows and all(any(lo <= r["t_s"] <= hi for lo, hi in zone) for r in rows)


def test_blocks_the_test_did_not_finish_are_not_scored_and_the_run_says_so():
    meta, windows = synthetic_run(seed=2)
    cut = meta["schedule"][5]["start_s"] + 12.0
    meta["completed_s"], meta["interrupted"] = cut, True
    rows, dropped, problems = la.score_run(meta, windows)
    assert {int(r["block"].split("#")[1]) for r in rows} == set(range(5))
    assert dropped["unfinished_block"] > 0 and any("stopped early: 5 of 15 blocks" in p for p in problems)


def test_a_block_with_lost_lines_is_reported():
    meta, windows = synthetic_run(seed=2)
    b = meta["schedule"][3]
    keep = [w for w in windows if not (b["start_s"] + 6 <= w["t_s"] <= b["start_s"] + 14)]
    _, _, problems = la.score_run(meta, keep)
    assert any("block 4" in p and "fewer than the 11" in p for p in problems)


def test_the_boards_own_problems_are_passed_on():
    meta, windows = synthetic_run()
    meta["monitor_problems"] = ["the sensor overwrote 3 samples before they were read"]
    assert "the board reported: the sensor overwrote 3 samples before they were read" in la.score_run(meta, windows)[2]


# ---------------------------------------------------------------- the scores
def test_a_hand_worked_example():
    res = la.evaluate(rows_from_cm([[8, 2], [1, 9]], LABELS2), LABELS2, n_boot=50)
    assert res["n_windows"] == 20 and res["accuracy"] == pytest.approx(17 / 20)
    a, b = res["per_class"]["a"], res["per_class"]["b"]
    assert (a["recall"], b["recall"]) == (pytest.approx(0.8), pytest.approx(0.9))
    assert (a["precision"], b["precision"]) == (pytest.approx(8 / 9), pytest.approx(9 / 11))
    assert (a["f1"], b["f1"]) == (pytest.approx(16 / 19), pytest.approx(18 / 21))
    assert res["macro_f1"] == pytest.approx((16 / 19 + 18 / 21) / 2) and res["chance_accuracy"] == 0.5
    assert res["confusion_matrix"] == [[8, 2], [1, 9]]


def test_the_scores_agree_with_scikit_learn_on_random_predictions():
    import random
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    rng = random.Random(4)
    labels = list(la.LABELS)
    rows = []
    for i in range(300):
        t = labels[i % 5]
        rows.append({"block": f"b{i // 20}", "target": t, "predicted": t if rng.random() < 0.7 else rng.choice(labels), "window": i, "t_s": 0.0})
    res = la.evaluate(rows, labels, n_boot=20)
    y, p = [r["target"] for r in rows], [r["predicted"] for r in rows]
    assert res["accuracy"] == pytest.approx(accuracy_score(y, p)) and res["macro_f1"] == pytest.approx(f1_score(y, p, labels=labels, average="macro"))
    assert res["confusion_matrix"] == confusion_matrix(y, p, labels=labels).tolist()
    assert [res["per_class"][g]["recall"] for g in labels] == pytest.approx(recall_score(y, p, labels=labels, average=None))
    assert [res["per_class"][g]["precision"] for g in labels] == pytest.approx(precision_score(y, p, labels=labels, average=None))


def test_a_gesture_that_was_not_tested_is_left_out_of_macro_f1_and_has_no_recall():
    rows = rows_from_cm([[9, 1], [0, 0]], LABELS2)
    res = la.evaluate(rows, LABELS2, n_boot=20)
    assert res["per_class"]["b"]["recall"] is None and res["macro_f1"] == pytest.approx(res["per_class"]["a"]["f1"])


def test_nothing_to_score_and_unknown_gestures_are_refused():
    with pytest.raises(ValueError, match="no windows"):
        la.evaluate([], LABELS2)
    bad = rows_from_cm([[3, 0], [0, 3]], LABELS2) + [{"block": "x", "target": "a", "predicted": "wave", "window": 1, "t_s": 0.0}]
    with pytest.raises(ValueError, match="wave"):
        la.evaluate(bad, LABELS2)


def test_the_interval_is_repeatable_for_a_given_seed():
    """Few resamples, several seeds: an unseeded generator would give a different interval from time to time, and agreeing every time is negligible."""
    rows = rows_from_cm([[16, 4], [3, 17]], LABELS2, per_block=4)
    for seed in range(8):
        assert la.evaluate(rows, LABELS2, seed=seed, n_boot=25)["ci"] == la.evaluate(rows, LABELS2, seed=seed, n_boot=25)["ci"]


def test_the_interval_holds_the_estimate_and_reports_how_it_was_made():
    rows = rows_from_cm([[16, 4], [3, 17]], LABELS2, per_block=4)
    a = la.evaluate(rows, LABELS2, seed=3, n_boot=400)
    assert a["ci"]["accuracy"][0] <= a["accuracy"] <= a["ci"]["accuracy"][1]
    assert a["ci"]["accuracy"][0] < a["ci"]["accuracy"][1] and a["ci"]["n_boot"] == 400 and a["ci"]["level"] == 0.95


def test_the_interval_is_the_middle_95_percent_of_the_resamples_and_each_resample_is_as_big_as_the_data():
    """Gesture a: 10 blocks of 10 windows, nine perfect and one all wrong. Gesture b: 2 perfect blocks. Each resample draws 10 blocks of a, so the number
    of wrong blocks drawn is Binomial(10, 0.1): P(3 or more) = 0.070 and P(4 or more) = 0.013, which puts the 2.5 % point at 3 wrong blocks, accuracy
    1 - 30/120 = 0.75. The full range of the resamples would reach down to 5 or 6 wrong blocks (0.58 or lower); drawing only one block per gesture would
    give a bound near 0.17 (the one block drawn is the wrong one)."""
    rows = []
    for i in range(10):
        rows += [{"block": f"a{i}", "target": "a", "predicted": "b" if i == 0 else "a", "window": j, "t_s": 0.0} for j in range(10)]
    for i in range(2):
        rows += [{"block": f"b{i}", "target": "b", "predicted": "b", "window": j, "t_s": 0.0} for j in range(10)]
    ci = la.evaluate(rows, LABELS2, seed=1, n_boot=2000)["ci"]
    assert ci["accuracy"][1] == 1.0 and 0.72 <= ci["accuracy"][0] <= 0.78


def test_a_perfect_model_has_a_degenerate_interval():
    ci = la.evaluate(rows_from_cm([[12, 0], [0, 12]], LABELS2), LABELS2, n_boot=100)["ci"]
    assert ci["accuracy"] == [1.0, 1.0] and ci["macro_f1"] == [1.0, 1.0]


def test_blocks_are_resampled_inside_each_gesture_so_the_mix_of_gestures_never_changes():
    """Gesture a: two blocks, all right. Gesture b: two blocks, all wrong. Resampling inside each gesture keeps accuracy at exactly 0.5 in every resample;
    resampling blocks from the whole pool would swing it anywhere from 0 to 1."""
    rows = []
    for blk, tgt, pred in (("a0", "a", "a"), ("a1", "a", "a"), ("b0", "b", "a"), ("b1", "b", "a")):
        rows += [{"block": blk, "target": tgt, "predicted": pred, "window": i, "t_s": 0.0} for i in range(10)]
    ci = la.evaluate(rows, LABELS2, n_boot=300)["ci"]
    assert ci["accuracy"] == [0.5, 0.5]


def test_a_single_block_of_a_gesture_means_no_interval_and_says_why():
    rows = rows_from_cm([[6, 0], [0, 6]], LABELS2, per_block=1)
    res = la.evaluate(rows, LABELS2, n_boot=50)
    assert res["ci"] is None and "at least 2 blocks" in res["ci_unavailable"] and "a" in res["ci_unavailable"]
    assert "no interval" in la.summary_text(res)


def test_the_chance_level_is_always_guessing_the_commonest_gesture_not_one_over_the_number_of_gestures():
    res = la.evaluate(rows_from_cm([[6, 0], [2, 2]], LABELS2), LABELS2, n_boot=20)
    assert res["chance_accuracy"] == pytest.approx(0.6)


def test_the_blocks_are_listed_with_their_own_counts():
    res = la.evaluate(rows_from_cm([[4, 2], [1, 5]], LABELS2, per_block=2), LABELS2, n_boot=20)
    pb = {x["block"]: x for x in res["per_block"]}
    assert sum(x["windows"] for x in pb.values()) == 12 and sum(x["correct"] for x in pb.values()) == 9
    assert all(x["target"] == x["block"][0] for x in pb.values())


def test_the_text_summary_shows_the_numbers_and_is_plain_ascii():
    meta, windows = synthetic_run(seed=2, wrong={0: "tilt"})
    rows, _, _ = la.score_run(meta, windows)
    res = la.evaluate(rows, la.LABELS, n_boot=100)
    text = la.summary_text(res)
    text.encode("ascii")
    assert f"accuracy {res['accuracy']:.3f}" in text and f"{res['n_windows']} windows counted in 15 blocks" in text and "rows are the gesture done" in text


# ---------------------------------------------------------------- the files
def test_a_run_is_saved_and_read_back_unchanged(tmp_path):
    meta, windows = synthetic_run()
    csv_path, json_path = la.write_run(tmp_path, meta, windows)
    assert csv_path.name == "live1_live.csv" and json_path.name == "live1_live.json"
    meta2, windows2 = la.read_run(json_path)
    assert meta2 == meta and len(windows2) == len(windows)
    assert [w["t_ticks"] for w in windows2] == [w["t_ticks"] for w in windows] and windows2[3]["predicted"] == windows[3]["predicted"]
    assert windows2[5]["t_s"] == pytest.approx(windows[5]["t_s"], abs=1e-3)


def test_a_run_is_never_overwritten(tmp_path):
    meta, windows = synthetic_run()
    la.write_run(tmp_path, meta, windows)
    with pytest.raises(FileExistsError, match="never overwritten"):
        la.write_run(tmp_path, meta, windows[:3])
    assert len(la.read_run(tmp_path / "live1_live.json")[1]) == len(windows)


def test_damaged_rows_in_a_run_are_skipped(tmp_path):
    meta, windows = synthetic_run()
    csv_path, json_path = la.write_run(tmp_path, meta, windows)
    csv_path.write_text(csv_path.read_text() + "x,1,2,rest,3\nbroken\n")
    assert len(la.read_run(json_path)[1]) == len(windows)
