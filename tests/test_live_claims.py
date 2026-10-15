import json
import re
from pathlib import Path

import pytest
from live_helpers import ORIGINAL_FILES, make_root, synthetic_run, two_runs

from tinyml_lab import live_accuracy_report as rep

ROOT = Path(__file__).resolve().parents[1]
STALE = ["accuracy not yet measured", "the on-device accuracy will come from a session", "The live accuracy: a new session", "No machine learning has run on this board yet",
         "Filling the 128-sample window from the sensor on the board, calling the tree", "Check on-device accuracy against the PC on a held-out set"]


def report(n=2):
    return rep.pool_runs(two_runs() if n == 2 else [synthetic_run("live1", seed=1)])


def read(root, rel):
    return (Path(root) / rel).read_text(encoding="utf-8")


def region(text, key):
    m = re.findall(rf"<!-- live:{key} -->(.*?)<!-- /live:{key} -->", text, re.S)
    assert len(m) == 1, f"{key}: {len(m)} marked places"
    return m[0]


# ---------------------------------------------------------------- the sentences
def test_the_sentences_carry_the_numbers_of_the_report_and_nothing_typed_in():
    r = report()
    acc, res = f"{r['result']['accuracy']:.3f}", r["result"]
    t = rep.claim_texts(r, {"model": "forest", "trees": 100})
    assert t["title"] == f"runs on the board, live accuracy {acc} (2 sittings, one person, one board)"
    assert f"measured in live1 and live2, which are not among its training sessions: {acc} on {res['n_windows']} windows in {res['n_blocks']} blocks" in t["status"]
    assert "so far 2 sittings (live1 and live2)" in t["notdone"] and f"whether {acc} holds" in t["notdone"]
    assert "a 100-tree forest" in t["features"] and "a 100-tree forest" in t["board"] and f"{acc} over {res['n_windows']} windows" in t["roadmap"]
    assert set(t) == set(rep.CLAIM_PLACES)


def test_one_sitting_is_written_in_the_singular_and_an_unknown_model_is_just_a_model():
    r = report(1)
    t = rep.claim_texts(r, None)
    assert "(1 sitting, one person" in t["title"] and "measured in live1, which is not among" in t["status"] and "so far 1 sitting (live1)" in t["notdone"]
    assert "(a model)" in t["features"] and "A gesture model (a model)" in t["board"]
    assert "a 100-tree forest" in rep.claim_texts(r, {"model": "forest", "trees": 100})["board"]


def test_three_sittings_are_listed_with_commas_and_an_and():
    assert rep._join(["a"]) == "a" and rep._join(["a", "b"]) == "a and b" and rep._join(["a", "b", "c"]) == "a, b and c"


# ---------------------------------------------------------------- replacing the old wording
def test_the_first_run_replaces_every_old_sentence_and_touches_nothing_else(tmp_path):
    root = make_root(tmp_path)
    r = report()
    rep.update_claims(root, r)
    texts = rep.claim_texts(r, {"model": "forest", "trees": 100})
    for key, (rel, _) in rep.CLAIM_PLACES.items():
        assert region(read(root, rel), key) == texts[key]
    everything = "".join(read(root, rel) for rel in ORIGINAL_FILES)
    assert not any(s in everything for s in STALE)
    demo = read(root, rep.DEMO)
    assert "intro line" in demo and "- A long run: not measured." in demo and "## Measured\ntext" in demo and demo.startswith("# Gesture chain for the SparkFun Edge: <!-- live:title -->")
    assert "5. Another thing." in read(root, "README.md") and "* The sample rate was not measured." in read(root, "firmware/sparkfun_edge/README.md")
    assert read(root, "firmware/sparkfun_edge/gesture_demo/gesture_model.json") == ORIGINAL_FILES["firmware/sparkfun_edge/gesture_demo/gesture_model.json"]


def test_a_second_run_with_other_numbers_replaces_the_marked_text_instead_of_adding_more(tmp_path):
    root = make_root(tmp_path)
    rep.update_claims(root, report(2))
    rep.update_claims(root, report(1))
    t = rep.claim_texts(report(1), {"model": "forest", "trees": 100})
    for key, (rel, _) in rep.CLAIM_PLACES.items():
        assert region(read(root, rel), key) == t[key]
    assert "2 sittings" not in "".join(read(root, rel) for rel in ORIGINAL_FILES)


def test_running_it_again_with_the_same_report_changes_nothing(tmp_path):
    root = make_root(tmp_path)
    r = report()
    rep.update_claims(root, r)
    first = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    rep.update_claims(root, r)
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == first


def test_if_one_place_cannot_be_found_nothing_is_written(tmp_path):
    root = make_root(tmp_path)
    (root / "README.md").write_text("# Lab\n\n## Roadmap\n4. Something someone already reworded.\n")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match=r"README\.md: cannot find the one sentence to replace for 'roadmap'"):
        rep.update_claims(root, report())
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_a_sentence_that_occurs_twice_is_not_guessed_at(tmp_path):
    root = make_root(tmp_path)
    p = root / "README.md"
    p.write_text(p.read_text() + "4. Check on-device accuracy against the PC on a held-out set.\n")
    with pytest.raises(ValueError, match="roadmap"):
        rep.update_claims(root, report())


def test_a_marker_that_occurs_twice_is_not_guessed_at(tmp_path):
    root = make_root(tmp_path)
    rep.update_claims(root, report())
    p = root / "README.md"
    p.write_text(p.read_text() + "<!-- live:roadmap -->x<!-- /live:roadmap -->\n")
    with pytest.raises(ValueError, match="more than once"):
        rep.update_claims(root, report())


def test_windows_line_endings_survive_in_the_file_that_has_them(tmp_path):
    root = make_root(tmp_path)
    p = root / "README.md"
    p.write_bytes(ORIGINAL_FILES["README.md"].replace("\n", "\r\n").encode())
    rep.update_claims(root, report())
    raw = p.read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
    other = (root / "firmware/sparkfun_edge/README.md").read_bytes()
    assert b"\r" not in other


def test_the_status_sentence_is_found_whichever_way_the_line_is_wrapped(tmp_path):
    files = dict(ORIGINAL_FILES)
    d = "firmware/sparkfun_edge/gesture_demo/README.md"
    files[d] = files[d].replace("every\nrecording", "every recording")
    root = make_root(tmp_path, files)
    rep.update_claims(root, report())
    assert region(read(root, d), "status") == rep.claim_texts(report(), {"model": "forest", "trees": 100})["status"]


# ---------------------------------------------------------------- the real repository says what the results say
def test_the_readmes_of_the_repository_say_what_the_results_file_says():
    results = ROOT / "results" / "live_accuracy.json"
    if not results.is_file():
        pytest.skip("no live accuracy results in this checkout")
    saved = json.loads(results.read_text(encoding="utf-8"))
    model = json.loads((ROOT / rep.MODEL_JSON).read_text(encoding="utf-8"))
    texts = rep.claim_texts(saved, model)
    for key, (rel, _) in rep.CLAIM_PLACES.items():
        assert region((ROOT / rel).read_text(encoding="utf-8"), key) == texts[key], f"{rel} ({key}) does not match results/live_accuracy.json: run scripts/evaluate_live_accuracy.py"
    everything = "".join((ROOT / rel).read_text(encoding="utf-8") for rel, _ in rep.CLAIM_PLACES.values())
    assert not any(s in everything for s in STALE)
    assert rep.START in (ROOT / rep.DEMO).read_text(encoding="utf-8")
