import csv
from pathlib import Path

CARD = Path(__file__).resolve().parents[1] / "results" / "scorecard.csv"
HEADER = ["step", "id", "prediction", "pct", "event_happened", "result", "note"]


def load():
    with CARD.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        body = list(reader)
    return header, body


def test_header():
    header, _ = load()
    assert header == HEADER


def test_rows_have_seven_fields():
    _, body = load()
    assert body
    assert all(len(r) == 7 for r in body)


def test_ids_are_unique():
    _, body = load()
    ids = [r[1] for r in body]
    assert len(ids) == len(set(ids))


def test_result_follows_the_fifty_percent_rule():
    _, body = load()
    for _step, pid, _pred, pct, happened, result, _note in body:
        p = int(pct)
        assert 1 <= p <= 99, pid
        assert happened in ("yes", "no"), pid
        assert result in ("hit", "miss"), pid
        expected = "hit" if (p >= 50) == (happened == "yes") else "miss"
        assert result == expected, pid


def test_every_miss_has_a_note():
    _, body = load()
    for r in body:
        if r[5] == "miss":
            assert r[6].strip(), r[1]
