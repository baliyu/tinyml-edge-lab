#!/usr/bin/env python3
"""Score the live accuracy test(s) of the gesture model running on the SparkFun Edge, and write the results.

Usage: python scripts/evaluate_live_accuracy.py --run data/own/edge_live/live1_live.json [--run data/own/edge_live/live2_live.json]
                                               [--out results] [--readme firmware/sparkfun_edge/gesture_demo/README.md] [--no-readme]
Reads the runs written by tools/live_accuracy_test.py, counts the windows that lie fully inside a block, and scores the board's predictions against the
gesture that was shown: confusion matrix, accuracy, macro-F1, recall and precision per gesture, and an interval that resamples whole blocks. Give every run
you made: a run is not left out because it went badly. Writes <out>/live_accuracy.json, <out>/figures/live_confusion.png and the "Live accuracy on the board"
section of the demo README, all from the runs, with no typed-in numbers. It refuses runs of a session the model was trained on.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tinyml_lab import live_accuracy as la  # noqa: E402
from tinyml_lab import live_accuracy_report as rep  # noqa: E402

DEFAULT_README = Path(__file__).resolve().parents[1] / "firmware" / "sparkfun_edge" / "gesture_demo" / "README.md"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, metavar="FILE.json", help="a run written by tools/live_accuracy_test.py; repeat for more than one")
    ap.add_argument("--out", default="results")
    ap.add_argument("--readme", default=str(DEFAULT_README))
    ap.add_argument("--no-readme", action="store_true", help="write the results and the figure but leave the README alone")
    a = ap.parse_args(argv)
    runs = []
    for f in a.run:
        if not Path(f).is_file() or not Path(f).with_suffix(".csv").is_file():
            print(f"{f} or its .csv does not exist.", file=sys.stderr)
            return 2
        runs.append(la.read_run(f))
    try:
        report = rep.pool_runs(runs)
    except ValueError as e:
        print(f"STOP: {e}", file=sys.stderr)
        return 1
    for x in report["runs"]:
        print(f"{x['session']}: {x['completed_blocks']} of {x['blocks']} blocks completed, {x['windows_received']} windows from the board, {x['counted']} counted")
        for p in x["problems"]:
            print("  -", p)
    print()
    print(la.summary_text(report["result"]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    offline = rep.offline_reference(out)
    (out / "live_accuracy.json").write_text(json.dumps({**report, "offline_reference": offline}, indent=2) + "\n", encoding="utf-8")
    rep.confusion_fig(report["result"], out / "figures" / "live_confusion.png")
    print(f"\nwrote {out / 'live_accuracy.json'} and {out / 'figures' / 'live_confusion.png'}")
    if not a.no_readme:
        rep.update_readme(a.readme, rep.render_block(report, offline))
        print(f"updated {a.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
