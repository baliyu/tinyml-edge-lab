# Quiet-level A/B result: does laptop power state change the SparkFun Edge microphone's quiet rms?

Result by the pre-registered rule: **not supported**.

## Question

Earlier quiet recordings gave an rms of about 140 in one sitting and about 82 to 87 in later ones. A fan-load test did not explain it. Plugged versus battery was the suspected cause, but that was never tested under control. This note records the controlled test.

## What was fixed before recording

The plan (`docs/QUIET_AB_PLAN.md`), the analysis script (`tools/mic_quiet_ab.py`) and its tests were committed before any recording, in commit `1a1d6df` ("written before recording"). A `git diff` of the plan and the script against that commit printed nothing before the analysis was run, so neither file changed after the pre-registration.

Decision rule, as committed:

- Supported: at least 2 complete blocks, repeats within 10% of their mean, and a plugged/battery rms ratio of at least 1.20 in every block.
- Not supported: at least 2 complete blocks, repeats within 10%, and a ratio between 0.90 and 1.10 in every block.
- Inconclusive: every other case.
- Not enough blocks: fewer than 2 complete blocks.

## Method

- Two blocks, `ab1` and `ab2`, on 11 November 2026. Block 1 started at 09:39 and block 2 at 13:18, so the blocks were more than 3 hours apart.
- Each block: four recordings of 90 s, in the order plugged, battery, battery, plugged, with a 60 s wait after each power change.
- Board flat in a marked spot, quiet room, one laptop, one board.
- Capture command: `py tools\mic_capture.py --port COM4 --label quiet --seconds 90 --session <ab1|ab2> --note "power=<plugged|battery> block=<n> order=<n>"`.
- Analysis: `python3 tools/mic_quiet_ab.py --sessions ab1,ab2`.

## Results

| block | order | power | rms | peak | hf | mean | valid windows |
|---|---|---|---|---|---|---|---|
| 1 | 1 | plugged | 105 | 464 | 95 | 8373 | 354 |
| 1 | 2 | battery | 101 | 444 | 95 | 8381 | 354 |
| 1 | 3 | battery | 101 | 445 | 95 | 8381 | 354 |
| 1 | 4 | plugged | 102 | 439 | 95 | 8381 | 354 |
| 2 | 1 | plugged | 103 | 445 | 102 | 8415 | 353 |
| 2 | 2 | battery | 100 | 432 | 101 | 8416 | 352 |
| 2 | 3 | battery | 98 | 428 | 101 | 8414 | 354 |
| 2 | 4 | plugged | 100 | 436 | 101 | 8413 | 354 |

| block | plugged rms | battery rms | ratio | spread, plugged | spread, battery |
|---|---|---|---|---|---|
| 1 | 104 | 101 | 1.02 | 2.9% | 0.0% |
| 2 | 102 | 99 | 1.02 | 3.0% | 1.5% |

Both ratios are inside 0.90 to 1.10 and every spread is under 10%, so the rule gives **not supported**: in these two sittings, plugged versus battery made no difference to the quiet rms.

## Predictions scorecard

Percentages below are computed from the rounded values in the table above, so they are approximate.

| # | Prediction | Result | Outcome |
|---|---|---|---|
| P1 | Plugged rms is at least 1.20 times battery in both blocks | Ratios 1.02 and 1.02 | Missed. The effect was not there at all. |
| P2 | Two recordings of one condition agree within 10% of their mean | Spreads from 0.0% to 3.0% | Hit |
| P3 | Plugged peak at least 20% higher and hf at least 5% higher | Peak about +1.6% (block 1) and +2.4% (block 2); hf about 0% and +0.5% | Missed on both thresholds. Reported only, not part of the decision rule. |

P1 was based on an uncontrolled comparison across different days. The controlled test does not support power state as the cause of the earlier difference.

## Deviation from the plan, disclosed

The first attempt at block 2, order 1 (`ab2_quiet_20261111_131804.csv`) came back from the capture tool with verdict PROBLEMS: 65,831 "missing" windows alongside only 355 saved windows, and 12 unreadable lines. I did not investigate the cause. I reran that recording straight away (`ab2_quiet_20261111_132003.csv`, verdict OK). The plan did not cover reruns. The rerun was triggered by the capture tool's verdict, before any rms was computed. The analysis script excludes recordings whose verdict is not OK, as committed, so the failed file was left out automatically: 9 recordings were read and 8 were used. The failed file is kept in the repo.

Several recordings reported unreadable lines before the first good window (old bytes from the serial port when it opened). No windows were lost: ab1 order 4 reported 3,124, and ab2 orders 2, 3 and 4 reported 728, 93 and 268.

## Observation, no cause claimed

`hf` was about 95 in block 1 and about 101 to 102 in block 2, and `mean` was about 8373 to 8381 in block 1 and about 8413 to 8416 in block 2. The rule did not test these and I have no explanation for the shift.

## What this does and does not show

- It shows that, for one person, one board, one room and two sittings, plugged versus battery did not change the quiet rms under the pre-registered rule.
- It does not explain the earlier 140 level or the 82 to 87 levels. These two sittings sat at about 98 to 105, which is neither. The sitting-to-sitting variation is still unexplained.
- It says nothing about other boards, rooms or laptops.

## AI assistance

The plan, the analysis script and its tests were written with Claude's help (AI-assisted). I chose the question, ran every recording myself on my own board, and the predictions were committed before I saw any result. The result was not discussed or reinterpreted after the rule ran.

## Files

Recordings are in `data/own/mic/`: `ab1_quiet_20261111_093923.csv`, `ab1_quiet_20261111_094347.csv`, `ab1_quiet_20261111_094603.csv`, `ab1_quiet_20261111_095017.csv`, `ab2_quiet_20261111_131804.csv` (failed, excluded), `ab2_quiet_20261111_132003.csv`, `ab2_quiet_20261111_132401.csv`, `ab2_quiet_20261111_132642.csv`, `ab2_quiet_20261111_133036.csv`.
