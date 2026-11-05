# Does the laptop's power state change the quiet level? A plan written before recording

Built with AI assistance (Claude). Written on 6 November 2026, before any of the recordings it describes exists. Nothing in it has been tested.

## The question

In the held-out test of the microphone classifier (`firmware/sparkfun_edge/mic_logger/README.md`), quiet detection failed whenever the test sitting's quiet level differed from the training sittings. Quiet `rms` was about 140 in sitting A and about 82 to 87 in later recordings, and I suspected the laptop's power state, but I have one 20 s recording per condition and no controlled comparison.
**Does the quiet level of the Edge's microphone depend on whether the laptop supplying its USB power is plugged in or running on its battery?**

## Competing explanations

1. **Power state** (the hypothesis): the supply noise on USB changes with the charger, and the microphone's ADC reading changes with it.
2. **The room or the time of day:** an acoustic difference I did not control (a fan, the house, the street).
3. **Drift:** the level changes from hour to hour or day to day for reasons unrelated to both.

## Design

Quiet recordings only, 90 s each, the same board in the same place (flat on the desk, microphone up, position marked), the same USB port, the same room, nothing running that makes a sound (no console, no TV, nothing near the desk), and no touching the desk during a recording.
One **block** is four recordings in the order **plugged, battery, battery, plugged**, so that a slow drift cancels in the comparison. Wait 60 seconds after plugging or unplugging before recording, and note the battery percentage. Two blocks, on two different days or at least three hours apart (`ab1` and `ab2`).
Each recording's note carries the tags `power=plugged block=1 order=1` and so on, so the analysis cannot depend on my memory.

```
py tools\mic_capture.py --port COM4 --label quiet --seconds 90 --session ab1 --note "power=plugged block=1 order=1"
py tools\mic_capture.py --port COM4 --label quiet --seconds 90 --session ab1 --note "power=battery block=1 order=2"
py tools\mic_capture.py --port COM4 --label quiet --seconds 90 --session ab1 --note "power=battery block=1 order=3"
py tools\mic_capture.py --port COM4 --label quiet --seconds 90 --session ab1 --note "power=plugged block=1 order=4"
```

The same four for `ab2` with `block=2`. Then `python3 tools/mic_quiet_ab.py --sessions ab1,ab2`.

## Predictions (written before recording)

| # | Prediction | Basis | What would prove it wrong |
|---|---|---|---|
| P1 | The plugged recordings have a median `rms` at least 1.20 times the battery recordings in both blocks | about 140 plugged against 82 to 107 on battery earlier, from different days and not controlled | a block ratio below 1.20, or the opposite direction |
| P2 | Two recordings of the same condition in one block agree to within 10% of their mean | quiet was stable within a sitting (the three recordings of sitting A agreed closely) | a spread above 10% |
| P3 | `peak` and `hf` also move in the same direction (plugged higher), `peak` by at least 20% and `hf` by at least 5% | sitting A had `peak` about 720 and `hf` 108 to 112 against 368 and 97 to 101 in sitting b | no movement in `peak` or `hf` while `rms` moves |

P3 is reported but is not part of the decision rule.

## The decision rule (fixed now; `tools/mic_quiet_ab.py` applies it and a test keeps these numbers equal to the ones in the code)

For each block take the median `rms` of the valid windows of each recording, the mean of the two plugged recordings and of the two battery recordings, and their ratio (plugged over battery). A recording counts only if its capture verdict is OK, it has at least 280 valid windows (70 seconds) and its note has the power, block and order tags.

- **Supported** if there are at least 2 complete blocks, in every block the two recordings of each condition agree to within 10%, and in every block the ratio is at least 1.20.
- **Not supported** if there are at least 2 complete blocks, they agree within 10%, and in every block the ratio is between 0.90 and 1.10.
- **Inconclusive** in every other case, including a mix of directions, ratios between 1.10 and 1.20, or recordings that do not agree with themselves.
- **Not enough blocks** with fewer than 2 complete blocks.

## What I will do with each outcome

- **Supported:** the next test is the calibration fix on a fresh sitting recorded under a stated power state, and the README says the cause is supported by a controlled comparison in one room.
- **Not supported:** the power state is not the cause on this setup; I would record repeated sittings with the power state fixed to look at the room and the time of day. The README says what was ruled out.
- **Inconclusive:** I say so, and I do not choose an explanation.

## Limits

One person, one board, one room, two sittings, 16 recordings at most. A controlled comparison of power state does not separate every possible cause of a change in supply noise (the charger, the cable and the port are one bundle).
The decision rule is deliberately simple; it is not a statistical test, and with two blocks a "supported" result is a strong hint, not a proof.
