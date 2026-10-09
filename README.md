# tinyml-edge-lab

Small, tested building blocks for edge machine learning on my own boards (SparkFun Edge, Arduino Nano 33 BLE).
The aim is a pipeline I can explain end to end: data, features, a baseline, an exported model, and checks that
the device matches the PC.

## Honest status
* The code and its tests run in CI (`pytest`). The tests use **synthetic data in the real UCI file layout**; they
  prove the code paths, not model quality.
* **No results are published here yet.** The table below is filled only after running on the real dataset.
* The decision-tree exporter is tested by compiling the generated C with `gcc` and checking it returns the same
  class as scikit-learn on random inputs. It has not yet been run on a microcontroller.

## What is here
| Part | What it does |
|---|---|
| `src/tinyml_lab/features.py` | Window features (mean, std, rms, min, max, dominant frequency, band energies), cheap enough for a device |
| `src/tinyml_lab/uci_har.py` | Loader for the official UCI HAR data; refuses to run if a volunteer is in both splits |
| `src/tinyml_lab/export_c.py` | Exports a scikit-learn decision tree as a C function |
| `src/tinyml_lab/har_baseline.py` | Decision-tree baseline, depth sweep, per-class recall, confusion matrix |
| `firmware/` | Notes for the SparkFun Edge and the Nano 33 BLE |
| `docs/DATASETS.md` | Sources, licences, evaluation rules |

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
python scripts/download_uci_har.py                     # official UCI zip, CC BY 4.0
python scripts/train_har_baseline.py --depth 6
```

## Results (fill in after running on real data)
| Model | Nodes | Test accuracy | Macro-F1 | Notes |
|---|---|---|---|---|
| Decision tree, depth 6 | _tbd_ | _tbd_ | _tbd_ | UCI's subject-disjoint test split |

## Roadmap
1. SparkFun Edge: hello world and wake-word from the TinyML book (log what worked).
2. Nano 33 BLE: IMU data logger with microSD and a TFT user interface; collect my own labelled data.
3. Compare a decision tree with a small neural network (int8), reporting size, RAM and latency.
4. Check on-device accuracy against the PC on a held-out set.
5. Signed model updates (reusing the secure-boot work).

## Licence
MIT for the code. Datasets keep their own licences (see `docs/DATASETS.md`).
