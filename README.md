# tinyml-edge-lab
![CI](https://github.com/baliyu/tinyml-edge-lab/actions/workflows/ci.yml/badge.svg)
   
Small, tested building blocks for edge machine learning on my own boards (SparkFun Edge, Arduino Nano 33 BLE).
The aim is a pipeline I can explain end to end: data, features, a baseline, an exported model, and checks that
the device matches the PC.

## Honest status
* The code and its tests run in CI (`pytest`). The tests use **synthetic data in the real UCI file layout**; they
  prove the code paths, not model quality.
* The results table below comes from **one run on the official UCI HAR data** (zip checksum in `docs/DATASETS.md`). It is a first baseline, not a tuned model: the depth was fixed before running, not picked on the test set.
* The decision-tree exporter is tested by compiling the generated C with `gcc` and checking it returns the same
  class as scikit-learn on random inputs. It has not yet been run on a microcontroller.

## What is here
| Part | What it does |
|---|---|
| `src/tinyml_lab/features.py` | Window features (mean, std, rms, min, max, dominant frequency, band energies), cheap enough for a device |
| `src/tinyml_lab/uci_har.py` | Loader for the official UCI HAR data; refuses to run if a volunteer is in both splits |
| `src/tinyml_lab/export_c.py` | Exports a scikit-learn decision tree as a C function |
| `src/tinyml_lab/har_baseline.py` | Decision-tree baseline, depth sweep, per-class recall, confusion matrix |
| `src/tinyml_lab/plots.py` | Figures drawn from the saved results: confusion matrix, per-class recall, depth sweep, example windows, most-used features |
| `src/tinyml_lab/har_compare.py` | Subject-grouped model selection, richer features and a random-forest comparison |
| `src/tinyml_lab/cwru*.py` | CWRU bearing pipeline: verified download, windowing, features and honest evaluation |
| `firmware/` | Notes for the SparkFun Edge and the Nano 33 BLE |
| `docs/DATASETS.md` | Sources, licences, evaluation rules |

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
python scripts/download_uci_har.py                     # official UCI zip, CC BY 4.0
python scripts/train_har_baseline.py --depth 6
python scripts/make_figures.py                         # figures into results/figures/
python scripts/compare_models.py                       # grouped CV, richer features, forest comparison
python scripts/download_cwru.py && python scripts/run_cwru.py   # CWRU bearings (about 300 MB download)
```

## Results (official UCI HAR data, run on 9 October 2026)
| Model | Nodes | Test accuracy | Macro-F1 | Notes |
|---|---|---|---|---|
| Decision tree, depth 6 | 61 | 0.797 | 0.790 | UCI's subject-disjoint test split; features from total acceleration only (no gyroscope) |

Per-class recall: walking 0.909, walking upstairs 0.715, walking downstairs 0.555, sitting 0.851, standing 0.703, laying 1.000. The weakest classes are walking downstairs and standing. This is a baseline to improve, not a strong result.

### Figures

Drawn by `scripts/make_figures.py` from the run above (`results/har_baseline.json` and the raw test windows).

![Confusion matrix](results/figures/confusion_matrix.png)
![Per-class recall](results/figures/per_class_recall.png)
![Accuracy and size against depth](results/figures/depth_sweep.png)
![One example window per activity](results/figures/class_examples.png)
![Most-used features](results/figures/top_features.png)

The depth plot shows test scores for information only. The depth was fixed before running and was not chosen from this plot.

<!-- compare:start -->
### Model selection and richer features

How the models were chosen:
- Depth (and forest size) came from 5-fold cross-validation on the 21 training volunteers only, with folds split by volunteer. The smallest model within one standard error of the best score was kept.
- The 9 test volunteers were used once per final model, after the choice was made.
- Extended features add UCI's body acceleration, gyroscope and gravity signals to the total-acceleration features. On a device the body/gravity separation would need a filter, which is not implemented here.

| Model | Features | Choice | Nodes | CV macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---|---|---|---|---|
| Tree, depth 6 fixed in advance | baseline | fixed depth 6 | 61 | 0.816 (SE 0.034) | 0.797 | 0.790 |
| Tree, depth chosen by CV | baseline | depth 4 (CV) | 17 | 0.819 (SE 0.032) | 0.810 | 0.807 |
| Tree, depth chosen by CV | extended | depth 5 (CV) | 33 | 0.833 (SE 0.014) | 0.839 | 0.834 |
| Random forest, size chosen by CV | extended | 30 trees, depth 6 (CV) | 2186 | 0.883 (SE 0.026) | 0.879 | 0.877 |

Adding the extra features moved the compact tree's test macro-F1 from 0.807 to 0.834 (33 nodes against 17). The random forest scored 0.877 with 2186 nodes in total. The extended tree's weakest class is walking downstairs (recall 0.669).

These are single runs with 9 test volunteers, so differences of a few points should not be over-read. The forest was not exported to C, so its size is shown in nodes only.

![Choosing the depth](results/figures/cv_curves.png)
![Size against test macro-F1](results/figures/model_comparison.png)
![Confusion matrix of the extended tree](results/figures/confusion_matrix_extended_tree.png)
<!-- compare:end -->

<!-- cwru:start -->
### Bearing faults (CWRU, run on a PC)

- Data: 40 official recordings (normal, and drive-end inner-race, ball and outer-race faults at 0.007, 0.014, 0.021 in, at motor loads 0, 1, 2, 3 hp), all at 48 kHz so every class was recorded the same way. Each file is checked against its SHA-256 value before use.
- 4096-sample windows, five time-domain statistics and 12 spectral band fractions as features. A decision tree (depth 8) and a random forest (100 trees), settings fixed in advance and not tuned.

| Evaluation | Tree accuracy / macro-F1 | Forest accuracy / macro-F1 |
|---|---|---|
| Random split of windows (leaky) | 0.944 / 0.944 | 0.989 / 0.989 |
| Leave one motor load out | 0.911 / 0.909 | 0.946 / 0.943 |
| Leave one fault size out (3 fault types, no normal class) | 0.362 / 0.354 | 0.329 / 0.332 |
| Leave one motor load out, signals reduced to 400 Hz | 0.495 / 0.492 | 0.602 / 0.598 |

For the forest, a random split of windows scored macro-F1 0.989; holding out a whole motor load scored 0.943; holding out a whole fault size (three fault types, no normal class) scored 0.332. After reducing the same recordings to 400 Hz, leave-one-load-out scored 0.598.

Forest accuracy on each held-out fold (an average can hide a bad fold; a score below chance means the model was consistently wrong there):
- leave one motor load out: 0 hp 0.818, 1 hp 0.989, 2 hp 0.986, 3 hp 0.923
- leave one fault size out (chance is 0.333): 0.007 in 0.525, 0.014 in 0.353, 0.021 in 0.112

How to read it:
- Windows cut from the same recording are close neighbours, so a random split can let the model recognise the recording instead of the fault. That row is shown to expose the risk, not as a result.
- Every damaged bearing was recorded at all four loads, so leaving a load out is easier than meeting a new bearing. Leaving a fault size out is the harder test, because each size is a different damaged bearing.
- The 400 Hz row decimates the same recordings with an anti-alias filter. It shows what a 400 Hz sensor could see of these signals; it says nothing about any real sensor or machine.
- CWRU is a small laboratory benchmark with seeded faults. These scores do not predict field performance.

![Protocols compared](results/figures/cwru_protocols.png)
![Vibration signatures](results/figures/cwru_signatures.png)
![Confusion matrix, forest, leave one load out](results/figures/cwru_confusion_forest.png)
<!-- cwru:end -->

## Roadmap
1. SparkFun Edge: hello world and wake-word from the TinyML book (log what worked).
2. Nano 33 BLE: IMU data logger with microSD and a TFT user interface; collect my own labelled data.
3. Compare a decision tree with a small neural network (int8), reporting size, RAM and latency.
4. Check on-device accuracy against the PC on a held-out set.
5. Signed model updates (reusing the secure-boot work).

## Licence
MIT for the code. Datasets keep their own licences (see `docs/DATASETS.md`).
