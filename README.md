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

## Roadmap
1. SparkFun Edge: hello world and wake-word from the TinyML book (log what worked).
2. Nano 33 BLE: IMU data logger with microSD and a TFT user interface; collect my own labelled data.
3. Compare a decision tree with a small neural network (int8), reporting size, RAM and latency.
4. Check on-device accuracy against the PC on a held-out set.
5. Signed model updates (reusing the secure-boot work).

## Licence
MIT for the code. Datasets keep their own licences (see `docs/DATASETS.md`).
