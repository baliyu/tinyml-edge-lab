# Components of tinyml-edge-lab

Two diagrams, drawn from the README table "What is here" and the C headers in `docs/API.md`. A box with a dashed border is drawn from file and folder names only: I have not read what is inside.

## 1. The parts and what feeds what

```mermaid
flowchart LR
    subgraph DATA["data/"]
        RAW["raw: UCI HAR and CWRU downloads"]
        OWN["own: recordings from the Edge"]
        SYN["synthetic: test data in the real file layout"]
    end
    subgraph PKG["src/tinyml_lab (Python on the PC)"]
        HAR["uci_har, features, har_baseline, har_compare"]
        CW["cwru*.py: bearing pipeline"]
        EG["edge_gestures, edge_gesture_eval, gesture_demo"]
        EXP["export_c: a model as a C function"]
        FOOT["footprint*.py: flash and RAM sizes"]
        LIVE["live_accuracy*.py: scores a live run"]
    end
    subgraph TOOLS["tools/ (Python over serial)"]
        MON["edge_gesture_monitor"]
        LT["live_accuracy_test"]
        MIC["mic_*.py"]
    end
    subgraph FW["firmware/sparkfun_edge (C)"]
        GF["gesture_features"]
        GD["gesture_demo: the on-device chain"]
        MF["mic_features and mic_logger"]
    end
    subgraph LQ["litert_quant/ (names only)"]
        LQS["scripts and 7 .tflite models"]
        LQF["firmware: digits_bench, har_bench, cwru_bench"]
    end
    TESTS["tests/, run in CI"]
    RAW --> HAR
    RAW --> CW
    OWN --> EG
    SYN --> TESTS
    HAR --> EXP
    EXP --> FOOT
    EG -->|"trains and exports the model"| GD
    GF --> GD
    GD -->|"gesture output over serial"| MON
    GD -->|"predictions over serial"| LT
    LT --> LIVE
    MF -->|"summary line every 0.25 s"| MIC
    LQS --> LQF
    classDef nameonly stroke-dasharray: 5 5
    class LQS,LQF nameonly
```

## 2. One desk gesture, from recording to a scored live run

```mermaid
flowchart TD
    A["Record sittings from the Edge over serial: tool names only, not read"] --> B["data/own: the Edge's own recordings"]
    B --> C["edge_gestures: loader and resampling (PC)"]
    C --> D["edge_gesture_eval: session-wise evaluation (PC)"]
    C --> E["gesture_demo: train and export the deployable model (PC)"]
    E --> F["C chain: resample to 50 Hz, window, features, model, checked against the Python chain on the PC"]
    F --> G["Runs on the SparkFun Edge (README)"]
    G -->|"serial"| H["edge_gesture_monitor: lost lines, overruns, slow windows"]
    G -->|"serial"| I["live_accuracy_test: shuffled gestures against predictions"]
    I --> J["live_accuracy*.py: windows that count, confusion matrix, accuracy, macro-F1"]
    classDef nameonly stroke-dasharray: 5 5
    class A nameonly
```

## Where the boxes come from

| Box | What the README says |
|---|---|
| gesture_features | The gesture features in C (1.5 KB of flash), checked against the Python features |
| gesture_demo (firmware) | The whole on-device chain in C: resample to 50 Hz, window, features, model. Checked against the Python chain on the PC and running on the Edge |
| gesture_demo.py | Trains and exports the deployable model, builds the C chain for the PC and compares it with the Python chain |
| edge_gesture_monitor | Watches the board's gesture output over serial and points out lost lines, sensor overruns and slow windows |
| live_accuracy_test and live_accuracy*.py | Gestures in a known, shuffled order while the board predicts, then a score with an interval that resamples whole blocks |
| mic_logger and mic_features | A microphone logger and the summary features in C. The README reports a mixed result: quiet detection fails when the room's noise level changes between sittings |
| tests/ | Run in CI with synthetic data in the real UCI file layout. They prove the code paths, not model quality |

## Not drawn

- What is inside `litert_quant/`. Its README has not been read for this diagram, so those boxes are dashed.
- The recording tools and `imu_logger`, `mic_probe`, `evidence` and `scripts` under `firmware/sparkfun_edge`. Names only.
- `firmware/nano33ble`, which holds a README and no code.
- The roadmap items (a bridge service, an AI layer, cloud). Not built here.
