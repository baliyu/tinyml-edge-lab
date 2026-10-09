# Datasets: sources, licences and what to download

Always download from the **original publisher** and keep data out of Git. Third-party copies (for example on
Kaggle) can be a different version, can be pre-processed, and often carry a licence the uploader chose, not the
original owner. Check them before you rely on them.

| Dataset | Official source | Licence | Size / notes | Used for |
|---|---|---|---|---|
| UCI HAR (ID 240) | https://archive.ics.uci.edu/dataset/240 | CC BY 4.0 | 58.2 MB zip; 30 volunteers; 50 Hz; UCI's split is by volunteer (70 % / 30 %) | Windowing, features, baseline, subject-wise evaluation |
| Google Speech Commands v0.02 | TensorFlow Datasets catalog (`speech_commands`) or Hugging Face `google/speech_commands` | CC BY 4.0 | 16 kHz audio; v0.02 splits: 84,848 train / 9,982 validation / 4,890 test | Keyword spotting (the TinyML book's wake-word chapters train on this family of data) |
| CWRU bearing data | Case Western Reserve University Bearing Data Center (download page: https://csegroups.case.edu/bearingdatacenter/pages/download-data-file) | Not stated on the portal | `.mat` files; 12 kHz and 48 kHz; loads 0 to 3 hp | FFT features and fault classification on a PC (not deployable at a 400 Hz IMU rate) |

## MLPerf Tiny (yardstick for model size, latency and accuracy)
From the MLCommons MLPerf Tiny page (v1.4 supplemental): image classification (CIFAR-10, ResNet, 85 % top-1),
person detection (COCO, MobileNet, 80 % top-1), keyword spotting (Speech Commands, DS-CNN, 90 % top-1),
anomaly detection (ADMOS toy car, dense model, 0.85 AUC) and streaming wake word (custom, 1D DS-CNN).

## Evaluation rules for this repo
1. Split by **person, file or machine condition**, never by random windows.
2. Report accuracy **and** per-class recall and the confusion matrix.
3. Record the dataset checksum printed by the download script.
4. State the sampling rate. Features computed at 50 Hz are not comparable with 400 Hz or 12 kHz data.
