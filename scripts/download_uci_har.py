#!/usr/bin/env python3
"""Download the official UCI HAR dataset into data/raw/ (never committed to Git)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tinyml_lab.download import UCI_HAR_URLS, download, extract_uci_har, sha256_of  # noqa: E402
from tinyml_lab.uci_har import find_root  # noqa: E402

raw = Path(__file__).resolve().parents[1] / "data" / "raw"
zip_path = raw / "uci_har.zip"
url = download(UCI_HAR_URLS, zip_path)
print(f"downloaded from {url}")
print(f"size   : {zip_path.stat().st_size / 1e6:.1f} MB")
print(f"sha256 : {sha256_of(zip_path)}   (record this in docs/DATASETS.md)")
extract_uci_har(zip_path, raw)
print("unpacked to", find_root(raw))
