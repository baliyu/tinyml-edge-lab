"""Download and unpack the official UCI HAR zip. Data is never committed to Git."""
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

UCI_HAR_URLS = [
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
]


def sha256_of(path):
    """The SHA-256 of a file, as hex, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(urls, dest_file):
    """Try each URL in turn; return the URL that worked."""
    dest_file = Path(dest_file)
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(dest_file, "wb") as f:
                shutil.copyfileobj(r, f)
            return url
        except Exception as e:  # noqa: BLE001 - report and try the next mirror
            last = e
    raise RuntimeError(f"all download URLs failed (last error: {last})")


def extract_uci_har(zip_path, dest_dir):
    """Unzip, handling the case where the zip contains another 'UCI HAR Dataset.zip'."""
    zip_path, dest_dir = Path(zip_path), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    for inner in dest_dir.glob("*.zip"):
        with zipfile.ZipFile(inner) as z:
            z.extractall(dest_dir)
    return dest_dir
