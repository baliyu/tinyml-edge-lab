"""CWRU bearing data: the files used here, checked loading, windowing and a verified downloader.

The manifest lists the 40 official recordings used (all at 48 kHz, so every class was recorded the same way):
the four normal baseline files and the drive-end inner-race (IR), ball (B) and outer-race (OR, centred at 6 o'clock)
faults at 0.007, 0.014 and 0.021 inch, each at motor loads 0 to 3 hp. It was generated from the bearing-datasets
package's copy of the four official CWRU index pages; the byte sizes and SHA-256 values are the ones that package
recorded for the files on the official site. Files are checked against them before use.
"""
import hashlib
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

FS = 48000.0
BASE_URL = "https://engineering.case.edu/sites/default/files/"
CLASS_ORDER = ["normal", "IR007", "IR014", "IR021", "B007", "B014", "B021", "OR007", "OR014", "OR021"]
KIND_NAMES = {"normal": "normal", "IR": "inner race", "B": "ball", "OR": "outer race"}

MANIFEST = [
    {"number": 97, "cls": "normal", "kind": "normal", "size_in": 0.0, "load_hp": 0, "rpm": 1797, "bytes": 3903344, "sha256": "16bf48babcf1c7ac224bc1a81cd9eafdb27e42d5cf559761907e067e8eeadf3c"},
    {"number": 98, "cls": "normal", "kind": "normal", "size_in": 0.0, "load_hp": 1, "rpm": 1772, "bytes": 7742720, "sha256": "37e6612c05e65c415dcfa2ab27a3fda648a5863160fa898b884a14743044e045"},
    {"number": 99, "cls": "normal", "kind": "normal", "size_in": 0.0, "load_hp": 2, "rpm": 1750, "bytes": 15503928, "sha256": "4b97e6b5361f45efb6951dc3b1aebcdb3b89cb69d0f96d6f5c297dd9f45eee75"},
    {"number": 100, "cls": "normal", "kind": "normal", "size_in": 0.0, "load_hp": 3, "rpm": 1730, "bytes": 7770624, "sha256": "88a5990cb541320e91505a1d72139e1993500ffe6e292a451011667f4138ca78"},
    {"number": 122, "cls": "B007", "kind": "B", "size_in": 0.007, "load_hp": 0, "rpm": 1797, "bytes": 3916160, "sha256": "f392129bf3b43360b964d4e5b3479d0797d1647af8359934b0f56b9abab4739a"},
    {"number": 123, "cls": "B007", "kind": "B", "size_in": 0.007, "load_hp": 1, "rpm": 1772, "bytes": 7798480, "sha256": "a1932ec4bad2460238088ca523fdcec6498e562e91f86255863d3987d9fcd558"},
    {"number": 124, "cls": "B007", "kind": "B", "size_in": 0.007, "load_hp": 2, "rpm": 1750, "bytes": 7789200, "sha256": "88719243f1b7dbe795c108acaf67ca0115c51517ac35a3fe3460eb07917b2a9a"},
    {"number": 125, "cls": "B007", "kind": "B", "size_in": 0.007, "load_hp": 3, "rpm": 1730, "bytes": 7817056, "sha256": "2607d0c291ed9a6138aed3eed7015e133f1cf866ac5cdf2251921f08d058f362"},
    {"number": 189, "cls": "B014", "kind": "B", "size_in": 0.014, "load_hp": 0, "rpm": 1797, "bytes": 3986672, "sha256": "6fd84310f781e1ab54a9a746b8626c73a863f2b02cea4c86229ca2444cb20a9a"},
    {"number": 190, "cls": "B014", "kind": "B", "size_in": 0.014, "load_hp": 1, "rpm": 1772, "bytes": 7779920, "sha256": "ae5fb00ce1eeefa5e21013197b186c0496f52820bdff00b385c58dfcb64a5b2f"},
    {"number": 191, "cls": "B014", "kind": "B", "size_in": 0.014, "load_hp": 2, "rpm": 1750, "bytes": 7798480, "sha256": "bb39df077745821a3917e432ab1bc2f5d7c4e01a6e2338d1271b6994ec68dc7f"},
    {"number": 192, "cls": "B014", "kind": "B", "size_in": 0.014, "load_hp": 3, "rpm": 1730, "bytes": 7789200, "sha256": "099b7f3399d0ea81aedfab20667a9517059441c10f2a9d52130687cc9643e2c7"},
    {"number": 226, "cls": "B021", "kind": "B", "size_in": 0.021, "load_hp": 0, "rpm": 1797, "bytes": 3903344, "sha256": "fc78f08085760f2e17bcf7efe486b3a278e108093cd184acf278daecad6765e4"},
    {"number": 227, "cls": "B021", "kind": "B", "size_in": 0.021, "load_hp": 1, "rpm": 1772, "bytes": 7789200, "sha256": "c5c7aa64243ecd1428999aae4be29ba5d4de9235fee9731352c6cce7de604675"},
    {"number": 228, "cls": "B021", "kind": "B", "size_in": 0.021, "load_hp": 2, "rpm": 1750, "bytes": 7798480, "sha256": "ea819eb0404f76ccdbe6074463469bda8fbbe04b7c0c75cf1443d2b5ac3484f6"},
    {"number": 229, "cls": "B021", "kind": "B", "size_in": 0.021, "load_hp": 3, "rpm": 1730, "bytes": 7789200, "sha256": "4fc20e3eeef375dc1a8ac577f04b4b0e0fa9e319ec51a61f7a02e756b0069499"},
    {"number": 109, "cls": "IR007", "kind": "IR", "size_in": 0.007, "load_hp": 0, "rpm": 1797, "bytes": 3903344, "sha256": "daddef5f784879becdf8fefbe2453f11c3cfc8d061a2a671da2f546d1bb48460"},
    {"number": 110, "cls": "IR007", "kind": "IR", "size_in": 0.007, "load_hp": 1, "rpm": 1772, "bytes": 7779920, "sha256": "9e2bc579af6f4e6d9d26bf63b7ed84ecba7775740d56a3737cc8f04f73c22d1a"},
    {"number": 111, "cls": "IR007", "kind": "IR", "size_in": 0.007, "load_hp": 2, "rpm": 1750, "bytes": 7770624, "sha256": "a2e64a397a990efb91ba55252a84796f755bda3eb097c4b47cc0730a08dab309"},
    {"number": 112, "cls": "IR007", "kind": "IR", "size_in": 0.007, "load_hp": 3, "rpm": 1730, "bytes": 7770624, "sha256": "1775db11999f268c6ccf04b3ee19857f90a3636f4a582f1abd9a627f991e72d4"},
    {"number": 174, "cls": "IR014", "kind": "IR", "size_in": 0.014, "load_hp": 0, "rpm": 1797, "bytes": 1020944, "sha256": "ca7113a900d9217aa5632e0dad4f8791caed4d314321daa011719e1bc99f3026"},
    {"number": 175, "cls": "IR014", "kind": "IR", "size_in": 0.014, "load_hp": 1, "rpm": 1772, "bytes": 17849704, "sha256": "18302e96425ad163db699da3a2ca63d8f6ef4dfc99172e97ab21a8a77a83ac17"},
    {"number": 176, "cls": "IR014", "kind": "IR", "size_in": 0.014, "load_hp": 2, "rpm": 1750, "bytes": 7807760, "sha256": "569c7b292018e7e88cf4b375b091433c7566b9503545188979e93f6428427247"},
    {"number": 177, "cls": "IR014", "kind": "IR", "size_in": 0.014, "load_hp": 3, "rpm": 1730, "bytes": 7761344, "sha256": "5cffc5aa2b1e5cc65a97ebcfaa369b886ad8db8b8840e0e0095e97b8b9be4c16"},
    {"number": 213, "cls": "IR021", "kind": "IR", "size_in": 0.021, "load_hp": 0, "rpm": 1797, "bytes": 3909760, "sha256": "6afb81f36a9efcdd370a52166d71c13cb42278c7de22752c81d4b2a22fa133f0"},
    {"number": 214, "cls": "IR021", "kind": "IR", "size_in": 0.021, "load_hp": 1, "rpm": 1772, "bytes": 7761344, "sha256": "a8fe29f358b7f9c140f38e0cd7fbafdd7441c44e30a5b704ecb9be64cbb841ed"},
    {"number": 215, "cls": "IR021", "kind": "IR", "size_in": 0.021, "load_hp": 2, "rpm": 1750, "bytes": 7863472, "sha256": "1e3d65088782a29aa70004896db4b9836499dd12edd3c58fb496f4befca6f746"},
    {"number": 217, "cls": "IR021", "kind": "IR", "size_in": 0.021, "load_hp": 3, "rpm": 1730, "bytes": 15689680, "sha256": "fb5b067c04b4c8449cbdd4ac4e376d7eb0804b9203efcab513e42aaf2f6a2cf7"},
    {"number": 135, "cls": "OR007", "kind": "OR", "size_in": 0.007, "load_hp": 0, "rpm": 1797, "bytes": 3896944, "sha256": "5a1c3ceec1f2b58af9e01051e425aa9d31dd018f58a8d31cda25bbdb1ac6dca9"},
    {"number": 136, "cls": "OR007", "kind": "OR", "size_in": 0.007, "load_hp": 1, "rpm": 1772, "bytes": 7789200, "sha256": "37f259eebbc92afa8bd1b445d02af14e83608a48cd7a3cf5c323515ddc2d0f24"},
    {"number": 137, "cls": "OR007", "kind": "OR", "size_in": 0.007, "load_hp": 2, "rpm": 1750, "bytes": 7789200, "sha256": "a81b621153e426aa32848be8157c42bcd81065a28ac7655a9a6597f3889fa3ef"},
    {"number": 138, "cls": "OR007", "kind": "OR", "size_in": 0.007, "load_hp": 3, "rpm": 1730, "bytes": 7807760, "sha256": "b43eeb67badeb129047ddd3658bf673815dc4bd445facb4217eb04653bb2e11d"},
    {"number": 201, "cls": "OR014", "kind": "OR", "size_in": 0.014, "load_hp": 0, "rpm": 1797, "bytes": 3922576, "sha256": "6c27615a61f19f05f65cf2cc2aa4bca5bda051c9d93a708d0ee8baf5138834fa"},
    {"number": 202, "cls": "OR014", "kind": "OR", "size_in": 0.014, "load_hp": 1, "rpm": 1772, "bytes": 7752064, "sha256": "16f1a7eadbd6703a7a1ab1a32f8d7b97b6b85c1b31b1e760c123a9e175d8ae63"},
    {"number": 203, "cls": "OR014", "kind": "OR", "size_in": 0.014, "load_hp": 2, "rpm": 1750, "bytes": 7789200, "sha256": "3cad2848f4c96cde04fc06ca0f541ad663298ecb50eb7fc74e25b72ab29e1c5a"},
    {"number": 204, "cls": "OR014", "kind": "OR", "size_in": 0.014, "load_hp": 3, "rpm": 1730, "bytes": 7817056, "sha256": "f3b1736698a580c2ea44cce78045833a9e76476d9153da8af5275c2e30e38a10"},
    {"number": 238, "cls": "OR021", "kind": "OR", "size_in": 0.021, "load_hp": 0, "rpm": 1797, "bytes": 3941808, "sha256": "63209a4057f199beb42beedf5b4481987fdb3ac746c46777dd385d689bc481df"},
    {"number": 239, "cls": "OR021", "kind": "OR", "size_in": 0.021, "load_hp": 1, "rpm": 1772, "bytes": 7826336, "sha256": "4ba7dc55258359577a68f6de5c0ff6675776d2b2c30d691dd4d71640e34834af"},
    {"number": 240, "cls": "OR021", "kind": "OR", "size_in": 0.021, "load_hp": 2, "rpm": 1750, "bytes": 7807760, "sha256": "c5d8b4db09880b660da88865fff0e81afef0f5ad68bdf0ad7b75c722ae3e7e1c"},
    {"number": 241, "cls": "OR021", "kind": "OR", "size_in": 0.021, "load_hp": 3, "rpm": 1730, "bytes": 7826336, "sha256": "cb64eed96d401837c25df63598b7d1a00868368e1ed503a8dcef1e5226ec9caa"},
]


def file_name(entry):
    """The file name of a manifest entry: '<number>.mat'."""
    return f"{entry['number']}.mat"


def sha256_file(path):
    """The SHA-256 of a file, as hex, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path, entry, check_sha=True):
    """Return None if the file matches the manifest, otherwise a short reason."""
    path = Path(path)
    if not path.is_file():
        return "missing"
    if path.stat().st_size != entry["bytes"]:
        return f"wrong size ({path.stat().st_size} bytes, expected {entry['bytes']})"
    if check_sha and sha256_file(path) != entry["sha256"]:
        return "wrong SHA-256 (the file is corrupted or has been changed)"
    return None


def check_dir(data_dir, manifest=MANIFEST, check_sha=True):
    """List (number, reason) for every manifest file that is missing or does not match."""
    return [(e["number"], r) for e in manifest
            if (r := verify_file(Path(data_dir) / file_name(e), e, check_sha)) is not None]


def _fetch(url, dest):
    """Fetch url into dest, continuing a partial file if dest already holds the start of it."""
    dest = Path(dest)
    curl = shutil.which("curl")
    if curl:
        r = subprocess.run([curl, "-L", "-C", "-", "--fail", "--silent", "--show-error", "--connect-timeout", "20",
                            "--max-time", "900", "-A", "Mozilla/5.0", "-o", str(dest), url], capture_output=True, text=True)
        if r.returncode != 0:
            if r.returncode == 33:          # the server cannot resume: start from scratch next time
                dest.unlink(missing_ok=True)
            raise RuntimeError(r.stderr.strip() or f"curl exit code {r.returncode}")
    else:
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as resp:
            mode = "ab" if have and resp.status == 206 else "wb"
            with open(dest, mode) as f:
                shutil.copyfileobj(resp, f)


def download_all(data_dir, manifest=MANIFEST, base_url=BASE_URL, log=print, check_sha=True, max_consecutive_failures=3,
                 attempts=5, retry_wait=2.0):
    """Download every missing file, verify it, and only then keep it.

    A dropped transfer is resumed from where it stopped (the partial file is kept as <number>.part) and each file gets
    `attempts` tries. A file that arrives complete but fails its checksum is thrown away and fetched again from scratch.
    Returns {'ok', 'skipped', 'failed', 'not_attempted'}. After `max_consecutive_failures` files in a row have failed
    every attempt it stops (the site is probably unreachable), instead of trying every remaining file."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out = {"ok": 0, "skipped": 0, "failed": [], "not_attempted": 0}
    streak = 0
    for i, e in enumerate(manifest, 1):
        if streak >= max_consecutive_failures:
            out["not_attempted"] += 1
            continue
        dest = data_dir / file_name(e)
        if verify_file(dest, e, check_sha) is None:
            out["skipped"] += 1
            continue
        part = dest.with_suffix(".part")
        log(f"[{i}/{len(manifest)}] downloading {file_name(e)} ({e['bytes'] / 1e6:.1f} MB) ...")
        last = None
        for attempt in range(1, attempts + 1):
            try:
                _fetch(base_url + file_name(e), part)
                got = part.stat().st_size if part.exists() else 0
                if got < e["bytes"]:                  # cut off: keep the .part file and resume it on the next attempt
                    raise RuntimeError(f"incomplete download ({got} of {e['bytes']} bytes)")
                reason = verify_file(part, e, check_sha)
                if reason is not None:
                    part.unlink(missing_ok=True)      # a complete but wrong file: never resume from it
                    raise RuntimeError(reason)
                part.replace(dest)
                out["ok"] += 1
                streak = 0
                last = None
                break
            except Exception as ex:  # noqa: BLE001 - retry, then record
                last = ex
                if attempt < attempts:
                    log(f"    attempt {attempt} of {attempts} failed ({ex}); retrying, resuming where it stopped ...")
                    time.sleep(min(retry_wait * attempt, 6.0))
        if last is not None:
            dest.unlink(missing_ok=True)              # the .part file is kept so the next run can resume it
            out["failed"].append((e["number"], str(last)))
            streak += 1
            log(f"    FAILED {file_name(e)} after {attempts} attempts: {last}")
            if streak >= max_consecutive_failures:
                log(f"    stopping after {streak} failed files in a row: the site may be unreachable from here")
    return out


def load_recording(path, number, rpm_fallback=None):
    """Drive-end accelerometer signal (1-D) and shaft speed of one CWRU file.

    Some files also hold arrays belonging to another file number, so the file's own variable is preferred."""
    mat = loadmat(str(path))
    found = {}
    for key, value in mat.items():
        m = re.match(r"^X(\d+)_DE_time$", key)
        if m:
            found[m.group(1)] = np.asarray(value, dtype=np.float64).ravel()
    if not found:
        raise ValueError(f"{path}: no drive-end signal (X..._DE_time) found")
    own = f"{int(number):03d}"
    var = own if own in found else next(iter(found))
    rpm_key = f"X{var}RPM"
    rpm = float(np.asarray(mat[rpm_key]).squeeze()) if rpm_key in mat else rpm_fallback
    return found[var], rpm


def make_windows(x, win, hop=None):
    """Non-overlapping (by default) windows of a 1-D signal: shape (n, win). Empty if the signal is too short."""
    x = np.asarray(x)
    hop = hop or win
    n = 1 + (len(x) - win) // hop if len(x) >= win else 0
    if n == 0:
        return np.empty((0, win))
    return np.stack([x[i * hop:i * hop + win] for i in range(n)])


def reduce_rate(x, fs_in, fs_out):
    """Reduce the sampling rate by an integer factor with an anti-alias filter (polyphase FIR)."""
    factor = fs_in / fs_out
    if abs(factor - round(factor)) > 1e-9 or factor < 1:
        raise ValueError(f"{fs_in} Hz to {fs_out} Hz is not an integer reduction")
    return x if round(factor) == 1 else resample_poly(x, 1, int(round(factor)))
