import functools
import http.server
import json
import threading
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from tinyml_lab import cwru, cwru_eval
from tinyml_lab.cwru_features import batch_features, band_edges, feature_names
from tinyml_lab.cwru_plots import build_cwru_figures
from tinyml_lab.cwru_readme import END, START, render_block, update_readme

PNG = b"\x89PNG\r\n\x1a\n"


# ------------------------------------------------------------------ manifest and files
def test_manifest_matches_the_documented_layout():
    m = cwru.MANIFEST
    assert len(m) == 40
    assert {e["cls"] for e in m} == set(cwru.CLASS_ORDER)
    assert len({e["number"] for e in m}) == 40
    for cls in cwru.CLASS_ORDER:
        assert sorted(e["load_hp"] for e in m if e["cls"] == cls) == [0, 1, 2, 3]
    assert all(len(e["sha256"]) == 64 and int(e["sha256"], 16) >= 0 for e in m)
    ir021_3hp = next(e for e in m if e["cls"] == "IR021" and e["load_hp"] == 3)
    assert ir021_3hp["number"] == 217          # the official table skips 216 for this one
    assert [e["number"] for e in m if e["cls"] == "normal"] == [97, 98, 99, 100]


def test_verify_file_reports_each_kind_of_problem(tmp_path):
    entry = {"number": 1, "bytes": 5, "sha256": cwru.hashlib.sha256(b"hello").hexdigest()}
    p = tmp_path / "1.mat"
    assert cwru.verify_file(p, entry) == "missing"
    p.write_bytes(b"hello")
    assert cwru.verify_file(p, entry) is None
    p.write_bytes(b"hellO")
    assert "SHA-256" in cwru.verify_file(p, entry)
    p.write_bytes(b"hell")
    assert "size" in cwru.verify_file(p, entry)


def test_load_recording_prefers_the_files_own_variable_and_falls_back_on_rpm(tmp_path):
    p = tmp_path / "119.mat"
    savemat(p, {"X118_DE_time": np.ones((10, 1)), "X119_DE_time": np.full((10, 1), 2.0), "X119RPM": np.array([[1750]])})
    x, rpm = cwru.load_recording(p, 119)
    assert np.all(x == 2.0) and rpm == 1750.0
    q = tmp_path / "5.mat"
    savemat(q, {"X005_DE_time": np.zeros((4, 1))})
    x, rpm = cwru.load_recording(q, 5, rpm_fallback=1797)
    assert len(x) == 4 and rpm == 1797
    bad = tmp_path / "bad.mat"
    savemat(bad, {"other": np.zeros(3)})
    with pytest.raises(ValueError):
        cwru.load_recording(bad, 1)


def test_windows_and_rate_reduction():
    x = np.arange(10_000, dtype=float)
    w = cwru.make_windows(x, 4096)
    assert w.shape == (2, 4096) and w[1, 0] == 4096
    assert cwru.make_windows(x[:100], 4096).shape == (0, 4096)
    y = cwru.reduce_rate(np.random.default_rng(0).normal(size=48_000), 48000.0, 400.0)
    assert len(y) == 400
    with pytest.raises(ValueError):
        cwru.reduce_rate(x, 48000.0, 7000.0)


# ------------------------------------------------------------------ features
def test_features_have_the_right_shape_and_meaning():
    fs, n = 48000.0, 4096
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    tone = np.sin(2 * np.pi * 3000 * t)[None, :]
    noise = rng.normal(size=(1, n))
    impulses = rng.normal(0, 0.1, (1, n)); impulses[0, ::400] += 8.0
    f = batch_features(np.vstack([tone, noise, impulses]), fs)
    names = feature_names()
    assert f.shape == (3, len(names)) == (3, 17)
    kurt = names.index("kurtosis")
    assert f[1, kurt] == pytest.approx(3.0, abs=0.5)     # Gaussian noise
    assert f[2, kurt] > 10                                # impulsive signal
    edges = band_edges(fs)
    band = int(np.searchsorted(edges, 3000) - 1)
    assert f[0, names.index(f"band{band}_fraction")] > 0.9     # a pure tone sits in one band
    assert sum(f[1, names.index(f"band{b}_fraction")] for b in range(12)) <= 1.0 + 1e-9


# ------------------------------------------------------------------ protocols
def test_load_and_size_splits_never_leak():
    loads = np.repeat([0, 1, 2, 3], 5)
    for held, tr, te in cwru_eval.lolo_splits(loads):
        assert set(loads[te]) == {held} and held not in set(loads[tr])
        assert len(tr) + len(te) == len(loads)
    sizes = np.array([0.0] * 5 + [0.007] * 5 + [0.014] * 5 + [0.021] * 5)
    for held, tr, te in cwru_eval.loso_splits(sizes, sizes > 0):
        assert held not in set(sizes[tr]) and set(sizes[te]) == {held}
        assert 0.0 not in set(sizes[tr]) | set(sizes[te])      # the normal class is excluded


def test_pipeline_end_to_end_with_audit(fake_cwru, tmp_path):
    data, manifest = fake_cwru
    out = tmp_path / "r"
    rep = cwru_eval.run_cwru(data, out_dir=out, manifest=manifest, win_400=32, log=lambda *_: None)
    assert set(rep["protocols"]) == {"random_windows", "leave_one_load_out", "leave_one_size_out_fault_types", "leave_one_load_out_400hz"}
    for fold in rep["protocols"]["leave_one_load_out"]["folds"]:
        assert fold["test_load"] not in fold["train_loads"]
    for fold in rep["protocols"]["leave_one_load_out_400hz"]["folds"]:
        assert fold["test_load"] not in fold["train_loads"]
    sizes = rep["protocols"]["leave_one_size_out_fault_types"]
    for fold in sizes["folds"]:
        assert fold["test_size_in"] not in fold["train_sizes_in"]
    assert sizes["models"]["tree"]["labels"] == ["IR", "B", "OR"]
    assert [f["test_size_in"] for f in sizes["folds"]] == [0.007, 0.014]      # only real fault sizes; never the normal class
    assert sum(sum(row) for row in sizes["models"]["forest"]["confusion_matrix"]) == sum(f["n_test"] for f in sizes["folds"])
    for p in rep["protocols"].values():
        for m in p["models"].values():
            assert 0 <= m["accuracy"] <= 1 and 0 <= m["macro_f1"] <= 1
            assert np.sum(m["confusion_matrix"]) > 0
    assert rep["protocols"]["leave_one_load_out"]["models"]["forest"]["accuracy"] > 0.8   # the synthetic faults are separable
    saved = json.loads((out / "cwru.json").read_text())
    assert saved["versions"]["scipy"] and saved["settings"]["note"].startswith("model settings were fixed")
    assert (out / "cwru.md").read_text().startswith("| Evaluation")


def test_figures_are_real_png_files(fake_cwru, tmp_path):
    data, manifest = fake_cwru
    out = tmp_path / "r"
    cwru_eval.run_cwru(data, out_dir=out, manifest=manifest, win_400=32, log=lambda *_: None)
    names = build_cwru_figures(data, out, manifest=manifest)
    assert names == ["cwru_protocols.png", "cwru_signatures.png", "cwru_confusion_forest.png"]
    for n in names:
        b = (out / "figures" / n).read_bytes()
        assert b[:8] == PNG and len(b) > 3000


def test_readme_block_comes_from_results_and_is_idempotent(fake_cwru, tmp_path):
    data, manifest = fake_cwru
    rep = cwru_eval.run_cwru(data, out_dir=tmp_path / "r", manifest=manifest, win_400=32, log=lambda *_: None)
    block = render_block(rep)
    for p in rep["protocols"].values():
        assert f"{p['models']['forest']['macro_f1']:.3f}" in block
    for key in ("leave_one_load_out", "leave_one_size_out_fault_types"):        # per-fold numbers are shown, not just averages
        for acc in rep["protocols"][key]["models"]["forest"]["fold_accuracy"]:
            assert f"{acc:.3f}" in block
    assert "held-out fold" in block and "chance is 0.333" in block
    readme = tmp_path / "README.md"
    readme.write_text("# T\n\n| Part | What |\n|---|---|\n| `src/tinyml_lab/plots.py` | figs |\n\n```bash\n"
                      "python scripts/compare_models.py   # x\n```\n\n## Roadmap\n1. next\n")
    update_readme(readme, block)
    once = readme.read_text()
    update_readme(readme, block)
    assert readme.read_text() == once
    assert once.count(START) == 1 and once.count(END) == 1
    assert once.count("cwru*.py") == 1 and once.count("scripts/run_cwru.py") == 1
    assert once.index(START) < once.index("## Roadmap") and once.startswith("# T\n")


# ------------------------------------------------------------------ downloader (against a local web server)
class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    servers = []

    def start(directory, handler_class=_Quiet):
        handler = functools.partial(handler_class, directory=str(directory))
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}/"

    yield start
    for s in servers:
        s.shutdown()


def test_downloader_verifies_skips_and_rejects_corrupted_files(fake_cwru, tmp_path, local_server):
    src, manifest = fake_cwru
    base = local_server(src)
    dest = tmp_path / "downloaded"
    res = cwru.download_all(dest, manifest, base_url=base, log=lambda *_: None, retry_wait=0)
    assert res == {"ok": 28, "skipped": 0, "failed": [], "not_attempted": 0}
    assert cwru.check_dir(dest, manifest) == []
    again = cwru.download_all(dest, manifest, base_url=base, log=lambda *_: None, retry_wait=0)
    assert again["skipped"] == 28 and again["ok"] == 0
    # corrupt one file on the server: it must be rejected and must not be left in the data folder
    victim = manifest[3]
    path = src / cwru.file_name(victim)
    data = bytearray(path.read_bytes()); data[100] ^= 0xFF; path.write_bytes(bytes(data))
    dest2 = tmp_path / "downloaded2"
    res2 = cwru.download_all(dest2, manifest, base_url=base, log=lambda *_: None, retry_wait=0, attempts=2)
    assert [n for n, _ in res2["failed"]] == [victim["number"]]
    assert "SHA-256" in res2["failed"][0][1]
    assert not (dest2 / cwru.file_name(victim)).exists() and not (dest2 / (str(victim["number"]) + ".part")).exists()
    assert res2["ok"] == 27


def test_downloader_gives_up_quickly_when_the_site_is_unreachable(fake_cwru, tmp_path):
    _, manifest = fake_cwru
    logged = []
    res = cwru.download_all(tmp_path / "d", manifest, base_url="http://127.0.0.1:1/", log=logged.append, max_consecutive_failures=3, attempts=2, retry_wait=0)
    assert res["ok"] == 0 and len(res["failed"]) == 3 and res["not_attempted"] == len(manifest) - 3
    assert any("stopping after 3 failed files" in m for m in logged)
    assert not list((tmp_path / "d").glob("*.mat")) and not list((tmp_path / "d").glob("*.part"))


class _Flaky(http.server.SimpleHTTPRequestHandler):
    """Supports Range requests, but cuts any full-file transfer of the named files in half (a dropped connection)."""
    drop_names = set()

    def log_message(self, *args):
        pass

    def do_GET(self):
        name = self.path.lstrip("/")
        data = (Path(self.directory) / name).read_bytes()
        rng = self.headers.get("Range")
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        if rng:
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not rng and name in self.drop_names:
            self.wfile.write(body[: len(body) // 2])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)


@pytest.mark.parametrize("use_curl", [True, False])
def test_dropped_transfers_are_resumed_not_restarted(fake_cwru, tmp_path, local_server, monkeypatch, use_curl):
    import shutil
    if use_curl and shutil.which("curl") is None:
        pytest.skip("curl not installed")
    if not use_curl:
        monkeypatch.setattr(cwru.shutil, "which", lambda *_: None)      # exercise the urllib fallback
    src, manifest = fake_cwru
    _Flaky.drop_names = {cwru.file_name(manifest[2]), cwru.file_name(manifest[9])}
    base = local_server(src, _Flaky)
    logged = []
    res = cwru.download_all(tmp_path / "d", manifest, base_url=base, log=logged.append, retry_wait=0)
    assert res == {"ok": 28, "skipped": 0, "failed": [], "not_attempted": 0}
    assert sum("resuming where it stopped" in m for m in logged) >= 2     # both flaky files needed a resume
    assert cwru.check_dir(tmp_path / "d", manifest) == []
    assert not list((tmp_path / "d").glob("*.part"))


def test_a_partial_file_from_an_earlier_run_is_continued(fake_cwru, tmp_path, local_server):
    src, manifest = fake_cwru
    victim = manifest[5]
    _Flaky.drop_names = {cwru.file_name(victim)}
    base = local_server(src, _Flaky)
    d = tmp_path / "d"
    d.mkdir()
    full = (src / cwru.file_name(victim)).read_bytes()
    (d / (str(victim["number"]) + ".part")).write_bytes(full[: len(full) // 3])   # an interrupted earlier download
    res = cwru.download_all(d, [victim], base_url=base, log=lambda *_: None, retry_wait=0)
    assert res["ok"] == 1 and res["failed"] == []
    assert cwru.verify_file(d / cwru.file_name(victim), victim) is None


def test_a_complete_but_corrupted_download_is_discarded_and_refetched(fake_cwru, tmp_path, local_server):
    src, manifest = fake_cwru
    victim = manifest[1]
    d = tmp_path / "d"
    d.mkdir()
    (d / (str(victim["number"]) + ".part")).write_bytes(b"x" * victim["bytes"])    # right size, wrong content
    base = local_server(src)
    res = cwru.download_all(d, [victim], base_url=base, log=lambda *_: None, retry_wait=0)
    assert res["ok"] == 1 and cwru.verify_file(d / cwru.file_name(victim), victim) is None
