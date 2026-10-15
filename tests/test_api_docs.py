"""docs/API.md must not drift from the code: its generated sections are current, the supported API is fully described, and the hand-written
protocol and example sections still match what the firmware prints and what the package contains."""
import ast
import importlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_api_docs", ROOT / "scripts" / "make_api_docs.py")
mad = importlib.util.module_from_spec(spec)
sys.modules["make_api_docs"] = mad
spec.loader.exec_module(mad)
mspec = importlib.util.spec_from_file_location("edge_gesture_monitor", ROOT / "tools" / "edge_gesture_monitor.py")
egm = importlib.util.module_from_spec(mspec)
sys.modules["edge_gesture_monitor"] = egm
mspec.loader.exec_module(egm)

DOC = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
MAIN_C = (ROOT / "firmware" / "sparkfun_edge" / "gesture_demo" / "main.c").read_text(encoding="utf-8")


def blocks(lang):
    return re.findall(rf"```{lang}\n(.*?)```", DOC, re.S)


# ---------------------------------------------------------------- the generated sections
def test_the_generated_sections_are_up_to_date():
    assert mad.main(["--check"]) == 0


def test_every_public_function_of_a_supported_module_is_listed_and_has_a_docstring():
    assert mad.problems() == []


def test_a_missing_name_or_a_missing_docstring_is_reported(monkeypatch):
    monkeypatch.setitem(mad.API_PYTHON, "features", [n for n in mad.API_PYTHON["features"] if n != "with_magnitude"])
    assert any("features.with_magnitude is public but not in API_PYTHON" in p for p in mad.problems())
    monkeypatch.setitem(mad.API_PYTHON, "features", mad.API_PYTHON["features"] + ["no_such_function"])
    assert any("no_such_function is listed but does not exist" in p for p in mad.problems())


def test_a_module_that_is_neither_supported_nor_internal_is_reported(monkeypatch):
    monkeypatch.setattr(mad, "INTERNAL_MODULES", mad.INTERNAL_MODULES - {"plots"})
    assert any("module plots is neither" in p for p in mad.problems())


def test_a_stale_document_is_noticed(tmp_path, monkeypatch):
    stale = tmp_path / "API.md"
    stale.write_text(DOC.replace("| `read_capture` |", "| `read_capture_old` |"), encoding="utf-8")
    monkeypatch.setattr(mad, "DOC", stale)
    assert mad.main(["--check"]) == 1
    assert mad.main([]) == 0 and mad.main(["--check"]) == 0                      # regenerating repairs it


def test_the_document_needs_its_markers(monkeypatch, tmp_path):
    bare = tmp_path / "API.md"
    bare.write_text("# nothing generated here\n", encoding="utf-8")
    monkeypatch.setattr(mad, "DOC", bare)
    with pytest.raises(SystemExit):
        mad.main(["--check"])


def test_long_defaults_are_shown_as_dots_and_dataclasses_by_their_fields():
    from tinyml_lab import cwru, edge_gestures as eg
    assert "manifest=..." in mad.short_sig(cwru.check_dir) and "data_dir" in mad.short_sig(cwru.check_dir)
    assert mad.short_sig(eg.load_dataset).startswith("(root, fs=50.0, win=128")
    assert mad.short_sig(eg.Capture).startswith("(path, label, subject") and mad.short_sig(eg.Capture).endswith("xyz_mg)")


def test_the_summary_is_the_first_paragraph_and_never_runs_on():
    def f():
        """First sentence. Second sentence of the same paragraph.

        A second paragraph that is not shown."""
    assert mad.summary(f) == "First sentence. Second sentence of the same paragraph."

    def g():
        pass
    g.__doc__ = "A sentence of about eighty characters, which goes on and on in order to be long enough. " * 6
    s = mad.summary(g)
    assert len(s) <= 281 and s.endswith(".")


def test_every_script_and_tool_has_a_row_in_the_table():
    for d in ("scripts", "tools"):
        for p in (ROOT / d).glob("*.py"):
            assert f"| `{d}/{p.name}` |" in DOC


def test_every_c_header_is_included_verbatim():
    for h in mad.C_HEADERS:
        assert (ROOT / h).read_text(encoding="utf-8").rstrip() in DOC


# ---------------------------------------------------------------- the serial protocol section against the firmware and the reference reader
def test_the_documented_protocol_lines_are_what_the_firmware_prints_and_the_monitor_reads():
    proto = re.search(r"```\n(# sparkfun_edge gesture demo.*?)```", DOC, re.S).group(1)
    kinds = [(egm.parse_line(ln) or ("none", None))[0] for ln in proto.strip().splitlines()]
    assert kinds == ["hello", "config", "comment", "window", "stats", "error"]
    assert "gesture demo v1" in MAIN_C and "demo v1" in proto                                          # one version number, in both places
    assert "# columns: G,window,t_ticks,gesture,compute_us" in MAIN_C and "# columns: G,window,t_ticks,gesture,compute_us" in proto
    stats_fmt = re.search(r'"# stats ([^"]*?)\\r\\n"', MAIN_C).group(1)                       # the printf format itself, not a comment about it
    stats_doc = next(ln for ln in proto.splitlines() if ln.startswith("# stats ")).split("# stats ", 1)[1]
    assert [kv.split("=")[0] for kv in stats_fmt.split()] == [kv.split("=")[0] for kv in stats_doc.split()]
    assert "# error: accelerometer init failed" in MAIN_C and "# error: accelerometer init failed" in proto
    g_fmt = re.search(r'"(G,[^"]*?)\\r\\n"', MAIN_C).group(1)
    g_doc = next(ln for ln in proto.splitlines() if ln.startswith("G,"))
    assert g_fmt.count(",") == g_doc.count(",") == 4 and proto.count("G,12,123456789,rest,3812") == 1


def test_the_documented_constants_match_the_code():
    h = (ROOT / "firmware/sparkfun_edge/gesture_demo/gesture_pipeline.h").read_text(encoding="utf-8")
    const = {k: int(v) for k, v in re.findall(r"#define (GP_\w+) (\d+)", h)}
    gap_s, window_s = const["GP_MAX_GAP_STEPS"] / const["GP_FS_HZ"], const["GP_STEP"] / const["GP_FS_HZ"]
    assert f"`GP_MAX_GAP_STEPS` grid steps ({gap_s:.2f} s)" in DOC
    assert f"every {window_s:.2f} s" in DOC and "a new one\n  every 64" in DOC and const["GP_STEP"] == 64


# ---------------------------------------------------------------- the Python examples
def test_the_python_examples_only_use_names_that_exist():
    sys.path.insert(0, str(ROOT / "src"))
    seen = 0
    for code in blocks("python"):
        tree = ast.parse(code)
        alias = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "tinyml_lab":
                for a in node.names:
                    alias[a.asname or a.name] = importlib.import_module(f"tinyml_lab.{a.name}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in alias:
                assert hasattr(alias[node.value.id], node.attr), f"{node.value.id}.{node.attr} in docs/API.md does not exist"
                seen += 1
    assert seen >= 6


def test_the_example_calls_match_the_real_signatures():
    import inspect
    from tinyml_lab import edge_gestures as eg, gesture_demo as gd
    assert list(inspect.signature(eg.load_dataset).parameters)[:2] == ["root", "fs"]
    assert list(inspect.signature(eg.feature_matrix).parameters)[:3] == ["windows", "fs", "feature_set"]
    assert list(inspect.signature(gd.train_deployable).parameters)[:3] == ["data_dir", "model", "feature_set"]
    assert list(inspect.signature(gd.write_model).parameters)[:5] == ["clf", "labels", "feature_set", "model", "out_dir"]
    assert "log" in inspect.signature(gd.run_host_sim).parameters
