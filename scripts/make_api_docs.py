#!/usr/bin/env python3
"""Keep docs/API.md in step with the code: write its generated sections (C headers, Python reference, command-line tools), or check them.

Usage: python scripts/make_api_docs.py           # rewrite the generated sections of docs/API.md
       python scripts/make_api_docs.py --check   # exit 1 if docs/API.md is out of date (the test suite does this)
The sections between <!-- api:NAME:start --> and <!-- api:NAME:end --> are generated from the C headers, from the signatures and docstrings of the
supported Python functions (API_PYTHON below) and from the first line of each script's docstring. Everything else in the file is written by hand.
A public function in a supported module that is missing from API_PYTHON, or without a docstring, fails the tests, so the reference cannot fall behind.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DOC = ROOT / "docs" / "API.md"

C_HEADERS = ["firmware/sparkfun_edge/gesture_features/edge_features.h", "firmware/sparkfun_edge/gesture_demo/gesture_pipeline.h",
             "firmware/sparkfun_edge/gesture_demo/gesture_model.h"]

# The supported Python surface: module -> public names, in the order they are shown.
API_PYTHON = {
    "features": ["feature_names", "window_features", "batch_features", "with_magnitude", "gravity_mean", "gravity_names"],
    "selection": ["group_folds", "assert_disjoint", "group_cv_macro_f1", "one_se_choice"],
    "uci_har": ["find_root", "load_signal", "load_meta", "load_split", "assert_subject_disjoint"],
    "har_baseline": ["build_features", "run"],
    "har_compare": ["build_features", "forest_nodes", "forest_leaves", "run_compare"],
    "cwru": ["file_name", "sha256_file", "verify_file", "check_dir", "download_all", "load_recording", "make_windows", "reduce_rate"],
    "cwru_features": ["feature_names", "band_edges", "batch_features"],
    "cwru_eval": ["build_table", "lolo_splits", "loso_splits", "run_cwru"],
    "download": ["sha256_of", "download", "extract_uci_har"],
    "export_c": ["floor_f32", "f32_literal", "tree_to_c", "forest_to_c"],
    "footprint": ["find_toolchain", "parse_berkeley_size", "measure", "host_agreement", "variants_for", "measure_model", "refit_models", "refit_matches_saved"],
    "edge_gestures": ["Capture", "Dataset", "read_capture", "resample_to_grid", "trim_edges", "make_windows", "load_dataset", "design_problems",
                      "feature_names_for", "feature_matrix"],
    "edge_gesture_eval": ["session_splits", "assert_session_disjoint", "ordered_labels", "run_gestures", "best_honest", "export_tree"],
    "edge_synth": ["gesture_motion", "write_synthetic_gestures"],
    "edge_features_c": ["find_gcc", "CFeatures", "compare_features", "prediction_agreement"],
    "gesture_demo": ["train_deployable", "model_sources", "write_model", "CPipeline", "ticks_from_seconds", "python_chain", "compare_chains",
                     "check_recordings", "arm_footprint", "build_host_sim", "run_host_sim"],
}
# Modules that draw figures or write README blocks: not part of the API.
INTERNAL_MODULES = {"plots", "cwru_plots", "edge_gesture_report", "footprint_report", "cwru_readme", "readme_block"}


def public_names(mod) -> list[str]:
    return [n for n, o in vars(mod).items() if not n.startswith("_") and (inspect.isfunction(o) or inspect.isclass(o)) and getattr(o, "__module__", "") == mod.__name__]


def short_sig(obj) -> str:
    """The parameters of a function or class with short defaults only: long ones (a manifest, a path) are shown as ..."""
    if inspect.isclass(obj) and dataclasses.is_dataclass(obj):
        return "(" + ", ".join(f.name for f in dataclasses.fields(obj)) + ")"
    parts, star = [], False
    for p in inspect.signature(obj).parameters.values():
        if p.name == "self":
            continue
        if p.kind is p.VAR_POSITIONAL:
            parts.append("*" + p.name); star = True
        elif p.kind is p.VAR_KEYWORD:
            parts.append("**" + p.name)
        else:
            if p.kind is p.KEYWORD_ONLY and not star:
                parts.append("*"); star = True
            if p.default is p.empty:
                parts.append(p.name)
            else:
                r = repr(p.default)
                parts.append(f"{p.name}={r if len(r) <= 18 else '...'}")
    return "(" + ", ".join(parts) + ")"


def summary(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    para = " ".join(doc.split("\n\n")[0].split())
    if len(para) > 280:
        cut = para.rfind(". ", 0, 280)
        para = para[:cut + 1] if cut > 80 else para[:277] + "..."
    return para


def cell(text: str) -> str:
    return text.replace("|", "\\|")


def render_python() -> str:
    out = []
    for mod_name, names in API_PYTHON.items():
        mod = importlib.import_module(f"tinyml_lab.{mod_name}")
        head = (inspect.getdoc(mod) or "").split("\n")[0]
        out += [f"#### `tinyml_lab.{mod_name}`" + (f": {head}" if head else ""), "", "| Name | Parameters | What it does |", "|---|---|---|"]
        for n in names:
            o = getattr(mod, n)
            kind = "class " if inspect.isclass(o) else ""
            out.append(f"| {kind}`{n}` | `{cell(short_sig(o))}` | {cell(summary(o))} |")
        out.append("")
    return "\n".join(out).rstrip()


def render_c() -> str:
    out = []
    for h in C_HEADERS:
        text = (ROOT / h).read_text(encoding="utf-8").rstrip()
        out += [f"#### `{h}`", "", "```c", text, "```", ""]
    return "\n".join(out).rstrip()


def render_tools() -> str:
    rows = ["| Command | What it does |", "|---|---|"]
    for d in ("scripts", "tools"):
        for p in sorted((ROOT / d).glob("*.py")):
            doc = ast.get_docstring(ast.parse(p.read_text(encoding="utf-8")))
            first = " ".join((doc or "(no description)").split("\n\n")[0].split())
            rows.append(f"| `{d}/{p.name}` | {cell(first)} |")
    return "\n".join(rows)


SECTIONS = {"c": render_c, "python": render_python, "tools": render_tools}


def fill(text: str) -> str:
    for name, fn in SECTIONS.items():
        start, end = f"<!-- api:{name}:start -->", f"<!-- api:{name}:end -->"
        if text.count(start) != 1 or text.count(end) != 1:
            raise SystemExit(f"docs/API.md must contain exactly one {start} ... {end} pair")
        a, b = text.index(start) + len(start), text.index(end)
        text = text[:a] + "\n" + fn() + "\n" + text[b:]
    return text


def problems() -> list[str]:
    """Reasons the supported API is not fully described (empty if all is well)."""
    out = []
    import tinyml_lab
    modules = {m.name for m in pkgutil.iter_modules(tinyml_lab.__path__)}
    for m in sorted(modules - set(API_PYTHON) - INTERNAL_MODULES):
        out.append(f"module {m} is neither in API_PYTHON nor in INTERNAL_MODULES")
    for m in sorted((set(API_PYTHON) | INTERNAL_MODULES) - modules):
        out.append(f"{m} is listed but is not a module of tinyml_lab")
    for mod_name, names in API_PYTHON.items():
        mod = importlib.import_module(f"tinyml_lab.{mod_name}")
        for n in names:
            if not hasattr(mod, n):
                out.append(f"{mod_name}.{n} is listed but does not exist")
            elif not (inspect.getdoc(getattr(mod, n)) or "").strip():
                out.append(f"{mod_name}.{n} has no docstring")
        for n in sorted(set(public_names(mod)) - set(names)):
            out.append(f"{mod_name}.{n} is public but not in API_PYTHON (add it, or make it private with a leading underscore)")
    return out


def main(argv=None) -> int:
    check = "--check" in (argv if argv is not None else sys.argv[1:])
    bad = problems()
    if bad:
        print("The supported API is not fully described:\n  " + "\n  ".join(bad))
        return 1
    current = DOC.read_text(encoding="utf-8")
    new = fill(current)
    if check:
        if new != current:
            print("docs/API.md is out of date: run  python scripts/make_api_docs.py")
            return 1
        print("docs/API.md is up to date")
        return 0
    if new != current:
        DOC.write_text(new, encoding="utf-8")
        print("docs/API.md updated")
    else:
        print("docs/API.md already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
