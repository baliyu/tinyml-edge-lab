"""Write the CNN against tree figure. Use: PYTHONPATH=src python scripts/make_compare_figure.py

AI-assisted: drafted with Claude, checked by the author.
"""
from litert_quant.compare_figure import ROOT, make_comparison_figure

if __name__ == "__main__":
    r = ROOT / "results"
    out = make_comparison_figure(
        r / "model2_pc.json", r / "model2_device.json", r / "model2_flash.json",
        r / "tree_footprint.json", r / "figures" / "model2_vs_tree.png",
    )
    print("wrote", out.relative_to(ROOT))
