"""Draw the model 2 figure. Use: PYTHONPATH=src python scripts/make_model2_figure.py

AI-assisted: drafted with Claude, checked by the author.
"""
from litert_quant.figure_model2 import ROOT, make_model2_figure

if __name__ == "__main__":
    out = make_model2_figure(
        ROOT / "results" / "model2_pc_predictions.npz",
        ROOT / "results" / "model2_pc.json",
        ROOT.parent / "results" / "har_baseline.json",
        ROOT / "results" / "figures" / "model2_pc.png",
    )
    print("wrote", out)
