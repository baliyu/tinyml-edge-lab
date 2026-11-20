"""Make the model 3 PC figure. Use: PYTHONPATH=src python scripts/make_model3_figure.py

AI-assisted: drafted with Claude, checked by the author.
"""
from litert_quant.figure_model3 import make_model3_figure
from litert_quant.figures import ROOT

if __name__ == "__main__":
    out = make_model3_figure(
        ROOT / "results" / "model3_pc_scores.npz",
        ROOT / "results" / "model3_pc.json",
        ROOT / "results" / "figures" / "model3_pc.png",
    )
    print(out)
