"""Draw the model 1 figure. Use: PYTHONPATH=src python scripts/make_model1_figure.py

AI-assisted: drafted with Claude, checked by the author.
"""
from litert_quant.figures import ROOT, make_model1_figure

if __name__ == "__main__":
    out = make_model1_figure(
        ROOT / "results" / "model1_pc_predictions.npz",
        ROOT / "results" / "model1_pc.json",
        ROOT / "results" / "figures" / "model1_pc.png",
    )
    print("wrote", out)
