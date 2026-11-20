"""Write the CWRU device figure. Use: PYTHONPATH=src python scripts/make_cwru_device_figure.py

AI-assisted: drafted with Claude, checked by the author.
"""
from litert_quant.cwru_device_figure import ROOT, make_device_figure

if __name__ == "__main__":
    out = make_device_figure(
        ROOT / "results" / "model3_device.json",
        ROOT / "results" / "model3_flash.json",
        ROOT / "results" / "figures" / "model3_device.png",
    )
    print("wrote", out.relative_to(ROOT))
