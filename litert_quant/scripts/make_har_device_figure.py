"""Write the device figure. Use: PYTHONPATH=src python scripts/make_har_device_figure.py

AI-assisted: drafted with Claude, checked by the author.
"""
from litert_quant.har_device_figure import ROOT, make_device_figure

if __name__ == "__main__":
    out = make_device_figure(
        ROOT / "results" / "model2_device.json",
        ROOT / "results" / "model2_flash.json",
        ROOT / "results" / "figures" / "model2_device.png",
    )
    print("wrote", out.relative_to(ROOT))
