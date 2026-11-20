"""Write the HAR C headers. Use: PYTHONPATH=src python scripts/make_har_firmware_data.py

AI-assisted: drafted with Claude, checked by the author.
"""
from litert_quant.har_firmware_data import ROOT, build

if __name__ == "__main__":
    report = build(
        ROOT / "models",
        ROOT / "firmware" / "har_bench",
        ROOT / "results" / "model2_pc_predictions.npz",
        ROOT / "results" / "model2_subset.json",
    )
    print("wrote", ", ".join(sorted(report["header_bytes"])), "and model2_subset.json")
