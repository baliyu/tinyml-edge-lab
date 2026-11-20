"""Write the CWRU C headers. Use: PYTHONPATH=src python scripts/make_cwru_firmware_data.py

AI-assisted: drafted with Claude, checked by the author.
"""
from litert_quant.cwru_firmware_data import ROOT, build

if __name__ == "__main__":
    report = build(
        ROOT / "models",
        ROOT / "firmware" / "cwru_bench",
        ROOT / "results" / "model3_pc_scores.npz",
        ROOT / "results" / "model3_pc.json",
        ROOT / "results" / "model3_subset.json",
    )
    print("wrote", ", ".join(sorted(report["header_bytes"])), "and model3_subset.json")
