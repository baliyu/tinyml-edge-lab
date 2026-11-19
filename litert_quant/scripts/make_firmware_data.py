"""Write the C headers. Use: PYTHONPATH=src python scripts/make_firmware_data.py

AI-assisted: drafted with Claude, checked by the author.
"""
from litert_quant.firmware_data import ROOT, make_headers

if __name__ == "__main__":
    names = make_headers(ROOT / "models", ROOT / "firmware" / "digits_bench")
    print("wrote", ", ".join(names))
