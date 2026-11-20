"""Net flash of the two CWRU builds. Use: PYTHONPATH=src python scripts/measure_cwru_flash.py

AI-assisted: drafted with Claude, checked by the author.
"""
from litert_quant.cwru_flash import measure

if __name__ == "__main__":
    measure()
    print("wrote results/model3_flash.json")
