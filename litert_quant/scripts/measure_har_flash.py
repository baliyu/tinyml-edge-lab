"""Net flash of the two HAR builds. Use: PYTHONPATH=src python scripts/measure_har_flash.py

AI-assisted: drafted with Claude, checked by the author.
"""
from litert_quant.har_flash import measure

if __name__ == "__main__":
    measure()
    print("wrote results/model2_flash.json")
