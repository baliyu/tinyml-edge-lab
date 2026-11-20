"""Compare the CWRU board captures with the PC run. Use: PYTHONPATH=src python scripts/analyse_cwru_device.py

AI-assisted: drafted with Claude, checked by the author.
"""
from litert_quant.cwru_device import analyse

if __name__ == "__main__":
    analyse()
    print("wrote results/model3_device.json")
