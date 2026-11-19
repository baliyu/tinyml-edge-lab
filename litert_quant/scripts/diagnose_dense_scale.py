"""Reproduce the dense-scale finding. Use: PYTHONPATH=src python scripts/diagnose_dense_scale.py

AI-assisted: drafted with Claude, checked by the author.
"""
import sys

from litert_quant.dense_diagnosis import main

if __name__ == "__main__":
    sys.exit(main())
