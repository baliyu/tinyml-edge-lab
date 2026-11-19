"""Capture a board run. Use: set PYTHONPATH=src, then python scripts/board_capture.py --port COM6 --out FILE

AI-assisted: drafted with Claude, checked by the author.
"""
import sys

from litert_quant.board_capture import main

if __name__ == "__main__":
    sys.exit(main())
