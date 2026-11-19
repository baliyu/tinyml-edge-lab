"""Analyse a board log. Use: PYTHONPATH=src python scripts/analyse_board.py results/board_int8_run1.txt

AI-assisted: drafted with Claude, checked by the author.
"""
import sys

from litert_quant.board_log import cli

if __name__ == "__main__":
    sys.exit(cli(sys.argv))
