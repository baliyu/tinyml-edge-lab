"""Tests for the device figure caption.

AI-assisted: drafted with Claude, checked by the author.
"""
from litert_quant.cwru_device_figure import CAPTION, CAPTION_WIDTH, wrap_caption


def test_every_line_fits_the_width():
    lines = wrap_caption().split("\n")
    assert all(len(line) <= CAPTION_WIDTH for line in lines)


def test_wrapping_keeps_every_word():
    assert " ".join(wrap_caption().split()) == " ".join(CAPTION.split())


def test_caption_keeps_the_clock_caveat():
    assert "64 MHz" in " ".join(wrap_caption().split())


def test_caption_wraps_to_at_most_four_lines():
    assert len(wrap_caption().split("\n")) <= 4
