"""Formatting helpers shared by the renderers."""

from __future__ import annotations


def length(value: float) -> str:
    """Format a theme length without a trailing ``.0`` (so ``6.0`` becomes ``"6"``, ``1.2`` stays ``"1.2"``).

    Args:
        value: Length in the renderer's own unit — pixels for SVG, points for TikZ.

    Returns:
        The length as the shortest string that reads the same in both output formats.
    """
    return str(int(value)) if value == int(value) else str(value)
