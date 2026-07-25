"""Visual themes for the renderers: colors and typography, shared across SVG and TikZ.

A :class:`Theme` is a plain, torch-free value — like the IR in :mod:`torchdiagram.graph` — that both
renderers accept as an optional argument. It carries only the part of the visual contract that maps
cleanly to both backends: colors, the font family, and two small lengths. Geometry (node spacing, box
size) is deliberately not themeable yet, because SVG measures in pixels and TikZ in millimeters, so a
single shared value would have no honest unit; see ``docs/design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    r"""Colors and typography for a rendered diagram.

    All colors are hex strings (``"#rrggbb"``). Renderers apply these directly (SVG) or translate them to their own
    model (TikZ emits ``\definecolor`` entries). The instance is frozen so the built-in presets can be shared safely;
    build a variant with :func:`dataclasses.replace`.

    Attributes:
        block_fill: Fill color of computation blocks.
        block_stroke: Outline color of computation blocks.
        block_text: Label text color inside computation blocks.
        io_fill: Fill color of input/output blocks.
        io_stroke: Outline color of input/output blocks.
        io_text: Label text color inside input/output blocks.
        edge_color: Color of data-flow arrows.
        shape_text: Color of the secondary output-shape line under a label.
        font_family: Font family for labels. Applies to SVG only; TikZ uses the LaTeX document font.
        corner_radius: Block corner rounding — pixels in SVG, points in TikZ.
        stroke_width: Block outline width — pixels in SVG, points in TikZ.
    """

    block_fill: str = "#e8eef9"
    block_stroke: str = "#5b7db1"
    block_text: str = "#2e3440"
    io_fill: str = "#f1f3f5"
    io_stroke: str = "#8a919a"
    io_text: str = "#2e3440"
    edge_color: str = "#4c566a"
    shape_text: str = "#68707c"
    font_family: str = "'Helvetica Neue', Helvetica, Arial, sans-serif"
    corner_radius: float = 6.0
    stroke_width: float = 1.2


DEFAULT = Theme()
"""The default palette — a soft blue block on a light background."""

MONOCHROME = Theme(
    block_fill="#f0f0f0",
    block_stroke="#888888",
    block_text="#111111",
    io_fill="#ffffff",
    io_stroke="#aaaaaa",
    io_text="#111111",
    edge_color="#555555",
    shape_text="#666666",
)
"""Grayscale palette, for print figures where color is costly."""

DARK = Theme(
    block_fill="#2e3440",
    block_stroke="#81a1c1",
    block_text="#eceff4",
    io_fill="#3b4252",
    io_stroke="#616e88",
    io_text="#eceff4",
    edge_color="#d8dee9",
    shape_text="#a3adbb",
)
"""Palette for dark backgrounds, e.g. a dark-mode README or slide."""
