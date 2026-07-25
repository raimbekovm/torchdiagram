"""Tests for the theming API — Theme, the built-in presets, and how they flow into each renderer."""

from dataclasses import FrozenInstanceError, replace

import pytest

import torchdiagram as td


def demo_graph() -> td.Graph:
    """Build a small graph with a shape annotation so the shape-line color is exercised."""
    return td.Graph(
        name="demo",
        nodes=[
            td.Node(id="x", op="input", label="input", output_shape=(1, 16)),
            td.Node(id="fc", op="linear", label="Linear"),
            td.Node(id="out", op="output", label="output"),
        ],
        edges=[td.Edge("x", "fc"), td.Edge("fc", "out")],
    )


def test_default_theme_is_the_implicit_default():
    """Omitting theme is the same as passing DEFAULT, and yields the documented default colors."""
    svg = td.to_svg(demo_graph())
    assert svg == td.to_svg(demo_graph(), theme=td.DEFAULT)
    assert td.DEFAULT.block_fill in svg  # "#e8eef9"
    assert td.DEFAULT.block_stroke in svg  # "#5b7db1"


def test_custom_colors_flow_into_svg():
    """A custom block color appears in the SVG and displaces the default one."""
    theme = replace(td.DEFAULT, block_fill="#123456")
    svg = td.to_svg(demo_graph(), theme=theme)
    assert "#123456" in svg
    assert td.DEFAULT.block_fill not in svg


def test_custom_colors_flow_into_tikz():
    """A custom color is emitted as a named xcolor definition in the TikZ preamble."""
    theme = replace(td.DEFAULT, block_fill="#123456")
    tikz = td.to_tikz(demo_graph(), theme=theme)
    assert r"\definecolor{tdBlockFill}{HTML}{123456}" in tikz


def test_tikz_default_defines_named_colors():
    """The default TikZ preamble defines a named color per theme field and references it."""
    tikz = td.to_tikz(demo_graph())
    assert r"\definecolor{tdBlockFill}{HTML}{E8EEF9}" in tikz
    assert "fill=tdBlockFill" in tikz


def test_presets_produce_distinct_output():
    """Each preset renders differently, and its own palette shows up in the SVG."""
    default_svg = td.to_svg(demo_graph())
    for preset in (td.DARK, td.MONOCHROME):
        svg = td.to_svg(demo_graph(), theme=preset)
        assert svg != default_svg
        assert preset.block_fill in svg


def test_block_and_io_text_colors_are_routed_separately():
    """A block node uses block_text; an io node uses io_text — the two aren't conflated."""
    theme = replace(td.DEFAULT, block_text="#111111", io_text="#222222")
    svg = td.to_svg(demo_graph(), theme=theme)
    # The "Linear" block label carries block_text; the "input"/"output" io labels carry io_text.
    assert 'fill="#111111">Linear' in svg
    assert 'fill="#222222">input' in svg
    assert 'fill="#222222">output' in svg


def test_theme_is_frozen():
    """A Theme is immutable, so the shared presets can't be mutated in place."""
    with pytest.raises(FrozenInstanceError):
        td.DEFAULT.block_fill = "#000000"  # type: ignore[misc]
