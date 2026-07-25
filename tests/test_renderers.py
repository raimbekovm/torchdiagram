"""Tests for the SVG, TikZ, and PNG renderers and the render() dispatcher."""

import re
import struct
import sys

import pytest

import torchdiagram as td

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def png_size(data: bytes) -> tuple[int, int]:
    """Read the pixel dimensions from a PNG's IHDR chunk, which always starts at byte 16."""
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def svg_size(svg: str) -> tuple[int, int]:
    """Read the declared width and height off an SVG document's root element."""
    match = re.search(r'width="(\d+)" height="(\d+)"', svg)
    assert match is not None
    return int(match[1]), int(match[2])


def demo_graph() -> td.Graph:
    """Build a small hand-written graph with a skip connection."""
    return td.Graph(
        name="demo",
        nodes=[
            td.Node(id="x", op="input", label="input", output_shape=(1, 16)),
            td.Node(id="fc", op="linear", label="Linear"),
            td.Node(id="act", op="relu", label="relu"),
            td.Node(id="out", op="output", label="output"),
        ],
        edges=[
            td.Edge("x", "fc"),
            td.Edge("fc", "act"),
            td.Edge("act", "out"),
            td.Edge("x", "act"),  # skip connection
        ],
    )


def test_svg_contains_all_labels():
    """Every node label appears in the SVG output."""
    svg = td.to_svg(demo_graph())
    assert svg.startswith("<svg")
    assert "Linear" in svg
    assert "relu" in svg


def test_svg_routes_skip_edges_as_curves():
    """Non-adjacent edges are drawn as curved paths, not straight lines."""
    assert "<path" in td.to_svg(demo_graph())


def test_svg_escapes_markup_in_labels():
    """XML-special characters in labels are escaped."""
    graph = td.Graph(nodes=[td.Node(id="a", op="custom", label="a<b>&c")])
    assert "a&lt;b&gt;&amp;c" in td.to_svg(graph)


def test_tikz_is_a_standalone_document():
    """TikZ output is a complete standalone LaTeX document."""
    tikz = td.to_tikz(demo_graph())
    assert r"\documentclass" in tikz
    assert r"\begin{tikzpicture}" in tikz
    assert r"\end{document}" in tikz


def test_tikz_escapes_special_characters():
    """LaTeX-special characters in labels are escaped."""
    graph = td.Graph(nodes=[td.Node(id="a", op="custom", label="d_model & 50%")])
    tikz = td.to_tikz(graph)
    assert r"d\_model" in tikz
    assert r"\&" in tikz
    assert r"\%" in tikz


def test_tikz_escapes_multiplication_sign():
    """The '×' character used in aggregated-block labels renders as a math-mode LaTeX command."""
    graph = td.Graph(nodes=[td.Node(id="a", op="block", label="BasicBlock ×4")])
    assert r"BasicBlock $\times$4" in td.to_tikz(graph)


def test_png_is_the_svg_rasterized_at_scale():
    """PNG output is a real PNG whose pixel size is the SVG's own size times the scale factor."""
    graph = demo_graph()
    data = td.to_png(graph, scale=2.0)
    assert data.startswith(_PNG_MAGIC)
    width, height = svg_size(td.to_svg(graph))
    assert png_size(data) == (width * 2, height * 2)


def test_png_scale_controls_resolution():
    """A larger scale factor produces a proportionally larger image from the same graph."""
    small = png_size(td.to_png(demo_graph(), scale=1.0))
    large = png_size(td.to_png(demo_graph(), scale=3.0))
    assert large == (small[0] * 3, small[1] * 3)


def test_png_applies_the_theme():
    """The theme reaches the rasterizer, so two palettes give different pixels at the same size."""
    default = td.to_png(demo_graph(), scale=1.0)
    dark = td.to_png(demo_graph(), scale=1.0, theme=td.DARK)
    assert png_size(default) == png_size(dark)
    assert default != dark


def test_png_rejects_non_positive_scale():
    """A zero or negative scale raises instead of reaching the rasterizer."""
    with pytest.raises(ValueError, match="scale must be positive"):
        td.to_png(demo_graph(), scale=0)


def test_png_reports_a_missing_rasterizer(monkeypatch):
    """Without the optional dependency, to_png() raises an ImportError naming the install extra."""
    monkeypatch.setitem(sys.modules, "resvg_py", None)
    with pytest.raises(ImportError, match=r"torchdiagram\[png\]"):
        td.to_png(demo_graph())


def test_render_dispatches_on_extension(tmp_path):
    """Render() picks the format from the output file extension."""
    svg_path = td.render(demo_graph(), tmp_path / "model.svg")
    assert svg_path.read_text().startswith("<svg")
    tex_path = td.render(demo_graph(), tmp_path / "model.tex")
    assert r"\documentclass" in tex_path.read_text()
    png_path = td.render(demo_graph(), tmp_path / "model.png")
    assert png_path.read_bytes().startswith(_PNG_MAGIC)


def test_render_forwards_scale_to_png(tmp_path):
    """Render() passes its scale factor to the PNG renderer."""
    path = td.render(demo_graph(), tmp_path / "model.png", scale=1.0)
    assert png_size(path.read_bytes()) == svg_size(td.to_svg(demo_graph()))


def test_render_rejects_unknown_extension(tmp_path):
    """Render() raises on unsupported output formats."""
    with pytest.raises(ValueError, match="unsupported output format"):
        td.render(demo_graph(), tmp_path / "model.pdf")


def test_render_validates_graph(tmp_path):
    """Render() validates the graph before writing anything."""
    bad = td.Graph(nodes=[td.Node(id="a", op="custom", label="a")], edges=[td.Edge("a", "ghost")])
    with pytest.raises(ValueError, match="unknown node id"):
        td.render(bad, tmp_path / "model.svg")
