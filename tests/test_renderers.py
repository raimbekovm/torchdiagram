"""Tests for the SVG and TikZ renderers and the render() dispatcher."""

import pytest

import torchdiagram as td


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


def test_render_dispatches_on_extension(tmp_path):
    """Render() picks the format from the output file extension."""
    svg_path = td.render(demo_graph(), tmp_path / "model.svg")
    assert svg_path.read_text().startswith("<svg")
    tex_path = td.render(demo_graph(), tmp_path / "model.tex")
    assert r"\documentclass" in tex_path.read_text()


def test_render_rejects_unknown_extension(tmp_path):
    """Render() raises on unsupported output formats."""
    with pytest.raises(ValueError, match="unsupported output format"):
        td.render(demo_graph(), tmp_path / "model.png")


def test_render_validates_graph(tmp_path):
    """Render() validates the graph before writing anything."""
    bad = td.Graph(nodes=[td.Node(id="a", op="custom", label="a")], edges=[td.Edge("a", "ghost")])
    with pytest.raises(ValueError, match="unknown node id"):
        td.render(bad, tmp_path / "model.svg")
