"""Render a :class:`~torchdiagram.graph.Graph` to a concrete output format."""

from __future__ import annotations

from pathlib import Path

from torchdiagram.graph import Graph
from torchdiagram.renderers.svg import to_svg
from torchdiagram.renderers.tikz import to_tikz
from torchdiagram.theme import DEFAULT, Theme

__all__ = ["render", "to_svg", "to_tikz"]

_BY_SUFFIX = {
    ".svg": to_svg,
    ".tex": to_tikz,
    ".tikz": to_tikz,
}


def render(graph: Graph, path: str | Path, *, theme: Theme = DEFAULT) -> Path:
    """Write ``graph`` to ``path``, picking the format from the file extension.

    Args:
        graph: Graph to render. Must pass ``graph.validate()``.
        path: Destination file path. The extension selects the renderer: ``.svg`` for SVG, ``.tex``/``.tikz`` for a
            standalone TikZ/LaTeX document.
        theme: Colors and typography to apply. Defaults to :data:`torchdiagram.theme.DEFAULT`.

    Returns:
        The path that was written, as a ``Path``.

    Raises:
        ValueError: If the extension is not supported, or ``graph`` fails validation.
    """
    path = Path(path)
    try:
        renderer = _BY_SUFFIX[path.suffix.lower()]
    except KeyError:
        supported = ", ".join(sorted(_BY_SUFFIX))
        raise ValueError(f"unsupported output format {path.suffix!r} (expected one of: {supported})") from None
    graph.validate()
    path.write_text(renderer(graph, theme=theme), encoding="utf-8")
    return path
