"""Render a :class:`~torchdiagram.graph.Graph` to a concrete output format."""

from __future__ import annotations

from pathlib import Path

from torchdiagram.graph import Graph
from torchdiagram.renderers.png import to_png
from torchdiagram.renderers.svg import to_svg
from torchdiagram.renderers.tikz import to_tikz
from torchdiagram.theme import DEFAULT, Theme

__all__ = ["render", "to_png", "to_svg", "to_tikz"]

_TEXT_BY_SUFFIX = {
    ".svg": to_svg,
    ".tex": to_tikz,
    ".tikz": to_tikz,
}

# PNG is handled apart from the table: it writes bytes rather than text, and it is the one format with a scale factor.
_SUFFIXES = (*_TEXT_BY_SUFFIX, ".png")


def render(graph: Graph, path: str | Path, *, theme: Theme = DEFAULT, scale: float = 2.0) -> Path:
    """Write ``graph`` to ``path``, picking the format from the file extension.

    Args:
        graph: Graph to render. Must pass ``graph.validate()``.
        path: Destination file path. The extension selects the renderer: ``.svg`` for SVG, ``.tex``/``.tikz`` for a
            standalone TikZ/LaTeX document, ``.png`` for a rasterized image.
        theme: Colors and typography to apply. Defaults to :data:`torchdiagram.theme.DEFAULT`.
        scale: Pixel scale factor for ``.png`` output, ignored by the vector formats. See
            :func:`torchdiagram.renderers.png.to_png`.

    Returns:
        The path that was written, as a ``Path``.

    Raises:
        ImportError: If ``.png`` output was requested without the optional ``resvg-py`` dependency installed.
        ValueError: If the extension is not supported, ``graph`` fails validation, or ``scale`` is not positive.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in _SUFFIXES:
        supported = ", ".join(sorted(_SUFFIXES))
        raise ValueError(f"unsupported output format {path.suffix!r} (expected one of: {supported})")
    graph.validate()
    if suffix == ".png":
        path.write_bytes(to_png(graph, theme=theme, scale=scale))
    else:
        path.write_text(_TEXT_BY_SUFFIX[suffix](graph, theme=theme), encoding="utf-8")
    return path
