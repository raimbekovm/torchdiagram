"""PNG renderer — the SVG output rasterized for READMEs, slides, and issue threads.

Rasterization is delegated to resvg through the optional ``resvg-py`` wheel, so PNG output needs no system
libraries and no headless browser. Install it with ``pip install 'torchdiagram[png]'``.
"""

from __future__ import annotations

from torchdiagram.graph import Graph
from torchdiagram.renderers.svg import to_svg
from torchdiagram.theme import DEFAULT, Theme

_MISSING_RASTERIZER = "PNG output needs the optional rasterizer: pip install 'torchdiagram[png]'"


def to_png(graph: Graph, *, theme: Theme = DEFAULT, scale: float = 2.0) -> bytes:
    """Render ``graph`` as PNG image data.

    The image is the SVG rendering rasterized at ``scale``. The background stays transparent, as in the SVG, so the
    diagram sits on whatever page it is embedded in. Label text is typeset with the system fonts that match
    ``theme.font_family``; on a machine with no fonts installed the boxes and arrows still render but the text is
    dropped, so prefer SVG or TikZ in minimal containers.

    Args:
        graph: Graph to render, in execution order.
        theme: Colors and typography to apply. Defaults to :data:`torchdiagram.theme.DEFAULT`.
        scale: Pixel scale factor applied to the SVG's own dimensions, e.g. ``2.0`` renders a 400x600 diagram as a
            800x1200 image. Vector formats ignore it; this is the only knob PNG needs.

    Returns:
        The encoded PNG file as bytes.

    Raises:
        ImportError: If the optional ``resvg-py`` dependency is not installed.
        ValueError: If ``scale`` is not positive, or the SVG cannot be rasterized.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    try:
        import resvg_py
    except ImportError as exc:
        raise ImportError(_MISSING_RASTERIZER) from exc
    return resvg_py.svg_to_bytes(svg_string=to_svg(graph, theme=theme), zoom=scale)
