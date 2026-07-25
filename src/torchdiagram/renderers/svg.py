"""Static SVG renderer — instant preview with zero external dependencies.

Layout is a single vertical column in execution order. Edges between adjacent nodes are drawn as straight arrows;
skip/residual edges are routed as curves in lanes to the right of the column.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from torchdiagram.graph import Graph, Node
from torchdiagram.theme import DEFAULT, Theme

_PAD = 24
_GAP = 30
_CHAR_W = 7.4
_SKIP_LANE = 28


def to_svg(graph: Graph, *, theme: Theme = DEFAULT) -> str:
    """Render ``graph`` as a self-contained SVG document string.

    Args:
        graph: Graph to render, in execution order.
        theme: Colors and typography to apply. Defaults to :data:`torchdiagram.theme.DEFAULT`.

    Returns:
        A complete SVG document as a string.
    """
    lines_by_id = {node.id: _label_lines(node) for node in graph.nodes}
    two_line = any(len(lines) > 1 for lines in lines_by_id.values())
    box_h = 46 if two_line else 34
    longest = max((len(line) for lines in lines_by_id.values() for line in lines), default=8)
    box_w = max(150, int(longest * _CHAR_W) + 28)

    index = {node.id: i for i, node in enumerate(graph.nodes)}
    skip_edges = [e for e in graph.edges if abs(index[e.target] - index[e.source]) > 1]
    lanes = {(e.source, e.target): lane for lane, e in enumerate(skip_edges, start=1)}

    center_x = _PAD + box_w / 2
    right_x = _PAD + box_w
    width = int(right_x + len(skip_edges) * _SKIP_LANE + _PAD)
    height = _PAD * 2 + len(graph.nodes) * box_h + max(0, len(graph.nodes) - 1) * _GAP

    def y_top(i: int) -> float:
        return _PAD + i * (box_h + _GAP)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{theme.font_family}">',
        f"<title>{escape(graph.name)}</title>",
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{theme.edge_color}"/></marker>',
        "</defs>",
    ]

    for edge in graph.edges:
        si, ti = index[edge.source], index[edge.target]
        if ti - si == 1:
            y1, y2 = y_top(si) + box_h, y_top(ti) - 2
            parts.append(
                f'<line x1="{center_x}" y1="{y1}" x2="{center_x}" y2="{y2}" '
                f'stroke="{theme.edge_color}" stroke-width="1.4" marker-end="url(#arrow)"/>'
            )
        else:
            lane_x = right_x + lanes[(edge.source, edge.target)] * _SKIP_LANE
            y1 = y_top(si) + box_h / 2
            y2 = y_top(ti) + box_h / 2
            parts.append(
                f'<path d="M {right_x} {y1} C {lane_x} {y1}, {lane_x} {y2}, {right_x + 3} {y2}" '
                f'fill="none" stroke="{theme.edge_color}" stroke-width="1.4" marker-end="url(#arrow)"/>'
            )

    rx = _num(theme.corner_radius)
    stroke_w = _num(theme.stroke_width)
    for i, node in enumerate(graph.nodes):
        y = y_top(i)
        fill, stroke = (theme.io_fill, theme.io_stroke) if node.is_io else (theme.block_fill, theme.block_stroke)
        text_color = theme.io_text if node.is_io else theme.block_text
        parts.append(
            f'<rect x="{_PAD}" y="{y}" width="{box_w}" height="{box_h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}"/>'
        )
        lines = lines_by_id[node.id]
        if len(lines) == 1:
            parts.append(_text(center_x, y + box_h / 2, lines[0], size=13, fill=text_color))
        else:
            parts.append(_text(center_x, y + box_h * 0.36, lines[0], size=13, fill=text_color))
            parts.append(_text(center_x, y + box_h * 0.72, lines[1], size=11, fill=theme.shape_text))

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _num(value: float) -> str:
    """Format a length without a trailing ``.0`` (so ``6.0`` becomes ``"6"``, ``1.2`` stays ``"1.2"``)."""
    return str(int(value)) if value == int(value) else str(value)


def _text(x: float, y: float, content: str, *, size: int, fill: str) -> str:
    """Build a centered SVG ``<text>`` element."""
    return (
        f'<text x="{x}" y="{y}" text-anchor="middle" dominant-baseline="central" '
        f'font-size="{size}" fill="{fill}">{escape(content)}</text>'
    )


def _label_lines(node: Node) -> list[str]:
    """Split ``node`` into the one or two text lines rendered inside its box."""
    lines = [node.label]
    if node.output_shape is not None:
        lines.append("×".join(str(dim) for dim in node.output_shape))
    return lines
