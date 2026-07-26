"""Static SVG renderer — instant preview with zero external dependencies.

Layout comes from :mod:`torchdiagram.renderers.layout`: nodes that can run in parallel share a row and sit side by
side, everything else falls into one column in execution order. Edges between neighbouring rows are drawn as straight
arrows; edges that reach further are routed as curves in lanes to the right.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from torchdiagram.graph import Graph, Node
from torchdiagram.renderers._format import length
from torchdiagram.renderers.layout import place
from torchdiagram.theme import DEFAULT, Theme

_PAD = 24
_GAP = 30
_COLUMN_GAP = 24
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

    grid = place(graph)
    columns_right = _PAD + grid.width * box_w + (grid.width - 1) * _COLUMN_GAP
    width = int(columns_right + grid.lane_count * _SKIP_LANE + _PAD)
    height = _PAD * 2 + grid.height * box_h + max(0, grid.height - 1) * _GAP

    def left_x(node_id: str) -> float:
        return _PAD + grid.columns[node_id] * (box_w + _COLUMN_GAP)

    def center_x(node_id: str) -> float:
        return left_x(node_id) + box_w / 2

    def top_y(node_id: str) -> float:
        return _PAD + grid.rows[node_id] * (box_h + _GAP)

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
        if grid.is_straight(edge.source, edge.target):
            x1, x2 = center_x(edge.source), center_x(edge.target)
            y1, y2 = top_y(edge.source) + box_h, top_y(edge.target) - 2
            parts.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{theme.edge_color}" stroke-width="1.4" marker-end="url(#arrow)"/>'
            )
        else:
            lane_x = columns_right + grid.lanes[(edge.source, edge.target)] * _SKIP_LANE
            start_x = left_x(edge.source) + box_w
            end_x = left_x(edge.target) + box_w
            # Leave from the bottom corner and arrive at the top one, so the curve travels the gap between rows
            # instead of crossing the boxes standing to the right of its source at their own height.
            y1 = top_y(edge.source) + box_h
            y2 = top_y(edge.target)
            parts.append(
                f'<path d="M {start_x} {y1} C {lane_x} {y1}, {lane_x} {y2}, {end_x + 3} {y2}" '
                f'fill="none" stroke="{theme.edge_color}" stroke-width="1.4" marker-end="url(#arrow)"/>'
            )

    rx = length(theme.corner_radius)
    stroke_w = length(theme.stroke_width)
    for node in graph.nodes:
        x, y = left_x(node.id), top_y(node.id)
        fill, stroke = (theme.io_fill, theme.io_stroke) if node.is_io else (theme.block_fill, theme.block_stroke)
        text_color = theme.io_text if node.is_io else theme.block_text
        parts.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}"/>'
        )
        lines = lines_by_id[node.id]
        if len(lines) == 1:
            parts.append(_text(x + box_w / 2, y + box_h / 2, lines[0], size=13, fill=text_color))
        else:
            parts.append(_text(x + box_w / 2, y + box_h * 0.36, lines[0], size=13, fill=text_color))
            parts.append(_text(x + box_w / 2, y + box_h * 0.72, lines[1], size=11, fill=theme.shape_text))

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


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
