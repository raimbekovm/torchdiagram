"""Static SVG renderer — instant preview with zero external dependencies.

Layout is a single vertical column in execution order. Edges between adjacent nodes are drawn as straight arrows;
skip/residual edges are routed as curves in lanes to the right of the column.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from torchdiagram.graph import Graph, Node

_PAD = 24
_GAP = 30
_CHAR_W = 7.4
_SKIP_LANE = 28

_BLOCK_FILL = "#e8eef9"
_BLOCK_STROKE = "#5b7db1"
_IO_FILL = "#f1f3f5"
_IO_STROKE = "#8a919a"
_EDGE_COLOR = "#4c566a"
_FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"


def to_svg(graph: Graph) -> str:
    """Render ``graph`` as a self-contained SVG document string.

    Args:
        graph: Graph to render, in execution order.

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
        f'viewBox="0 0 {width} {height}" font-family="{_FONT}">',
        f"<title>{escape(graph.name)}</title>",
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{_EDGE_COLOR}"/></marker>',
        "</defs>",
    ]

    for edge in graph.edges:
        si, ti = index[edge.source], index[edge.target]
        if ti - si == 1:
            y1, y2 = y_top(si) + box_h, y_top(ti) - 2
            parts.append(
                f'<line x1="{center_x}" y1="{y1}" x2="{center_x}" y2="{y2}" '
                f'stroke="{_EDGE_COLOR}" stroke-width="1.4" marker-end="url(#arrow)"/>'
            )
        else:
            lane_x = right_x + lanes[(edge.source, edge.target)] * _SKIP_LANE
            y1 = y_top(si) + box_h / 2
            y2 = y_top(ti) + box_h / 2
            parts.append(
                f'<path d="M {right_x} {y1} C {lane_x} {y1}, {lane_x} {y2}, {right_x + 3} {y2}" '
                f'fill="none" stroke="{_EDGE_COLOR}" stroke-width="1.4" marker-end="url(#arrow)"/>'
            )

    for i, node in enumerate(graph.nodes):
        y = y_top(i)
        io = node.op in ("input", "output")
        fill, stroke = (_IO_FILL, _IO_STROKE) if io else (_BLOCK_FILL, _BLOCK_STROKE)
        parts.append(
            f'<rect x="{_PAD}" y="{y}" width="{box_w}" height="{box_h}" rx="6" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
        )
        lines = lines_by_id[node.id]
        if len(lines) == 1:
            parts.append(_text(center_x, y + box_h / 2, lines[0], size=13))
        else:
            parts.append(_text(center_x, y + box_h * 0.36, lines[0], size=13))
            parts.append(_text(center_x, y + box_h * 0.72, lines[1], size=11, fill="#68707c"))

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _text(x: float, y: float, content: str, *, size: int, fill: str = "#2e3440") -> str:
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
