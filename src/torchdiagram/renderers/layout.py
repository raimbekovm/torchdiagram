"""Place an IR graph on a grid, shared by the SVG and TikZ renderers.

Both renderers used to walk the node list and put one node under the previous, which reads correctly for a chain and
badly for everything else. Four independent branches over one input, joined by a concat — an inception module, a
multi-head split, a U-Net's two paths — came out as four stacked boxes with curves arcing past them, which looks like a
chain with skip connections and hides the one thing the structure is about. And every non-adjacent edge claimed a lane
of its own to the right of the column, so the canvas grew sideways with the *number* of skip edges rather than with how
many are in flight at once: GPT-2's 168 residuals made a figure 4735 px wide.

Rows come from the longest path to a node, so anything that can run in parallel shares a row and is laid out side by
side. Lanes are reused once their edge's span ends, so lane count tracks maximum overlap. A model that really is a
chain still gets one node per row in one column, unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchdiagram.graph import Graph


@dataclass(frozen=True)
class Placement:
    """Where every node and skip edge sits on the grid.

    Attributes:
        rows: Node id mapped to its row, counted from the top.
        columns: Node id mapped to its column within that row.
        lanes: Skip edge, as a ``(source, target)`` pair, mapped to the routing lane it was given, counting from 1.
        width: Number of columns the widest row needs.
        height: Number of rows.
        lane_count: Number of routing lanes in use, which is the maximum number of skip edges ever in flight at once.
    """

    rows: dict[str, int]
    columns: dict[str, int]
    lanes: dict[tuple[str, str], int]
    width: int
    height: int
    lane_count: int

    def is_straight(self, source: str, target: str) -> bool:
        """Whether an edge is drawn as a direct line rather than routed through a lane."""
        return (source, target) not in self.lanes


def place(graph: Graph) -> Placement:
    """Assign every node a row and column, and every skip edge a routing lane.

    Args:
        graph: Graph to lay out, nodes in execution order.

    Returns:
        The grid placement.
    """
    rows = _rows(graph)
    columns = _columns(graph, rows)
    lanes = _lanes(graph, rows, columns)
    per_row: dict[int, int] = {}
    for node in graph.nodes:
        per_row[rows[node.id]] = per_row.get(rows[node.id], 0) + 1
    return Placement(
        rows=rows,
        columns=columns,
        lanes=lanes,
        width=max(per_row.values(), default=1),
        height=len(per_row),
        lane_count=max(lanes.values(), default=0),
    )


def _rows(graph: Graph) -> dict[str, int]:
    """Put each node one row below the last of its inputs, so independent branches land on the same row.

    Node order is the execution order, so one forward pass settles every row: a node's inputs are always already
    placed. Using the *longest* path rather than the shortest keeps an edge pointing downward, which is what lets the
    renderers treat a one-row step as a straight line.
    """
    rows: dict[str, int] = {}
    incoming: dict[str, list[str]] = {}
    for edge in graph.edges:
        incoming.setdefault(edge.target, []).append(edge.source)
    for node in graph.nodes:
        sources = [rows[source] for source in incoming.get(node.id, []) if source in rows]
        rows[node.id] = max(sources) + 1 if sources else 0
    return rows


def _columns(graph: Graph, rows: dict[str, int]) -> dict[str, int]:
    """Order the nodes sharing a row left to right, keeping each one under the branch it came from.

    Taking the first free column in execution order would let a branch drift sideways from row to row, so the four
    strands of an inception module would cross each other on the way down. Inheriting the column of the node that fed it
    keeps a branch in its own vertical lane for as long as it runs.
    """
    first_source: dict[str, str] = {}
    for edge in graph.edges:
        first_source.setdefault(edge.target, edge.source)
    columns: dict[str, int] = {}
    taken: dict[int, set[int]] = {}
    for node in graph.nodes:
        row = rows[node.id]
        occupied = taken.setdefault(row, set())
        inherited = columns.get(first_source.get(node.id, ""), 0)
        if inherited in occupied:
            inherited = next(free for free in range(len(occupied) + 1) if free not in occupied)
        occupied.add(inherited)
        columns[node.id] = inherited
    return columns


def _reaches_directly(
    source: str,
    target: str,
    rows: dict[str, int],
    columns: dict[str, int],
    filled: set[tuple[int, int]],
) -> bool:
    """Whether an arrow can go straight from one node to the other without passing through a box."""
    step = rows[target] - rows[source]
    if step == 1:
        return True
    if step < 1 or columns[source] != columns[target]:
        return False
    return all((row, columns[source]) not in filled for row in range(rows[source] + 1, rows[target]))


def _lanes(graph: Graph, rows: dict[str, int], columns: dict[str, int]) -> dict[tuple[str, str], int]:
    """Route every edge that cannot be drawn as a straight line through a lane, reusing lanes as their spans end.

    Allocating one lane per skip edge makes the canvas width a function of how many the architecture has in total, which
    for a transformer is two per block for as many blocks as it has. What actually needs separating is edges whose spans
    overlap, so a lane is free again the moment the edge in it has landed.

    An edge reaching straight down its own column past nothing at all needs no lane either: a branch that finishes early
    and waits for the join is a longer arrow, not a detour.
    """
    filled = {(rows[node.id], columns[node.id]) for node in graph.nodes}
    spans = [
        (rows[edge.source], rows[edge.target], (edge.source, edge.target))
        for edge in graph.edges
        if not _reaches_directly(edge.source, edge.target, rows, columns, filled)
    ]
    # Longest span first, so the outermost lanes carry the edges that reach furthest and short hops nest inside them.
    spans.sort(key=lambda span: (min(span[0], span[1]), -abs(span[1] - span[0])))
    free_at: list[int] = []  # per lane, the first row at which it is available again
    lanes: dict[tuple[str, str], int] = {}
    for source_row, target_row, edge in spans:
        top, bottom = min(source_row, target_row), max(source_row, target_row)
        # A curve leaves below its source and arrives above its target, so two edges meeting at one row do not
        # overlap and can share a lane — which is what a chain of residual blocks is made of.
        lane = next((index for index, row in enumerate(free_at) if row <= top), len(free_at))
        if lane == len(free_at):
            free_at.append(0)
        free_at[lane] = bottom
        lanes[edge] = lane + 1
    return lanes
