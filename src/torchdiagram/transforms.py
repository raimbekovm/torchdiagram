"""Pure IR-to-IR transforms applied between the frontend and the renderers.

Transforms only see :mod:`torchdiagram.graph` dataclasses, so they work on traced graphs and
hand-built ones alike, and require no changes to the renderers.
"""

from __future__ import annotations

from dataclasses import dataclass

from .graph import Edge, Graph, Node


@dataclass
class _Group:
    """A maximal run of consecutive nodes traced from the same module scope."""

    scope: str
    scope_class: str
    nodes: list[Node]


def aggregate_blocks(graph: Graph, *, min_repeats: int = 2) -> Graph:
    """Collapse runs of repeated, structurally identical blocks into a single labeled node.

    Nodes are grouped by the ``scope`` module they were traced from (see :func:`torchdiagram.trace.trace`). Consecutive
    groups that share a ``scope_class`` and an identical internal structure — same op sequence, same internal edge
    topology, same external-entry positions — are merged into one synthetic ``"block"`` node labeled e.g. ``"BasicBlock
    ×5"``. A group is only eligible to merge if it has a single external entry point and its only external exit
    originates from its last node; groups that fail this check, or whose structure differs from their neighbors, are
    left untouched. Nodes with no scope (``scope is None``) are never grouped.

    Args:
        graph: Graph to transform. Not mutated.
        min_repeats: Minimum run length, in groups, required to collapse a run.

    Returns:
        A new, validated graph with eligible runs collapsed; everything else is copied through unchanged, in
        its original order.
    """
    incoming: dict[str, list[Edge]] = {}
    outgoing: dict[str, list[Edge]] = {}
    for edge in graph.edges:
        incoming.setdefault(edge.target, []).append(edge)
        outgoing.setdefault(edge.source, []).append(edge)

    segments = _segment_by_scope(graph.nodes)
    items = _merge_runs(segments, incoming, outgoing)

    new_nodes: list[Node] = []
    collapsed: dict[str, str] = {}
    for item in items:
        if isinstance(item, Node):
            new_nodes.append(item)
        elif len(item) >= min_repeats:
            synthetic = _synthesize(item)
            new_nodes.append(synthetic)
            for group in item:
                for node in group.nodes:
                    collapsed[node.id] = synthetic.id
        else:
            for group in item:
                new_nodes.extend(group.nodes)

    new_edges: list[Edge] = []
    seen: set[tuple[str, str]] = set()
    for edge in graph.edges:
        source = collapsed.get(edge.source, edge.source)
        target = collapsed.get(edge.target, edge.target)
        if source == target or (source, target) in seen:
            continue
        seen.add((source, target))
        new_edges.append(Edge(source=source, target=target))

    result = Graph(name=graph.name, nodes=new_nodes, edges=new_edges)
    result.validate()
    return result


def _segment_by_scope(nodes: list[Node]) -> list[Node | _Group]:
    """Partition ``nodes`` into scope-less nodes and maximal same-scope groups, in order."""
    segments: list[Node | _Group] = []
    current: _Group | None = None
    for node in nodes:
        has_scope = node.scope is not None and node.scope_class is not None
        if has_scope and current is not None and current.scope == node.scope:
            current.nodes.append(node)
            continue
        current = None
        if has_scope:
            assert node.scope is not None and node.scope_class is not None  # narrowed by has_scope
            current = _Group(scope=node.scope, scope_class=node.scope_class, nodes=[node])
            segments.append(current)
        else:
            segments.append(node)
    return segments


def _merge_runs(
    segments: list[Node | _Group],
    incoming: dict[str, list[Edge]],
    outgoing: dict[str, list[Edge]],
) -> list[Node | list[_Group]]:
    """Scan ``segments`` for maximal runs of adjacent, eligible, structurally matching groups."""
    items: list[Node | list[_Group]] = []
    i = 0
    while i < len(segments):
        segment = segments[i]
        if isinstance(segment, Node):
            items.append(segment)
            i += 1
            continue
        if not _eligible(segment, incoming, outgoing):
            items.append([segment])
            i += 1
            continue
        signature = _signature(segment, incoming, outgoing)
        run = [segment]
        j = i + 1
        while j < len(segments):
            candidate = segments[j]
            if (
                not isinstance(candidate, _Group)
                or not _eligible(candidate, incoming, outgoing)
                or candidate.scope_class != segment.scope_class
                or _signature(candidate, incoming, outgoing) != signature
            ):
                break
            run.append(candidate)
            j += 1
        items.append(run)
        i = j
    return items


def _eligible(group: _Group, incoming: dict[str, list[Edge]], outgoing: dict[str, list[Edge]]) -> bool:
    """Whether ``group`` has a single external entry point and only exits from its last node."""
    ids = {node.id for node in group.nodes}
    last_id = group.nodes[-1].id
    entry_sources = {
        edge.source for node in group.nodes for edge in incoming.get(node.id, []) if edge.source not in ids
    }
    if len(entry_sources) > 1:
        return False
    exit_edges = [edge for node in group.nodes for edge in outgoing.get(node.id, []) if edge.target not in ids]
    return all(edge.source == last_id for edge in exit_edges)


def _signature(
    group: _Group, incoming: dict[str, list[Edge]], outgoing: dict[str, list[Edge]]
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...], tuple[bool, ...]]:
    """A structural fingerprint of ``group``, independent of node ids, for cross-group comparison."""
    ids = [node.id for node in group.nodes]
    id_set = set(ids)
    index = {node_id: i for i, node_id in enumerate(ids)}
    ops = tuple(node.op for node in group.nodes)
    topology = []
    external_entry = []
    for i, node_id in enumerate(ids):
        edges_in = incoming.get(node_id, [])
        internal_offsets = tuple(sorted(i - index[edge.source] for edge in edges_in if edge.source in id_set))
        topology.append(internal_offsets)
        external_entry.append(any(edge.source not in id_set for edge in edges_in))
    return ops, tuple(topology), tuple(external_entry)


def _synthesize(run: list[_Group]) -> Node:
    """Build the single synthetic node that replaces a collapsed run of matching groups."""
    first, last = run[0], run[-1]
    repeats = len(run)
    return Node(
        id=f"{first.nodes[0].id}__agg",
        op="block",
        label=f"{first.scope_class} ×{repeats}",
        params={"repeats": repeats, "block_class": first.scope_class, "ops_per_repeat": len(first.nodes)},
        output_shape=last.nodes[-1].output_shape,
    )
