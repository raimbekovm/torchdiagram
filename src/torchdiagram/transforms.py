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
    """Collapse scoped blocks into single labeled nodes, recursively from the deepest nesting outward.

    Nodes are grouped by the ``scope`` module they were traced from (see :func:`torchdiagram.trace.trace`). A group that
    has a single external entry point and whose only external exit originates from its last node (see :func:`_eligible`)
    always collapses into one synthetic ``"block"`` node — even if it occurs only once, e.g. an attention or MLP block
    that appears exactly once per transformer layer. Consecutive groups that additionally share a ``scope_class`` and an
    identical internal structure — same op sequence, same internal edge topology, same external-entry positions — are
    merged into a single node labeled e.g. ``"BasicBlock ×5"`` instead of one node per group, provided the run has at
    least ``min_repeats`` groups; shorter runs still collapse, just one node per group, unbadged. Nodes with no scope
    (``scope is None``) are never grouped, and groups that fail the eligibility check are left fully expanded.

    Collapsing runs deepest-scope-first means an outer scope (e.g. a transformer block containing an attention and an
    MLP sub-block) only needs to compare its own short, already-collapsed node sequence against its siblings, rather
    than the full expanded internals — which is what lets a repeated transformer block collapse to ``"Block ×N"`` even
    though its inner attention/MLP blocks are not themselves repeated.

    Args:
        graph: Graph to transform. Not mutated.
        min_repeats: Minimum run length, in groups, required to merge multiple groups into one badged node; below this,
            each eligible group still collapses, just individually.

    Returns:
        A new, validated graph with eligible scopes collapsed; everything else is copied through unchanged,
        in its original order.
    """
    nodes, edges = list(graph.nodes), list(graph.edges)
    max_depth = max((_depth(node.scope) for node in nodes if node.scope is not None), default=0)
    for depth in range(max_depth, 0, -1):
        nodes, edges = _aggregate_one_level(nodes, edges, depth, min_repeats)

    result = Graph(name=graph.name, nodes=nodes, edges=edges)
    result.validate()
    return result


def _depth(scope: str) -> int:
    """Nesting depth of a dotted scope path, e.g. ``"blocks.0.attn"`` is depth 3."""
    return scope.count(".") + 1


def _parent_scope(scope: str) -> str | None:
    """The dotted scope path one level up from ``scope``, or ``None`` if ``scope`` is already top-level."""
    parent, _, _ = scope.rpartition(".")
    return parent or None


def _aggregate_one_level(
    nodes: list[Node], edges: list[Edge], target_depth: int, min_repeats: int
) -> tuple[list[Node], list[Edge]]:
    """Collapse eligible scope groups at exactly ``target_depth``, leaving other depths untouched."""
    incoming: dict[str, list[Edge]] = {}
    outgoing: dict[str, list[Edge]] = {}
    for edge in edges:
        incoming.setdefault(edge.target, []).append(edge)
        outgoing.setdefault(edge.source, []).append(edge)

    path_class = {node.scope: node.scope_class for node in nodes if node.scope is not None}
    segments = _segment_by_scope(nodes, target_depth)
    items = _merge_runs(segments, incoming, outgoing)

    new_nodes: list[Node] = []
    collapsed: dict[str, str] = {}
    for item in items:
        if isinstance(item, Node):
            new_nodes.append(item)
            continue
        if not _eligible(item[0], incoming, outgoing):
            for group in item:
                new_nodes.extend(group.nodes)
            continue
        runs = [item] if len(item) >= min_repeats else [[group] for group in item]
        for run in runs:
            synthetic = _synthesize(run)
            parent = _parent_scope(run[0].scope)
            synthetic.scope = parent
            synthetic.scope_class = path_class.get(parent) if parent is not None else None
            new_nodes.append(synthetic)
            for group in run:
                for node in group.nodes:
                    collapsed[node.id] = synthetic.id

    new_edges: list[Edge] = []
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        source = collapsed.get(edge.source, edge.source)
        target = collapsed.get(edge.target, edge.target)
        if source == target or (source, target) in seen:
            continue
        seen.add((source, target))
        new_edges.append(Edge(source=source, target=target))
    return new_nodes, new_edges


def _segment_by_scope(nodes: list[Node], target_depth: int) -> list[Node | _Group]:
    """Partition ``nodes`` into non-groupable nodes and maximal same-scope groups at ``target_depth``."""
    segments: list[Node | _Group] = []
    current: _Group | None = None
    for node in nodes:
        has_scope = node.scope is not None and node.scope_class is not None and _depth(node.scope) == target_depth
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


def _op_key(node: Node) -> str:
    """Signature key for a node's operation: disambiguates synthetic ``"block"`` nodes by what they collapsed."""
    if node.op == "block":
        return f"block:{node.params.get('block_class')}"
    return node.op


def _signature(
    group: _Group, incoming: dict[str, list[Edge]], outgoing: dict[str, list[Edge]]
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...], tuple[bool, ...]]:
    """A structural fingerprint of ``group``, independent of node ids, for cross-group comparison."""
    ids = [node.id for node in group.nodes]
    id_set = set(ids)
    index = {node_id: i for i, node_id in enumerate(ids)}
    ops = tuple(_op_key(node) for node in group.nodes)
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
    label = first.scope_class if repeats == 1 else f"{first.scope_class} ×{repeats}"
    return Node(
        id=f"{first.nodes[0].id}__agg",
        op="block",
        label=label,
        params={"repeats": repeats, "block_class": first.scope_class, "ops_per_repeat": len(first.nodes)},
        output_shape=last.nodes[-1].output_shape,
    )
