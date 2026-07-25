"""Pure IR-to-IR transforms applied between the frontend and the renderers.

Transforms only see :mod:`torchdiagram.graph` dataclasses, so they work on traced graphs and
hand-built ones alike, and require no changes to the renderers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from .graph import Edge, Graph, Node, scope_leaf

# Containers that hold layers without being one. Their class name says nothing about what a block does, so a collapsed
# block gets named after the attribute holding it — "classifier" rather than "Sequential".
_CONTAINERS = frozenset({"Sequential", "ModuleList", "ModuleDict"})

_Signature = tuple[tuple[str, ...], tuple[tuple[int, ...], ...], tuple[bool, ...]]


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
    structure: dict[str, str] = {}  # synthetic node id -> digest of everything that node collapsed
    path_class = {**graph.scopes}
    path_class.update({node.scope: node.scope_class for node in nodes if node.scope and node.scope_class})

    nodes, edges = _collapse_repeats(nodes, edges, min_repeats, structure)
    max_depth = max((_depth(node.scope) for node in nodes if node.scope is not None), default=0)
    for depth in range(max_depth, 0, -1):
        nodes, edges = _aggregate_one_level(nodes, edges, depth, min_repeats, structure, path_class)

    result = Graph(name=graph.name, nodes=nodes, edges=edges, scopes=dict(graph.scopes))
    result.validate()
    return result


def _collapse_repeats(
    nodes: list[Node], edges: list[Edge], min_repeats: int, structure: dict[str, str]
) -> tuple[list[Node], list[Edge]]:
    """Badge a run of adjacent interchangeable layers held in one container as a single node.

    A stack of N identical layers held directly in an ``nn.Sequential`` — four ``nn.TransformerEncoderLayer``, a CNN of
    convolutions at one width — is a single group to :func:`_segment_by_scope`, because a leaf layer reports the
    container holding it rather than itself, so all N siblings share one ``scope``. :func:`_merge_runs` only ever
    compares whole groups against each other, so the repetition *inside* one group was invisible and the whole stack
    came out as one box named after the attribute, with nothing saying how many layers went into it. A hand-written
    block gets this right only because each instance carries its own ``scope``.

    Runs are found before any grouping, so what the level above sees is already one badged node.

    Args:
        nodes: Nodes in execution order.
        edges: Data-flow edges.
        min_repeats: Minimum run length required to badge; shorter runs are left as they are.
        structure: Digest registry, extended with an entry per node created here.

    Returns:
        The nodes and edges with each qualifying run replaced by one node.
    """
    incoming, outgoing = _edge_index(edges)
    new_nodes: list[Node] = []
    collapsed: dict[str, str] = {}
    index = 0
    while index < len(nodes):
        end = index + 1
        while (
            end < len(nodes)
            and _interchangeable(nodes[index], nodes[end], structure)
            and _links_only(nodes[end - 1], nodes[end], incoming, outgoing)
        ):
            end += 1
        run = nodes[index:end]
        if len(run) >= min_repeats and _repeatable(nodes[index]):
            synthetic = _synthesize_run(run, structure)
            new_nodes.append(synthetic)
            collapsed.update({node.id: synthetic.id for node in run})
        else:
            new_nodes.extend(run)
        index = end
    return new_nodes, _rewire(edges, collapsed)


def _edge_index(edges: list[Edge]) -> tuple[dict[str, list[Edge]], dict[str, list[Edge]]]:
    """Index ``edges`` by target and by source."""
    incoming: dict[str, list[Edge]] = {}
    outgoing: dict[str, list[Edge]] = {}
    for edge in edges:
        incoming.setdefault(edge.target, []).append(edge)
        outgoing.setdefault(edge.source, []).append(edge)
    return incoming, outgoing


def _repeatable(node: Node) -> bool:
    """Whether ``node`` is a layer inside a container, which is the only thing a repeat count can be claimed about."""
    return node.scope is not None and node.scope_class is not None and node.op != "block"


def _interchangeable(first: Node, second: Node, structure: dict[str, str]) -> bool:
    """Whether two nodes are the same layer twice: same container, same operation, same configuration."""
    return (
        first.scope == second.scope
        and first.scope_class == second.scope_class
        and _op_key(first, structure) == _op_key(second, structure)
    )


def _links_only(previous: Node, node: Node, incoming: dict[str, list[Edge]], outgoing: dict[str, list[Edge]]) -> bool:
    """Whether the flow goes from ``previous`` straight into ``node`` and nowhere else on either side."""
    return [edge.target for edge in outgoing.get(previous.id, [])] == [node.id] and [
        edge.source for edge in incoming.get(node.id, [])
    ] == [previous.id]


def _synthesize_run(run: list[Node], structure: dict[str, str]) -> Node:
    """Build the badged node that replaces a run of repeated sibling layers."""
    first = run[0]
    repeats = len(run)
    name = first.label.partition(" (")[0]  # drop the attribute qualifier; the run covers several attributes
    signature: _Signature = ((_op_key(first, structure),), ((),), (True,))
    node = Node(
        id=f"{first.id}__agg",
        op="block",
        label=f"{name} ×{repeats}",
        params={"repeats": repeats, "block_class": name, "ops_per_repeat": 1},
        output_shape=run[-1].output_shape,
        scope=first.scope,
        scope_class=first.scope_class,
    )
    structure[node.id] = _fingerprint(name, repeats, signature)
    return node


def _rewire(edges: list[Edge], collapsed: dict[str, str]) -> list[Edge]:
    """Redirect ``edges`` onto the nodes that replaced their endpoints, dropping self-edges and duplicates."""
    rewired: list[Edge] = []
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        source = collapsed.get(edge.source, edge.source)
        target = collapsed.get(edge.target, edge.target)
        if source == target or (source, target) in seen:
            continue
        seen.add((source, target))
        rewired.append(Edge(source=source, target=target))
    return rewired


def _depth(scope: str) -> int:
    """Nesting depth of a dotted scope path, e.g. ``"blocks.0.attn"`` is depth 3."""
    return scope.count(".") + 1


def _parent_scope(scope: str) -> str | None:
    """The dotted scope path one level up from ``scope``, or ``None`` if ``scope`` is already top-level."""
    parent, _, _ = scope.rpartition(".")
    return parent or None


def _aggregate_one_level(
    nodes: list[Node],
    edges: list[Edge],
    target_depth: int,
    min_repeats: int,
    structure: dict[str, str],
    path_class: dict[str, str],
) -> tuple[list[Node], list[Edge]]:
    """Collapse eligible scope groups at exactly ``target_depth``, leaving other depths untouched."""
    incoming, outgoing = _edge_index(edges)
    segments = _unwrap_containers(_segment_by_scope(nodes, target_depth), path_class)
    items = _merge_runs(segments, incoming, outgoing, structure)

    new_nodes: list[Node] = []
    collapsed: dict[str, str] = {}
    for item in items:
        if isinstance(item, Node):
            new_nodes.append(item)
            continue
        runs = [item] if len(item) >= min_repeats else [[group] for group in item]
        for run in runs:
            synthetic = _synthesize(run, _signature(run[0], incoming, structure), structure)
            parent = _parent_scope(run[0].scope)
            synthetic.scope = parent
            synthetic.scope_class = path_class.get(parent) if parent is not None else None
            new_nodes.append(synthetic)
            for group in run:
                for node in group.nodes:
                    collapsed[node.id] = synthetic.id
    return new_nodes, _rewire(edges, collapsed)


def _unwrap_containers(segments: list[Node | _Group], path_class: dict[str, str]) -> list[Node | _Group]:
    """Let a bare container holding nothing but collapsed blocks pass through to the level above.

    Collapsing it would replace boxes that name themselves — ``BasicBlock ×4``, ``DenseLayer`` — with one named after
    the attribute holding them, ``layer1``, which says strictly less. Its children are lifted to the parent scope
    instead, so the walk continues upward and a real class one level up (``DenseBlock``) still gets its box. A container
    holding raw operations is a different case and still collapses: ``classifier`` is the only name those three layers
    have.

    Args:
        segments: Output of :func:`_segment_by_scope`.
        path_class: Every scope path mapped to its class name.

    Returns:
        The segments with such groups replaced by their nodes, rescoped one level up.
    """
    unwrapped: list[Node | _Group] = []
    for segment in segments:
        if not isinstance(segment, _Group) or segment.scope_class not in _CONTAINERS:
            unwrapped.append(segment)
        elif all(node.op == "block" for node in segment.nodes):
            parent = _parent_scope(segment.scope)
            scope_class = path_class.get(parent) if parent is not None else None
            unwrapped.extend(replace(node, scope=parent, scope_class=scope_class) for node in segment.nodes)
        else:
            unwrapped.append(segment)
    return unwrapped


def _segment_by_scope(nodes: list[Node], target_depth: int) -> list[Node | _Group]:
    """Partition ``nodes`` into non-groupable nodes and maximal same-scope groups at ``target_depth``."""
    segments: list[Node | _Group] = []
    current: _Group | None = None
    for node in nodes:
        scope, scope_class = node.scope, node.scope_class
        if scope is None or scope_class is None or _depth(scope) != target_depth:
            current = None
            segments.append(node)
            continue
        if current is not None and current.scope == scope:
            current.nodes.append(node)
            continue
        current = _Group(scope=scope, scope_class=scope_class, nodes=[node])
        segments.append(current)
    return segments


def _merge_runs(
    segments: list[Node | _Group],
    incoming: dict[str, list[Edge]],
    outgoing: dict[str, list[Edge]],
    structure: dict[str, str],
) -> list[Node | list[_Group]]:
    """Scan ``segments`` for maximal runs of adjacent, eligible, structurally matching groups.

    Groups that fail :func:`_eligible` are dissolved back into their individual nodes here, so everything the caller
    receives as a run is already known to be collapsible.
    """
    items: list[Node | list[_Group]] = []
    i = 0
    while i < len(segments):
        segment = segments[i]
        if isinstance(segment, Node):
            items.append(segment)
            i += 1
            continue
        if not _eligible(segment, incoming, outgoing):
            items.extend(segment.nodes)
            i += 1
            continue
        signature = _signature(segment, incoming, structure)
        run = [segment]
        j = i + 1
        while j < len(segments):
            candidate = segments[j]
            if (
                not isinstance(candidate, _Group)
                or not _eligible(candidate, incoming, outgoing)
                or candidate.scope_class != segment.scope_class
                # Siblings only: merging `blocks.0.layers` with `blocks.1.layers` would put one node under one
                # arbitrary parent, and the repeat count belongs one level up, where the parents themselves repeat.
                or _parent_scope(candidate.scope) != _parent_scope(segment.scope)
                or _signature(candidate, incoming, structure) != signature
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


def _op_key(node: Node, structure: dict[str, str]) -> str:
    """Signature key for one node: what it computes, precise enough that two nodes match only if interchangeable.

    Layer configuration is part of the key, because two blocks are the same block only if their layers are sized the
    same: a VGG stage running 64 channels and the next one running 128 share an op sequence but are not repeats of each
    other, and badging them ``×2`` would claim they are.

    A collapsed block is keyed by the digest of what it collapsed rather than by its class name, since the class of a
    block says nothing about its contents — an ``nn.Sequential`` of two convolutions and one of three are both
    ``Sequential``. Blocks that came from somewhere other than this transform have no digest, so they fall back to the
    class name, which is all a hand-built graph records.
    """
    if node.op == "block":
        return f"block:{structure.get(node.id) or node.params.get('block_class')}"
    return f"{node.op}:{node.params.get('config', '')}"


def _fingerprint(scope_class: str, repeats: int, signature: _Signature) -> str:
    """Digest everything a collapsed run contains, so the level above can compare it in one string comparison."""
    payload = repr((scope_class, repeats, signature))
    return hashlib.blake2s(payload.encode(), digest_size=8).hexdigest()


def _block_name(scope: str, scope_class: str) -> str:
    """Name a collapsed block after its class, or after the attribute holding it when the class is a bare container."""
    return scope_leaf(scope) if scope_class in _CONTAINERS else scope_class


def _signature(group: _Group, incoming: dict[str, list[Edge]], structure: dict[str, str]) -> _Signature:
    """A structural fingerprint of ``group``, independent of node ids, for cross-group comparison."""
    ids = [node.id for node in group.nodes]
    id_set = set(ids)
    index = {node_id: i for i, node_id in enumerate(ids)}
    ops = tuple(_op_key(node, structure) for node in group.nodes)
    topology = []
    external_entry = []
    for i, node_id in enumerate(ids):
        edges_in = incoming.get(node_id, [])
        internal_offsets = tuple(sorted(i - index[edge.source] for edge in edges_in if edge.source in id_set))
        topology.append(internal_offsets)
        external_entry.append(any(edge.source not in id_set for edge in edges_in))
    return ops, tuple(topology), tuple(external_entry)


def _synthesize(run: list[_Group], signature: _Signature, structure: dict[str, str]) -> Node:
    """Build the single synthetic node that replaces a collapsed run of matching groups.

    The node's digest is registered in ``structure`` on the way out, so that when the next level up compares this block
    against its siblings it compares contents rather than class names.
    """
    first, last = run[0], run[-1]
    repeats = len(run)
    name = _block_name(first.scope, first.scope_class)
    label = name if repeats == 1 else f"{name} ×{repeats}"
    node = Node(
        id=f"{first.nodes[0].id}__agg",
        op="block",
        label=label,
        params={"repeats": repeats, "block_class": first.scope_class, "ops_per_repeat": len(first.nodes)},
        output_shape=last.nodes[-1].output_shape,
    )
    structure[node.id] = _fingerprint(first.scope_class, repeats, signature)
    return node
