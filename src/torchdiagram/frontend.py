"""Helpers shared by the two tracing frontends.

:mod:`torchdiagram.trace` and :mod:`torchdiagram.export_trace` disagree about how a model is traced, but both end up
walking a ``torch.fx`` graph and deciding which of its nodes become diagram blocks. What happens around that decision
— reconnecting the flow across the nodes they drop, and naming the blocks they keep — has to be identical, since both
frontends are contracted to emit the same IR for the same model.
"""

from __future__ import annotations

import torch.fx

from .graph import Edge, Node, scope_leaf


def build_edges(owner: dict[torch.fx.Node, str]) -> list[Edge]:
    """Build the IR edges between kept nodes, routing around the ones left out of the diagram.

    A dropped node must not break the chain. ``torch.arange(idx.shape[1])`` drops the two nodes that read the shape, and
    the range would be left floating with no incoming edge if its inputs were simply discarded; instead the search
    continues upward and connects it to the input tensor the shape came from. Parameter plumbing and guard chains have
    no kept ancestors, so they reconnect to nothing, as before.

    Args:
        owner: Every kept fx node, mapped to the id of the IR node representing it. Several fx nodes may map to one id
            when a layer lowered to more than one op; the self-edges that implies are dropped.

    Returns:
        The deduplicated edges, in the order the kept nodes were visited.
    """
    edges: list[Edge] = []
    seen: set[tuple[str, str]] = set()
    for fx_node, target_id in owner.items():
        for source_id in _sources(fx_node, owner):
            if source_id == target_id or (source_id, target_id) in seen:
                continue
            seen.add((source_id, target_id))
            edges.append(Edge(source=source_id, target=target_id))
    return edges


def _sources(fx_node: torch.fx.Node, owner: dict[torch.fx.Node, str]) -> list[str]:
    """Ids of the kept nodes feeding ``fx_node``, looking through any dropped nodes in between."""
    found: list[str] = []
    visited: set[torch.fx.Node] = set()
    queue = list(fx_node.all_input_nodes)
    while queue:
        upstream = queue.pop(0)
        if upstream in visited:
            continue
        visited.add(upstream)
        node_id = owner.get(upstream)
        if node_id is None:
            queue.extend(upstream.all_input_nodes)
        else:
            found.append(node_id)
    return found


def qualify_labels(nodes: list[Node], paths: dict[str, str]) -> None:
    """Disambiguate layers that would otherwise draw identical boxes, using the attribute each was traced from.

    A transformer's token and position embeddings are both an ``nn.Embedding`` at the same level of the model, so the
    class name alone labels two boxes the same and says nothing about which is which. Where several nodes in one scope
    share a label, each is qualified with its attribute name — ``Embedding (tok_emb)`` — and layers that are alone under
    their label keep the plain class name, which is what most of a diagram is.

    Args:
        nodes: The IR nodes, labeled and in execution order. Modified in place.
        paths: Node id mapped to the dotted submodule path it was traced from, for module nodes only; functional nodes
            have no attribute to name and are left alone.
    """
    groups: dict[tuple[str | None, str], list[Node]] = {}
    for node in nodes:
        if node.id in paths:
            groups.setdefault((node.scope, node.label), []).append(node)
    for group in groups.values():
        names = [scope_leaf(paths[node.id]) for node in group]
        if len(group) < 2 or len(set(names)) < len(group):
            continue  # nothing to tell apart, or the attribute names don't tell them apart either
        for node, name in zip(group, names, strict=True):
            node.label = f"{node.label} ({name})"
