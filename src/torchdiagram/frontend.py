"""Helpers shared by the two tracing frontends.

:mod:`torchdiagram.trace` and :mod:`torchdiagram.export_trace` disagree about how a model is traced, but both end up
walking a ``torch.fx`` graph and deciding which of its nodes become diagram blocks. What happens around that decision
— reconnecting the flow across the nodes they drop, and naming the blocks they keep — has to be identical, since both
frontends are contracted to emit the same IR for the same model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.fx

from .graph import Edge, Graph, Node, scope_leaf

ExampleInput = torch.Tensor | tuple[torch.Tensor, ...]
"""One example tensor, or one per positional argument of ``forward()``."""


def as_args(example_input: ExampleInput) -> tuple[torch.Tensor, ...]:
    """Normalize an example input into the argument tuple every call site downstream already wants.

    ``ShapeProp.propagate``, ``torch.export.export``, and ``draft_export`` all take one argument per parameter of
    ``forward()``; a bare tensor is the one-argument shorthand, so a two-tower or encoder-decoder model passes a tuple.

    Args:
        example_input: A single tensor, or a tuple with one per ``forward()`` argument.

    Returns:
        The arguments as a tuple.
    """
    return example_input if isinstance(example_input, tuple) else (example_input,)


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


def append_outputs(
    graph: Graph,
    owner: dict[torch.fx.Node, str],
    results: Sequence[tuple[str | None, torch.fx.Node]],
    shape: Callable[[torch.fx.Node], tuple[int, ...] | None],
) -> None:
    """Add one ``output`` node per returned tensor, each edged from whatever produced it.

    A model returning ``(p3, p4, p5)`` returns three separate tensors that the caller unpacks; drawing them as three
    arrows into one box says they merge, which is the one reading that is definitely wrong. Each gets its own box,
    carrying its own shape and the tuple index or dict key it was returned under, so the diagram distinguishes "returns
    three tensors" from "returns one". A lone return keeps the plain ``output`` box it always had.

    Args:
        graph: Graph to append to. Modified in place.
        owner: Every kept fx node, mapped to the id of the IR node representing it.
        results: The returned tensors as ``(key, producing fx node)`` pairs, in return order; ``key`` is ``None`` for a
            model that returns a single tensor.
        shape: Reads a node's output shape, which the two frontends record differently.
    """
    for key, producer in results:
        node_id, label = _output_identity(key)
        graph.nodes.append(Node(id=node_id, op="output", label=label, output_shape=shape(producer)))
        for source_id in [owner[producer]] if producer in owner else _sources(producer, owner):
            graph.edges.append(Edge(source=source_id, target=node_id))


def _output_identity(key: str | None) -> tuple[str, str]:
    """Id and label for one returned tensor, keyed by its tuple index or dict key."""
    return ("output", "output") if key is None else (f"output_{key}", f"output[{key}]")


def result_keys(count: int, structure: str | None, names: Sequence[str] | None) -> list[str | None]:
    """Name each returned tensor by its dict key or tuple index, or leave a single return unnamed.

    Args:
        count: How many tensors the model returns.
        structure: ``"dict"``, ``"tuple"``, or ``None`` when the model returns one bare tensor.
        names: Dict keys in return order, when ``structure`` is ``"dict"``.

    Returns:
        One key per returned tensor, or ``[None]`` for a single unwrapped return.
    """
    if structure is None and count == 1:
        return [None]
    if structure == "dict" and names is not None and len(names) == count:
        return [str(name) for name in names]
    return [str(index) for index in range(count)]


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
