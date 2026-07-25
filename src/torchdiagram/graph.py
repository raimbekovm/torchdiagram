"""Framework-agnostic intermediate representation (IR) of a model.

Frontends (:mod:`torchdiagram.trace`) produce a :class:`Graph`; renderers
(:mod:`torchdiagram.renderers`) consume one. Nothing here imports torch, so
graphs can also be built by hand or by future non-PyTorch frontends.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


def scope_leaf(path: str) -> str:
    """The attribute name at the end of a dotted module path, e.g. ``"features.3.conv"`` gives ``"conv"``.

    A numeric leaf is a list index rather than a name — ``nn.Sequential`` and ``nn.ModuleList`` children are ``"0"``,
    ``"1"``, and so on — so the segment above it is kept alongside, giving ``"blocks.0"`` rather than a bare ``"0"``.

    Args:
        path: Dotted submodule path, as recorded in :attr:`Node.scope`.

    Returns:
        The readable tail of the path.
    """
    head, _, leaf = path.rpartition(".")
    if leaf.isdigit() and head:
        return f"{head.rpartition('.')[2]}.{leaf}"
    return leaf


@dataclass
class Node:
    """A single block in the diagram: a layer, a function call, or a graph input/output.

    Attributes:
        id: Unique identifier, referenced by edges.
        op: Normalized operation kind, e.g. ``"conv2d"``, ``"add"``, ``"input"``.
        label: Human-readable text shown on the diagram.
        params: Layer configuration, e.g. ``{"config": "3, 16, kernel_size=(3, 3)"}``.
        output_shape: Output tensor shape, populated when tracing runs with an example input.
        scope: Dotted path of the immediate custom-container module this node was traced from, e.g. ``"layer1.0"``, or
            ``None`` if the node isn't nested in one.
        scope_class: Class name of that container, e.g. ``"BasicBlock"``. Normally ``None`` exactly when ``scope`` is,
            but a synthetic node produced by :func:`torchdiagram.transforms.aggregate_blocks` can carry a non-``None``
            ``scope`` with ``scope_class=None`` when the parent container's own class name couldn't be resolved.
    """

    id: str
    op: str
    label: str
    params: dict[str, Any] = field(default_factory=dict)
    output_shape: tuple[int, ...] | None = None
    scope: str | None = None
    scope_class: str | None = None

    @property
    def is_io(self) -> bool:
        """Whether this node is a graph input or output, rendered in a distinct style."""
        return self.op in ("input", "output")


@dataclass
class Edge:
    """A directed data-flow edge between two nodes, referenced by node id.

    Attributes:
        source: Id of the source node.
        target: Id of the target node.
    """

    source: str
    target: str


@dataclass
class Graph:
    """An ordered model graph; node order is the topological (execution) order.

    Attributes:
        name: Diagram title.
        nodes: Nodes in execution order.
        edges: Directed data-flow edges.
        scopes: Every container module the traced nodes sit under, as dotted path mapped to class name — including the
            levels that own no operation of their own, such as a bare ``nn.Sequential`` between two custom modules.
            Those levels appear as no node's ``scope``, so without this map the module tree has holes in it and
            :func:`torchdiagram.transforms.aggregate_blocks` cannot walk past them. Empty for a hand-built graph.
    """

    name: str = "model"
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    scopes: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """Check internal consistency of the graph.

        Raises:
            ValueError: If two nodes share an id, or an edge references an id not present in ``nodes``.
        """
        counts = Counter(node.id for node in self.nodes)
        duplicates = sorted(node_id for node_id, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate node ids: {duplicates}")
        for edge in self.edges:
            for ref in (edge.source, edge.target):
                if ref not in counts:
                    raise ValueError(f"edge references unknown node id: {ref!r}")
