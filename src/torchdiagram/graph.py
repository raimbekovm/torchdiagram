"""Framework-agnostic intermediate representation (IR) of a model.

Frontends (:mod:`torchdiagram.trace`) produce a :class:`Graph`; renderers
(:mod:`torchdiagram.renderers`) consume one. Nothing here imports torch, so
graphs can also be built by hand or by future non-PyTorch frontends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """A single block in the diagram: a layer, a function call, or a graph input/output.

    Attributes:
        id: Unique identifier, referenced by edges.
        op: Normalized operation kind, e.g. ``"conv2d"``, ``"add"``, ``"input"``.
        label: Human-readable text shown on the diagram.
        params: Layer configuration, e.g. ``{"config": "3, 16, kernel_size=(3, 3)"}``.
        output_shape: Output tensor shape, populated when tracing runs with an example input.
    """

    id: str
    op: str
    label: str
    params: dict[str, Any] = field(default_factory=dict)
    output_shape: tuple[int, ...] | None = None

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
    """

    name: str = "model"
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def validate(self) -> None:
        """Check internal consistency of the graph.

        Raises:
            ValueError: If two nodes share an id, or an edge references an id not present in ``nodes``.
        """
        ids = [node.id for node in self.nodes]
        unique = set(ids)
        if len(unique) != len(ids):
            duplicates = sorted({node_id for node_id in ids if ids.count(node_id) > 1})
            raise ValueError(f"duplicate node ids: {duplicates}")
        for edge in self.edges:
            for ref in (edge.source, edge.target):
                if ref not in unique:
                    raise ValueError(f"edge references unknown node id: {ref!r}")
