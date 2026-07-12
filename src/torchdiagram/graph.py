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
    """A single block in the diagram: a layer, a function call, or a graph input/output."""

    id: str
    op: str
    """Normalized operation kind, e.g. ``"conv2d"``, ``"add"``, ``"input"``."""
    label: str
    """Human-readable text shown on the diagram."""
    params: dict[str, Any] = field(default_factory=dict)
    """Layer configuration, e.g. ``{"config": "3, 16, kernel_size=(3, 3)"}``."""
    output_shape: tuple[int, ...] | None = None


@dataclass
class Edge:
    """A directed data-flow edge between two nodes, referenced by node id."""

    source: str
    target: str


@dataclass
class Graph:
    """An ordered model graph; node order is the topological (execution) order."""

    name: str = "model"
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def validate(self) -> None:
        """Raise :class:`ValueError` on duplicate node ids or dangling edge references."""
        ids = [node.id for node in self.nodes]
        unique = set(ids)
        if len(unique) != len(ids):
            duplicates = sorted({node_id for node_id in ids if ids.count(node_id) > 1})
            raise ValueError(f"duplicate node ids: {duplicates}")
        for edge in self.edges:
            for ref in (edge.source, edge.target):
                if ref not in unique:
                    raise ValueError(f"edge references unknown node id: {ref!r}")
