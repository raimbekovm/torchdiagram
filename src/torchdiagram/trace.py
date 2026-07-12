"""Trace a PyTorch ``nn.Module`` into a :class:`~torchdiagram.graph.Graph`.

Built on ``torch.fx`` symbolic tracing, so anything ``symbolic_trace`` can
handle — residual connections, parallel branches, functional ops inside an
arbitrary ``forward()`` — is captured without touching the model. Data-dependent
control flow is the known limitation of symbolic tracing; see docs/design.md
for the planned fallbacks.
"""

from __future__ import annotations

import torch
import torch.fx
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from .graph import Edge, Graph, Node


def trace(
    model: nn.Module,
    example_input: torch.Tensor | None = None,
    *,
    name: str | None = None,
) -> Graph:
    """Trace ``model`` into a renderable graph.

    Args:
        model: Module to trace; ``forward()`` may be arbitrary fx-traceable code.
        example_input: When given, a forward pass is shape-propagated so every node carries its output shape.
        name: Diagram title; defaults to the model's class name.

    Returns:
        The traced graph, with nodes in execution order.

    Raises:
        torch.fx.proxy.TraceError: If ``model`` is not symbolically traceable, e.g. due to data-dependent control flow
            in ``forward()``.
    """
    graph_module = torch.fx.symbolic_trace(model)
    if example_input is not None:
        ShapeProp(graph_module).propagate(example_input)

    graph = Graph(name=name or type(model).__name__)
    kept: dict[torch.fx.Node, str] = {}
    for fx_node in graph_module.graph.nodes:
        node = _to_ir(fx_node, graph_module)
        if node is None:
            continue
        kept[fx_node] = node.id
        graph.nodes.append(node)

    for fx_node, target_id in kept.items():
        for upstream in fx_node.all_input_nodes:
            if upstream in kept:
                graph.edges.append(Edge(source=kept[upstream], target=target_id))
    return graph


def _to_ir(fx_node: torch.fx.Node, graph_module: torch.fx.GraphModule) -> Node | None:
    """Convert a single fx node into an IR ``Node``, or ``None`` if it carries no diagram content.

    Args:
        fx_node: Node from the traced fx graph.
        graph_module: The graph module ``fx_node`` belongs to, used to resolve submodules.

    Returns:
        The corresponding IR node, or ``None`` for ``get_attr`` nodes (parameter/buffer plumbing).
    """
    shape = _output_shape(fx_node)
    scope, scope_class = _scope(fx_node)
    if fx_node.op == "placeholder":
        return Node(
            id=fx_node.name, op="input", label="input", output_shape=shape, scope=scope, scope_class=scope_class
        )
    if fx_node.op == "output":
        return Node(
            id=fx_node.name, op="output", label="output", output_shape=shape, scope=scope, scope_class=scope_class
        )
    if fx_node.op == "call_module":
        module = graph_module.get_submodule(str(fx_node.target))
        kind = type(module).__name__
        extra = module.extra_repr()
        return Node(
            id=fx_node.name,
            op=kind.lower(),
            label=kind,
            params={"config": extra} if extra else {},
            output_shape=shape,
            scope=scope,
            scope_class=scope_class,
        )
    if fx_node.op == "call_function":
        label = getattr(fx_node.target, "__name__", str(fx_node.target))
        return Node(id=fx_node.name, op=label, label=label, output_shape=shape, scope=scope, scope_class=scope_class)
    if fx_node.op == "call_method":
        label = str(fx_node.target)
        return Node(id=fx_node.name, op=label, label=label, output_shape=shape, scope=scope, scope_class=scope_class)
    return None  # get_attr: parameter/buffer plumbing, not a diagram block


def _output_shape(fx_node: torch.fx.Node) -> tuple[int, ...] | None:
    """Read the shape-propagated output shape of ``fx_node``, if available."""
    meta = fx_node.meta.get("tensor_meta")
    shape = getattr(meta, "shape", None)
    return tuple(shape) if shape is not None else None


def _scope(fx_node: torch.fx.Node) -> tuple[str | None, str | None]:
    """Resolve the immediate custom-container ancestor of ``fx_node``, or ``(None, None)``.

    Reads ``nn_module_stack``, which fx populates for every node kind (not just ``call_module``), so functional ops
    called from inside a submodule's ``forward()`` (e.g. a residual ``add``) still resolve to that submodule.
    """
    stack = fx_node.meta.get("nn_module_stack")
    if not stack:
        return None, None
    entries = list(stack.items())
    if fx_node.op == "call_module":
        entries = entries[:-1]  # drop the node's own leaf module, keep its parent
    if not entries:
        return None, None
    path, cls = entries[-1][1]
    return path, cls.__name__
