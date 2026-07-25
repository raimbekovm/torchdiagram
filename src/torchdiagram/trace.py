"""Trace a PyTorch ``nn.Module`` into a :class:`~torchdiagram.graph.Graph`.

Built on ``torch.fx`` symbolic tracing, so anything ``symbolic_trace`` can
handle — residual connections, parallel branches, functional ops inside an
arbitrary ``forward()`` — is captured without touching the model. Data-dependent
control flow is the known limitation of symbolic tracing, and is covered by the
``torch.export`` frontend in :mod:`torchdiagram.export_trace`, which this module
falls back to automatically; see docs/design.md.
"""

from __future__ import annotations

import io
import operator
import os
import sys
from contextlib import redirect_stderr
from types import FrameType

import torch
import torch.fx
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp, TensorMetadata

from .export_trace import trace_export
from .frontend import (
    LEAF_PROBE,
    ExampleInput,
    append_outputs,
    as_args,
    build_edges,
    class_name,
    continues_subscript,
    leaf_root_graph,
    normalize_label,
    parameter_node,
    qualify_labels,
    record_scopes,
    result_keys,
    unique_id,
)
from .graph import Graph, Node

BACKENDS = ("auto", "fx", "export")
"""Accepted values for the ``backend`` argument, shared with the CLI's ``--backend`` choices."""

# Tensor attributes and methods that report metadata rather than data. Reading one starts a computation about the
# tensor (`x.shape[1]`, `x.size(0) // heads`) that the model needs but a diagram of the architecture does not.
_METADATA_ATTRS = frozenset({"shape", "dtype", "device", "ndim", "requires_grad", "is_cuda", "is_leaf"})
_METADATA_METHODS = frozenset({"size", "dim", "numel", "item", "element_size", "stride", "get_device"})

# Plain Python arithmetic and indexing, the only operations metadata is allowed to spread through. A torch function
# fed a shape (`torch.arange(n)`, `x.view(b, -1)`) produces a tensor again and stays in the diagram.
_SCALAR_MODULES = frozenset({"builtins", "operator", "_operator"})

# Where torch itself lives, for telling its frames from the model's while walking the stack.
_TORCH_ROOT = os.path.dirname(torch.__file__)


def trace(
    model: nn.Module,
    example_input: ExampleInput | None = None,
    *,
    name: str | None = None,
    backend: str = "auto",
) -> Graph:
    """Trace ``model`` into a renderable graph.

    Args:
        model: Module to trace; ``forward()`` may be arbitrary fx-traceable code.
        example_input: When given, a forward pass is shape-propagated so every node carries its output shape. Required
            by the ``torch.export`` frontend, which cannot trace without concrete arguments. A model taking more than
            one argument takes a tuple with one tensor per argument, in ``forward()`` order.
        name: Diagram title; defaults to the model's class name.
        backend: Which frontend to use. ``"auto"`` (the default) symbolically traces with ``torch.fx`` and falls back to
            ``torch.export`` if that fails; ``"fx"`` and ``"export"`` pin one frontend.

    Returns:
        The traced graph, with nodes in execution order.

    Raises:
        ValueError: If ``backend`` is not one of ``"auto"``, ``"fx"``, or ``"export"``, or if ``backend="export"`` is
            requested without an ``example_input``.
        torch.fx.proxy.TraceError: If ``model`` is not symbolically traceable and the ``torch.export`` fallback is
            unavailable, disabled by ``backend="fx"``, or itself unable to export the model.

    Warns:
        UserWarning: If the fallback had to specialize the graph on ``example_input``, meaning branches that input does
            not take are absent from the diagram.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r} (expected one of: {', '.join(BACKENDS)})")
    args = None if example_input is None else as_args(example_input)
    if LEAF_PROBE.is_leaf_module(model, ""):
        return leaf_root_graph(model, args, name=name or type(model).__name__)
    if backend == "export":
        if args is None:
            raise ValueError("backend='export' requires an example input, since torch.export traces with real inputs")
        return trace_export(model, args, name=name)
    try:
        return _trace_fx(model, args, name=name)
    except torch.fx.proxy.TraceError as fx_error:
        if backend == "fx":
            raise
        if args is None:
            raise torch.fx.proxy.TraceError(
                f"{fx_error}\n\nPass an example input to fall back to the torch.export frontend, which can trace "
                "data-dependent control flow."
            ) from fx_error
        try:
            return trace_export(model, args, name=name)
        except Exception:
            raise fx_error from None


def _trace_fx(model: nn.Module, args: tuple[torch.Tensor, ...] | None, *, name: str | None) -> Graph:
    """Trace ``model`` with ``torch.fx`` symbolic tracing.

    Args:
        model: Module to trace; ``forward()`` may be arbitrary fx-traceable code.
        args: When given, one example tensor per ``forward()`` argument, shape-propagated so every node carries its
            output shape.
        name: Diagram title; defaults to the model's class name.

    Returns:
        The traced graph, with nodes in execution order.

    Raises:
        torch.fx.proxy.TraceError: If ``model`` is not symbolically traceable, e.g. due to data-dependent control flow
            in ``forward()``.
        ValueError: If ``args`` cannot be run through the model, e.g. an example input with the wrong channel count.
    """
    graph_module = _symbolic_trace(model)
    if args is not None:
        _propagate_shapes(graph_module, args)

    graph = Graph(name=name or type(model).__name__)
    owner: dict[torch.fx.Node, str] = {}
    paths: dict[str, str] = {}
    plumbing = _plumbing(graph_module)
    unpacking, unpacked_shapes = _tuple_unpacking(graph_module)
    used_ids: set[str] = set()
    indexing: dict[torch.fx.Node, Node] = {}  # indexing ops, mapped to the box the whole subscript draws as
    results: list[tuple[str | None, torch.fx.Node]] = []
    for fx_node in graph_module.graph.nodes:
        record_scopes(_stack(fx_node), graph.scopes)
        if fx_node in plumbing:
            continue
        if fx_node.op == "output":
            results = _results(fx_node)
            continue
        if fx_node in unpacking:
            owner[fx_node] = owner[unpacking[fx_node]]  # the layer's box stands in for the unpacking
            continue
        continued = continues_subscript(fx_node, _functional_label(fx_node), indexing)
        if continued is not None:
            owner[fx_node] = continued.id
            continued.output_shape = _output_shape(fx_node)
            indexing[fx_node] = continued
            continue
        node = _to_ir(fx_node, graph_module, used_ids)
        if node is None:
            continue
        if node.op == "index":
            indexing[fx_node] = node
        if fx_node.op == "call_module":
            paths[node.id] = str(fx_node.target)
            if fx_node in unpacked_shapes:
                node.output_shape = unpacked_shapes[fx_node]
        owner[fx_node] = node.id
        graph.nodes.append(node)

    qualify_labels(graph.nodes, paths)
    graph.edges = build_edges(owner)
    append_outputs(graph, owner, results, _output_shape, used_ids)
    return graph


class _LineTracer(torch.fx.Tracer):
    """A tracer that notes which line of the model wrote each node.

    The line is what tells ``x[0][1]`` from two subscripts written a line apart: both lower to the same pair of nodes,
    so the graph alone cannot separate them, and the export frontend — which does record the line — folds the first into
    one box. fx has to know the same thing to agree.

    ``torch.fx`` can supply it: setting ``record_stack_traces`` puts a formatted traceback on every node. It builds that
    by capturing and *formatting* a full stack per node, reading every source file it names, which takes tracing a GPT-2
    from 18 ms to 511 ms — measured on the first trace in a process, which is the only one a command-line run performs.
    Only the innermost line outside torch is ever read here, and reading one frame costs nothing measurable.
    """

    def create_node(self, *args: object, **kwargs: object) -> torch.fx.Node:
        """Create the node the base tracer would, tagged with the model line that produced it."""
        node = super().create_node(*args, **kwargs)  # type: ignore[arg-type]
        node.meta["stack_trace"] = _source_line()
        return node


def _source_line() -> str | None:
    """The innermost frame outside torch, as ``"path:line"``, or ``None`` if the stack never leaves torch.

    fx only ever descends into code the user wrote — every ``torch.nn`` module is a leaf and is traced as a call, not
    stepped through — so the first frame that is not torch's own is the line of ``forward()`` being traced.
    """
    frame: object = sys._getframe(2)  # 0 is this function, 1 is create_node, 2 is whoever asked for a node
    while isinstance(frame, FrameType):
        if not frame.f_code.co_filename.startswith(_TORCH_ROOT):
            return f"{frame.f_code.co_filename}:{frame.f_lineno}"
        frame = frame.f_back
    return None


def _symbolic_trace(model: nn.Module) -> torch.fx.GraphModule:
    """Symbolically trace ``model``, as ``torch.fx.symbolic_trace`` does but keeping each node's source line.

    Args:
        model: Module to trace.

    Returns:
        The traced graph module, every node's metadata carrying ``stack_trace``.

    Raises:
        torch.fx.proxy.TraceError: If ``model`` is not symbolically traceable.
    """
    tracer = _LineTracer()
    graph = tracer.trace(model)
    return torch.fx.GraphModule(tracer.root, graph, type(model).__name__)


class _ShapeProp(ShapeProp):
    """Shape propagation that remembers the node it is on, so a failure can name what rejected the input."""

    current: torch.fx.Node | None = None

    def run_node(self, n: torch.fx.Node) -> object:
        """Record ``n`` as the node in flight, then run it."""
        self.current = n
        return super().run_node(n)


def _propagate_shapes(graph_module: torch.fx.GraphModule, args: tuple[torch.Tensor, ...]) -> None:
    """Annotate every node with its output shape, reporting a mismatched example input in the reader's terms.

    Symbolic tracing has already succeeded by this point, so the failure is never about the model: it is about the
    tensor handed in. Left uncaught it surfaces as six lines of fx internals, an ``nn_module_stack`` repr, and a stack
    through four torch files, with the word ``torchdiagram`` nowhere in it and nothing saying that the fix is to change
    the input shape — which is the single likeliest mistake a first-time user makes.

    Args:
        graph_module: The traced graph module.
        args: One example tensor per ``forward()`` argument.

    Raises:
        ValueError: If ``args`` cannot be run through the model, naming the layer that rejected them and why.
    """
    propagation = _ShapeProp(graph_module)
    try:
        # ShapeProp.run_node calls traceback.print_exc() before re-raising, so the stack it prints reaches stderr
        # whatever the caller does with the exception. Redirecting is the only way to keep the error to one line.
        with redirect_stderr(io.StringIO()):
            propagation.propagate(*args)
    except Exception as error:
        shapes = ", ".join(str(tuple(tensor.shape)) for tensor in args)
        raise ValueError(
            f"example input {shapes} does not run through this model: {_blame(graph_module, propagation.current)} "
            f"rejected it — {_root_cause(error)}"
        ) from error


def _blame(graph_module: torch.fx.GraphModule, fx_node: torch.fx.Node | None) -> str:
    """Name the step that rejected the example input, as a layer where there is one."""
    if fx_node is None:
        return "the model"
    if fx_node.op == "call_module":
        return f"layer {fx_node.target!r} ({type(graph_module.get_submodule(str(fx_node.target))).__name__})"
    return f"{fx_node.op.replace('_', ' ')} {fx_node.name!r}"


def _root_cause(error: BaseException) -> str:
    """The innermost message in an exception chain, which is the one that says what was actually wrong."""
    cause: BaseException = error
    while cause.__cause__ is not None:
        cause = cause.__cause__
    return str(cause).strip().splitlines()[0]


def _tuple_unpacking(
    graph_module: torch.fx.GraphModule,
) -> tuple[dict[torch.fx.Node, torch.fx.Node], dict[torch.fx.Node, tuple[int, ...] | None]]:
    """Find the nodes that unpack a layer's tuple return, and what shape each such layer really produces.

    ``out, _ = self.lstm(x)`` draws four boxes today: the ``LSTM``, carrying no shape because its fx output is a tuple
    and shape propagation only records one for a tensor; a ``getitem`` carrying the shape the LSTM should have; and a
    second ``getitem`` for the discarded hidden state, which nothing consumes and which is a dead stub on the diagram.
    Two of the four are Python syntax, and the one a reader cares about is the unlabeled one.

    Unpacking is folded into the layer instead: the ``getitem`` nodes stand in for the layer's box, and the layer takes
    the shape of the element the rest of the graph actually consumes. Subscripting a layer that returns a plain tensor
    (``self.conv(x)[0]``) is a real operation and is left alone; a tuple return is told apart by shape propagation
    recording a tuple, or, with no example input to propagate, by the layer being selected from at more than one index,
    which is what tuple unpacking always generates.

    Args:
        graph_module: The traced graph module.

    Returns:
        Each unpacking node mapped to its layer, and each such layer mapped to the shape it should carry.
    """
    unpacking: dict[torch.fx.Node, torch.fx.Node] = {}
    shapes: dict[torch.fx.Node, tuple[int, ...] | None] = {}
    for fx_node in graph_module.graph.nodes:
        if not _selects(fx_node):
            continue
        source = fx_node.args[0]
        if not _returns_tuple(source):
            continue  # the source is a tensor, so this is a subscript of it — a real operation, not unpacking
        # `_, (h, c) = self.rnn(x)` selects the state tuple out of the layer and then selects again out of that, so a
        # selection whose source is itself unpacking belongs to the same layer.
        if source in unpacking:
            layer = unpacking[source]
        elif source.op == "call_module":
            layer = source
        else:
            continue
        unpacking[fx_node] = layer
        # The layer's real output is the first selected element anything downstream consumes, and only a tensor has a
        # shape to take — a selection that lands on another tuple is a step on the way, not the answer.
        if fx_node.users and _output_shape(fx_node) is not None and shapes.get(layer) is None:
            shapes[layer] = _output_shape(fx_node)
    return unpacking, shapes


def _selects(fx_node: torch.fx.Node) -> bool:
    """Whether ``fx_node`` selects a fixed position out of another node's return value."""
    return (
        fx_node.op == "call_function"
        and fx_node.target is operator.getitem
        and len(fx_node.args) == 2
        and isinstance(fx_node.args[0], torch.fx.Node)
        and isinstance(fx_node.args[1], int)
    )


def _returns_tuple(fx_node: torch.fx.Node) -> bool:
    """Whether a node produces a tuple the model unpacks, rather than a tensor the model subscripts.

    Asked of a layer, this separates ``out, _ = self.rnn(x)`` from ``self.conv(x)[0]``. Asked of an unpacking node, it
    separates a further step of the same unpacking — the state tuple in ``_, (h, c) = self.rnn(x)`` — from a subscript
    of the tensor it landed on, ``hidden[-1]``, which is a real operation and keeps its box.
    """
    meta = fx_node.meta.get("tensor_meta")
    if meta is not None:
        # TensorMetadata is itself a NamedTuple, so identity is the test, not "is this a tuple".
        return not isinstance(meta, TensorMetadata)
    # No example input, so nothing propagated a shape to read. `self.rnn(x)[0]` and `self.conv(x)[0]` look identical
    # from the graph alone, and answering "it is a tuple" only when two indices are selected would make the same model
    # draw a different box count with and without shapes. Where torch defines the return, the class settles it; the
    # two-index rule catches a custom module unpacked the ordinary way.
    module = fx_node.graph.owning_module
    if fx_node.op == "call_module" and module is not None:
        resolved = module.get_submodule(str(fx_node.target))
        if isinstance(resolved, (nn.RNNBase, nn.MultiheadAttention)):
            return True
    return len({user.args[1] for user in fx_node.users if _selects(user)}) > 1


def _results(fx_node: torch.fx.Node) -> list[tuple[str | None, torch.fx.Node]]:
    """Read what an fx ``output`` node returns as ``(key, producing node)`` pairs.

    fx keeps the returned structure intact in the node's single argument: a bare node for ``return y``, a tuple for
    ``return p3, p4, p5``, a dict for a HuggingFace-style return. Anything in there that isn't a node is a constant the
    model returns unchanged, which has no producer to draw an arrow from.

    Args:
        fx_node: The graph's ``output`` node.

    Returns:
        One pair per returned tensor, in return order.
    """
    result = fx_node.args[0] if fx_node.args else None
    if isinstance(result, torch.fx.Node):
        return [(None, result)]
    if isinstance(result, dict):
        structure, names = "dict", list(result)
    elif isinstance(result, (tuple, list)):
        structure, names = "tuple", None
    else:
        return []
    values = _flatten(result)
    keys = result_keys(len(values), structure, names)
    return [(key, value) for key, value in zip(keys, values, strict=True) if isinstance(value, torch.fx.Node)]


def _flatten(result: object) -> list[object]:
    """Every leaf of a returned structure, in return order.

    A model returning ``(x, (p3, p4, p5))`` returns four tensors, and export flattens the structure away before the
    graph's ``output`` node ever sees it. Taking fx's nested argument at face value instead would drop the whole inner
    tuple, leaving its producers with no outgoing edge and dangling off the diagram.
    """
    if isinstance(result, dict):
        return [leaf for value in result.values() for leaf in _flatten(value)]
    if isinstance(result, (tuple, list)):
        return [leaf for value in result for leaf in _flatten(value)]
    return [result]


def _plumbing(graph_module: torch.fx.GraphModule) -> set[torch.fx.Node]:
    """Collect the nodes that compute with a tensor's metadata rather than with the tensor.

    ``x.shape[1]`` traces as a ``getattr`` feeding a ``getitem``, and an attention head's ``c // self.heads`` adds a
    ``floordiv`` on top. None of them is a layer, none carries a shape to annotate, and together they leave a diagram
    with a row of boxes reading ``getattr``, ``getitem``, ``getattr``. The ``torch.export`` frontend already drops their
    equivalents, so keeping them here would also break the promise that both frontends emit the same IR.

    A torch call that consumes a shape is not plumbing: ``torch.arange(idx.shape[1])`` produces a real tensor and is
    part of the model, so only plain Python arithmetic and indexing propagate the classification.

    Args:
        graph_module: The traced graph module.

    Returns:
        The nodes to leave out of the diagram.
    """
    plumbing: set[torch.fx.Node] = set()
    for fx_node in graph_module.graph.nodes:  # forward order, so a producer is classified before its consumers
        if _reads_metadata(fx_node) or _computes_on(plumbing, fx_node):
            plumbing.add(fx_node)
    return plumbing


def _reads_metadata(fx_node: torch.fx.Node) -> bool:
    """Whether ``fx_node`` asks a tensor about itself, e.g. ``x.shape`` or ``x.size(0)``."""
    if fx_node.op == "call_function" and fx_node.target is getattr:
        return len(fx_node.args) > 1 and fx_node.args[1] in _METADATA_ATTRS
    return fx_node.op == "call_method" and str(fx_node.target) in _METADATA_METHODS


def _computes_on(plumbing: set[torch.fx.Node], fx_node: torch.fx.Node) -> bool:
    """Whether ``fx_node`` is plain Python arithmetic or indexing over nothing but metadata."""
    if fx_node.op != "call_function" or getattr(fx_node.target, "__module__", "") not in _SCALAR_MODULES:
        return False
    inputs = fx_node.all_input_nodes
    return bool(inputs) and all(upstream in plumbing for upstream in inputs)


def _to_ir(fx_node: torch.fx.Node, graph_module: torch.fx.GraphModule, used_ids: set[str]) -> Node | None:
    """Convert a single fx node into an IR ``Node``, or ``None`` if it carries no diagram content.

    Args:
        fx_node: Node from the traced fx graph.
        graph_module: The graph module ``fx_node`` belongs to, used to resolve submodules.
        used_ids: Ids already handed out, so a functional node can be numbered off its normalized name rather than off
            the tracer's own — which is what keeps the two frontends' ids matching once a name is normalized.

    Returns:
        The corresponding IR node, or ``None`` for ``get_attr`` nodes (parameter/buffer plumbing).
    """
    extra = ""  # only a leaf layer has hyperparameters to record
    node_id = fx_node.name
    if fx_node.op == "placeholder":
        op = label = "input"
    elif fx_node.op == "call_module":
        module = graph_module.get_submodule(str(fx_node.target))
        label = type(module).__name__
        op = label.lower()
        extra = module.extra_repr()
        # Numbered off the submodule path, not off fx's own node name: fx strips a trailing `_<digits>` before
        # renumbering, so a second call of `body.0` becomes `body_2` there and `body_0_1` here.
        node_id = unique_id(str(fx_node.target).replace(".", "_"), used_ids)
    elif fx_node.op in ("call_function", "call_method"):
        op = label = _functional_label(fx_node)
        node_id = unique_id(op, used_ids)
    elif fx_node.op == "get_attr":
        # fx only emits get_attr for an attribute the traced forward() reads itself; a layer's own weights stay inside
        # its call_module node, so everything reaching here is a tensor the model uses in its own code.
        node = parameter_node(graph_module, str(fx_node.target), _output_shape(fx_node))
        used_ids.add(node.id)
        return node
    else:
        return None

    used_ids.add(node_id)
    scope, scope_class = _scope(fx_node)
    return Node(
        id=node_id,
        op=op,
        label=label,
        params={"config": extra} if extra else {},
        output_shape=_output_shape(fx_node),
        scope=scope,
        scope_class=scope_class,
    )


def _functional_label(fx_node: torch.fx.Node) -> str:
    """The label a non-module node draws under, normalized to the name the export frontend uses for it."""
    if fx_node.op == "call_function":
        return normalize_label(getattr(fx_node.target, "__name__", str(fx_node.target)))
    if fx_node.op == "call_method":
        return normalize_label(str(fx_node.target))
    return ""


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
    entries = _stack(fx_node)
    if fx_node.op == "call_module":
        entries = entries[:-1]  # drop the node's own leaf module, keep its parent
    if not entries:
        return None, None
    path, cls = entries[-1]
    return path, class_name(cls)


def _stack(fx_node: torch.fx.Node) -> list[tuple[str, object]]:
    """Read ``nn_module_stack`` as ``(path, class)`` entries, outermost first."""
    stack = fx_node.meta.get("nn_module_stack")
    return list(stack.values()) if stack else []
