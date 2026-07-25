"""Fallback frontend: trace a model with ``torch.export`` when ``torch.fx`` cannot.

``torch.fx.symbolic_trace`` gives up on data-dependent control flow (``if x.sum() > 0:``). This frontend runs the
model through ``torch.export`` instead and emits the same :class:`~torchdiagram.graph.Graph`, so nothing downstream
changes. Plain ``torch.export`` traces with fake tensors and hits the same guard fx does, so the branch-resolving
path is ``torch.export.draft_export``, which propagates real tensors alongside the fake ones and **specializes** on
the example input: the diagram then shows the branch that input takes, and no other.

The export graph is at ATen level, so module identity is recovered from ``nn_module_stack`` rather than read off
``call_module`` nodes; see docs/design.md for why that reproduces the fx frontend's labels.
"""

from __future__ import annotations

import io
import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr

import torch
import torch.fx
from torch import nn

from .frontend import (
    LEAF_PROBE,
    ExampleInput,
    append_outputs,
    as_args,
    build_edges,
    normalize_label,
    qualify_labels,
    result_keys,
    unique_id,
)
from .graph import Graph, Node

# Export narrates a failed attempt loudly: draft_export logs a tlparse banner about unsound specialization, and
# torch.export._trace prints the partial graph straight to stderr with no logger in between. Both are noise here —
# a failed attempt is the expected path for the models this frontend exists to handle — so _quiet_export() suppresses
# them and trace_export() says the one thing that matters in a single warning.
_LOG_PREFIX = "torch"

_Entries = list[tuple[str, object]]
"""A node's ``nn_module_stack`` as ``(submodule path, class)`` pairs, outermost first."""


def trace_export(model: nn.Module, example_input: ExampleInput, *, name: str | None = None) -> Graph:
    """Trace ``model`` into a renderable graph using ``torch.export``.

    Args:
        model: Module to trace. Unlike the fx frontend, ``forward()`` may contain data-dependent control flow.
        example_input: Input to trace with, as one tensor or a tuple with one per ``forward()`` argument. Required —
            ``torch.export`` needs concrete arguments — and used for output-shape annotation, which comes free with the
            export graph.
        name: Diagram title; defaults to the model's class name.

    Returns:
        The traced graph, with nodes in execution order.

    Raises:
        Exception: Whatever ``torch.export`` raises if the model cannot be exported at all. The exception type depends
            on how the export failed, so callers that want a stable type should catch broadly.

    Warns:
        UserWarning: If the graph had to be specialized on ``example_input``, meaning branches that input does not take
            are missing from the diagram.
    """
    graph_module, specialized = _export(model, as_args(example_input))
    if specialized:
        warnings.warn(
            f"{type(model).__name__} was traced with torch.export and specialized on the example input: branches "
            "not taken by this input are absent from the diagram",
            UserWarning,
            stacklevel=2,
        )
    return _build_graph(model, graph_module, name=name or type(model).__name__)


def _export(model: nn.Module, args: tuple[torch.Tensor, ...]) -> tuple[torch.fx.GraphModule, bool]:
    """Export ``model``, preferring a sound graph and falling back to a specialized one.

    Args:
        model: Module to export.
        args: One example tensor per ``forward()`` argument.

    Returns:
        The exported graph module, and whether it was specialized on ``args`` rather than soundly exported.

    Raises:
        Exception: The error from the sound attempt, if the specializing fallback is unavailable or also fails.
    """
    with _quiet_export():
        try:
            return torch.export.export(model, args, strict=False).module(), False
        except Exception as sound_error:
            draft_export = getattr(torch.export, "draft_export", None)
            if draft_export is None:  # torch too old to have the real-tensor path
                raise
            try:
                program = draft_export(model, args)
            except Exception:
                raise sound_error from None
    # `_report` is private, so treat its absence as "assume the worst" rather than silently claiming soundness.
    report = getattr(program, "_report", None)
    return program.module(), not (report is not None and report.successful())


def _drop(record: logging.LogRecord) -> bool:
    """Reject ``record`` so it never reaches the handler's stream."""
    return False


@contextmanager
def _quiet_export() -> Iterator[None]:
    """Silence torch's narration of a failed export attempt for the duration of the block.

    Both channels have to be covered: the log records go through handlers torch installs on its own loggers, and the
    partial-graph dump is a bare ``print(..., file=sys.stderr)`` that only a redirected stream catches. Real errors
    still surface — they travel as exceptions, not as log output.

    Muting happens on the handlers rather than by raising logger levels, because ``draft_export`` builds its
    specialization report from those very records: silencing the loggers would empty ``report.failures`` and make an
    unsound graph look sound, which is exactly the signal this frontend needs to keep.
    """
    names = [name for name in logging.root.manager.loggerDict if name.split(".")[0] == _LOG_PREFIX]
    handlers = {handler for name in [_LOG_PREFIX, *names] for handler in logging.getLogger(name).handlers}
    for handler in handlers:
        handler.addFilter(_drop)
    try:
        with redirect_stderr(io.StringIO()):
            yield
    finally:
        for handler in handlers:
            handler.removeFilter(_drop)


def _build_graph(model: nn.Module, graph_module: torch.fx.GraphModule, *, name: str) -> Graph:
    """Convert an exported graph module into the IR.

    Args:
        model: The original module, used to resolve submodules by the paths recorded in ``nn_module_stack``.
        graph_module: Result of ``ExportedProgram.module()``.
        name: Diagram title.

    Returns:
        The IR graph, with nodes in execution order.
    """
    graph = Graph(name=name)
    owner: dict[torch.fx.Node, str] = {}  # every kept fx node, mapped to the IR node that represents it
    by_module: dict[str, Node] = {}
    used_ids: set[str] = set()
    paths: dict[str, str] = {}  # IR node id mapped to the submodule it came from, for label disambiguation
    plumbing = _plumbing(graph_module)
    indexing: dict[torch.fx.Node, Node] = {}  # indexing ops, mapped to the box the whole subscript draws as
    results: list[tuple[str | None, torch.fx.Node]] = []

    for fx_node in graph_module.graph.nodes:
        if fx_node.op == "output":
            results = _results(graph_module, fx_node)
            continue
        if fx_node.op == "placeholder":
            node = Node(id=unique_id(fx_node.name, used_ids), op="input", label="input", output_shape=_shape(fx_node))
        elif fx_node.op != "call_function" or fx_node in plumbing:
            continue  # get_attr parameter plumbing, the synthetic _guards_fn module, and symbolic guard assertions
        else:
            path, module, ancestors = _leaf_module(model, _stack(fx_node))
            if module is None:
                label = _label(fx_node.target)
                continued = _continues_subscript(fx_node, label, indexing)
                if continued is not None:
                    owner[fx_node] = continued.id
                    continued.output_shape = _shape(fx_node)
                    indexing[fx_node] = continued
                    continue
                scope, scope_class = _scope(ancestors)
                node = Node(
                    # Numbered off the normalized label rather than off export's own node name, which is derived from
                    # the ATen overload and so would read `slice_1` where fx reads `index`.
                    id=unique_id(label, used_ids),
                    op=label,
                    label=label,
                    output_shape=_shape(fx_node),
                    scope=scope,
                    scope_class=scope_class,
                )
                if label == "index":
                    indexing[fx_node] = node
            elif path in by_module:
                # A single layer can lower to several ATen ops (nn.MultiheadAttention becomes 28 of them); they all
                # belong to one box, and the group's output is whichever op runs last.
                merged = by_module[path]
                owner[fx_node] = merged.id
                merged.output_shape = _shape(fx_node)
                continue
            else:
                kind = type(module).__name__
                extra = module.extra_repr()
                scope, scope_class = _scope(ancestors)  # a layer reports the container holding it, not itself
                node = Node(
                    id=unique_id(path.replace(".", "_"), used_ids),
                    op=kind.lower(),
                    label=kind,
                    params={"config": extra} if extra else {},
                    output_shape=_shape(fx_node),
                    scope=scope,
                    scope_class=scope_class,
                )
                by_module[path] = node
                paths[node.id] = path
        owner[fx_node] = node.id
        graph.nodes.append(node)

    qualify_labels(graph.nodes, paths)
    graph.edges = build_edges(owner)
    append_outputs(graph, owner, results, _shape)
    return graph


def _results(graph_module: torch.fx.GraphModule, fx_node: torch.fx.Node) -> list[tuple[str | None, torch.fx.Node]]:
    """Read what an exported graph returns as ``(key, producing node)`` pairs.

    Export flattens the return value, so the graph's ``output`` node always holds a flat tuple and says nothing about
    whether the model returned one tensor, three, or a dict. That shape lives in the module's output pytree spec, which
    is what tells a plain ``return y`` apart from a ``return (y,)`` — and so keeps the two frontends drawing the same
    boxes for the same model.

    Args:
        graph_module: Result of ``ExportedProgram.module()``, carrying the output pytree spec.
        fx_node: The graph's ``output`` node.

    Returns:
        One pair per returned tensor, in return order.
    """
    values = [
        value for value in _flatten(fx_node.args[0] if fx_node.args else None) if isinstance(value, torch.fx.Node)
    ]
    spec = getattr(graph_module, "_out_spec", None)
    structure, names = _out_structure(spec)
    keys = result_keys(len(values), structure, names)
    return list(zip(keys, values, strict=True))


def _flatten(result: object) -> list[object]:
    """The entries of an export ``output`` node's single argument, which is always a flat tuple."""
    return list(result) if isinstance(result, (tuple, list)) else [result]


def _out_structure(spec: object) -> tuple[str | None, list[str] | None]:
    """Classify an output pytree spec as a dict return, a tuple return, or a single unwrapped tensor.

    Args:
        spec: The module's ``_out_spec``, or ``None`` when torch didn't record one.

    Returns:
        ``("dict", keys)``, ``("tuple", None)``, or ``(None, None)`` for a bare tensor.
    """
    kind = getattr(spec, "type", None)
    if kind is None:  # a leaf spec, i.e. the model returns the tensor itself
        return None, None
    if kind is dict:
        context = getattr(spec, "context", None)
        return "dict", [str(key) for key in context] if isinstance(context, list) else None
    if kind in (tuple, list):
        return "tuple", None
    return None, None


def _continues_subscript(fx_node: torch.fx.Node, label: str, indexing: dict[torch.fx.Node, Node]) -> Node | None:
    """The box ``fx_node`` belongs to when it is another step of a subscript already being drawn.

    ``x[:, 0, :-1]`` is one expression to a reader and one node to fx, but export lowers it per axis — a ``select`` and
    then a ``slice``. Consecutive indexing ops that came from the same source line and feed nothing but each other are
    that one expression, so they fold back into a single box. Two subscripts written on separate lines keep a box each,
    which is what fx draws for them too.

    Args:
        fx_node: The node being converted.
        label: Its normalized label.
        indexing: Indexing nodes seen so far, mapped to the box each is drawn as.

    Returns:
        The box to extend, or ``None`` when this node starts one of its own.
    """
    if label != "index" or len(fx_node.all_input_nodes) != 1:
        return None
    source = fx_node.all_input_nodes[0]
    if source not in indexing or len(source.users) != 1:
        return None
    line = fx_node.meta.get("stack_trace")
    return indexing[source] if line is not None and line == source.meta.get("stack_trace") else None


def _leaf_module(model: nn.Module, entries: _Entries) -> tuple[str, nn.Module | None, _Entries]:
    """Resolve the leaf ``nn.Module`` a node was lowered from, if it was lowered from one.

    The stack is walked **outermost-first**, which is the direction fx applies the same predicate while tracing: the
    first leaf on the path stops the descent and nothing inside it is ever visited. Asking about the innermost entry
    instead makes every torch.nn descendant of a torch.nn composite qualify on its own, so an
    ``nn.TransformerEncoderLayer`` gets drawn as one box *and* as its nine children, and the two frontends disagree
    about a model neither had trouble tracing.

    When the node came from functional code instead, the returned path is still the innermost recorded one, which is
    what scope resolution needs.

    Args:
        model: The original module, whose submodule paths ``nn_module_stack`` refers to.
        entries: The node's ``nn_module_stack`` entries, outermost first.

    Returns:
        A ``(path, module, ancestors)`` triple; ``module`` is ``None`` when the node came from functional code rather:
            than a leaf layer, and ``ancestors`` is the entries above the resolved one, for scope resolution.
    """
    for index, (path, _) in enumerate(entries):
        try:
            module = model.get_submodule(path)
        except AttributeError:  # a path export synthesized that the original model doesn't have
            continue
        if LEAF_PROBE.is_leaf_module(module, path):
            return path, module, entries[:index]
    return (entries[-1][0] if entries else ""), None, entries


def _scope(entries: _Entries) -> tuple[str | None, str | None]:
    """Resolve the immediate custom-container ancestor from ``nn_module_stack`` entries, as the fx frontend does.

    Args:
        entries: The node's ``nn_module_stack`` entries, with the node's own leaf module already sliced off if it has
            one, so that a layer reports the container holding it rather than itself.

    Returns:
        The container's dotted path and class name, or ``(None, None)`` when the node sits at the model root.
    """
    if not entries:
        return None, None
    path, cls = entries[-1]
    # export records the class as a fully-qualified string, unlike fx, which records the class object itself.
    return path, cls.rsplit(".", 1)[-1] if isinstance(cls, str) else getattr(cls, "__name__", str(cls))


def _stack(fx_node: torch.fx.Node) -> _Entries:
    """Read ``nn_module_stack`` as ``(path, class)`` entries, dropping the root module fx never records."""
    stack = fx_node.meta.get("nn_module_stack")
    return [entry for entry in stack.values() if entry[0]] if stack else []


def _label(target: object) -> str:
    """Name a functional node the way the fx frontend would, e.g. ``aten.relu.default`` becomes ``"relu"``."""
    name = getattr(target, "__name__", None) or str(target)
    return normalize_label(name.split(".")[0])  # ATen overloads report as "relu.default" / "flatten.using_ints"


def _plumbing(graph_module: torch.fx.GraphModule) -> set[torch.fx.Node]:
    """Collect the ``call_function`` nodes that exist to serve a guard rather than to compute the model's output.

    Two kinds qualify. The guard nodes themselves produce no tensor data — ``aten.item``, ``sym_ite``, ``operator.ge``,
    ``aten._assert_scalar``. Above them sits the branch condition the model really did compute (``x.sum() > 0`` lowers
    to ``sum`` then ``gt`` then ``ne``), which is tensor-valued but feeds nothing except those guard nodes, and would
    otherwise trail off the diagram as a dead stub.

    A node with no users at all is kept: that is dead code the model actually contains, and the fx frontend draws it
    too, so dropping it here would make the two frontends disagree.

    Args:
        graph_module: Result of ``ExportedProgram.module()``.

    Returns:
        The nodes to leave out of the diagram.
    """
    plumbing: set[torch.fx.Node] = set()
    # Reverse order visits every consumer before its producer, so one pass reaches the top of each guard chain.
    for fx_node in reversed(list(graph_module.graph.nodes)):
        if fx_node.op != "call_function":
            continue
        is_guard = not _is_tensor_valued(fx_node.meta.get("val"))
        serves_only_guards = bool(fx_node.users) and all(user in plumbing for user in fx_node.users)
        if is_guard or serves_only_guards:
            plumbing.add(fx_node)
    return plumbing


def _is_tensor_valued(val: object) -> bool:
    """Whether a node's ``val`` metadata carries tensor data, which is what separates real ops from guard plumbing.

    ``draft_export`` emits the branch condition it specialized on as ``aten.item`` / ``sym_ite`` / ``operator.ge`` /
    ``aten._assert_scalar`` nodes, none of which produce a tensor. Multi-output ops (``torch.max(dim=...)``, ``split``,
    ``topk``) report a tuple of tensors and are real nodes, so they have to survive the same check.

    Args:
        val: The node's ``meta["val"]`` entry.

    Returns:
        Whether the node produces tensor data.
    """
    if isinstance(val, torch.Tensor):
        return True
    return isinstance(val, (tuple, list)) and any(isinstance(item, torch.Tensor) for item in val)


def _shape(fx_node: torch.fx.Node) -> tuple[int, ...] | None:
    """Read the node's output shape, which export records for free — no separate shape-propagation pass."""
    val = fx_node.meta.get("val")
    if isinstance(val, (tuple, list)) and len(val) == 1:
        val = val[0]  # a multi-output op that produced exactly one tensor, e.g. split into a single chunk
    if not isinstance(val, torch.Tensor):
        return None
    try:
        return tuple(int(dim) for dim in val.shape)
    except TypeError:  # a symbolic dimension, which has no single integer value
        return None
