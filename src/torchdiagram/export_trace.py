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

from .graph import Edge, Graph, Node

# fx's own leaf-module rule ("a torch.nn built-in that isn't a Sequential"), reused verbatim so that both frontends
# draw the same box for the same layer instead of each inventing its own notion of a leaf.
_LEAF_PROBE = torch.fx.Tracer()

# Export narrates a failed attempt loudly: draft_export logs a tlparse banner about unsound specialization, and
# torch.export._trace prints the partial graph straight to stderr with no logger in between. Both are noise here —
# a failed attempt is the expected path for the models this frontend exists to handle — so _quiet_export() suppresses
# them and trace_export() says the one thing that matters in a single warning.
_LOG_PREFIX = "torch"


def trace_export(model: nn.Module, example_input: torch.Tensor, *, name: str | None = None) -> Graph:
    """Trace ``model`` into a renderable graph using ``torch.export``.

    Args:
        model: Module to trace. Unlike the fx frontend, ``forward()`` may contain data-dependent control flow.
        example_input: Input to trace with. Required — ``torch.export`` needs concrete arguments — and used for
            output-shape annotation, which comes free with the export graph.
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
    graph_module, specialized = _export(model, example_input)
    if specialized:
        warnings.warn(
            f"{type(model).__name__} was traced with torch.export and specialized on the example input: branches "
            "not taken by this input are absent from the diagram",
            UserWarning,
            stacklevel=2,
        )
    return _build_graph(model, graph_module, name=name or type(model).__name__)


def _export(model: nn.Module, example_input: torch.Tensor) -> tuple[torch.fx.GraphModule, bool]:
    """Export ``model``, preferring a sound graph and falling back to a specialized one.

    Args:
        model: Module to export.
        example_input: Input to export with.

    Returns:
        The exported graph module, and whether it was specialized on ``example_input`` rather than soundly exported.

    Raises:
        Exception: The error from the sound attempt, if the specializing fallback is unavailable or also fails.
    """
    with _quiet_export():
        try:
            return torch.export.export(model, (example_input,), strict=False).module(), False
        except Exception as sound_error:
            draft_export = getattr(torch.export, "draft_export", None)
            if draft_export is None:  # torch too old to have the real-tensor path
                raise
            try:
                program = draft_export(model, (example_input,))
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
    plumbing = _plumbing(graph_module)

    for fx_node in graph_module.graph.nodes:
        if fx_node.op in ("placeholder", "output"):
            op = "input" if fx_node.op == "placeholder" else "output"
            node = Node(id=_unique(fx_node.name, used_ids), op=op, label=op, output_shape=_io_shape(fx_node))
        elif fx_node.op != "call_function" or fx_node in plumbing:
            continue  # get_attr parameter plumbing, the synthetic _guards_fn module, and symbolic guard assertions
        else:
            entries = _stack(fx_node)
            path, module = _leaf_module(model, entries)
            if module is None:
                label = _label(fx_node.target)
                scope, scope_class = _scope(entries)
                node = Node(
                    id=_unique(fx_node.name, used_ids),
                    op=label,
                    label=label,
                    output_shape=_shape(fx_node),
                    scope=scope,
                    scope_class=scope_class,
                )
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
                scope, scope_class = _scope(entries[:-1])  # a layer reports the container holding it, not itself
                node = Node(
                    id=_unique(path.replace(".", "_"), used_ids),
                    op=kind.lower(),
                    label=kind,
                    params={"config": extra} if extra else {},
                    output_shape=_shape(fx_node),
                    scope=scope,
                    scope_class=scope_class,
                )
                by_module[path] = node
        owner[fx_node] = node.id
        graph.nodes.append(node)

    seen_edges: set[tuple[str, str]] = set()
    for fx_node, target_id in owner.items():
        for upstream in fx_node.all_input_nodes:
            source_id = owner.get(upstream)
            # Merging a layer's ATen ops turns its internal edges into self-edges; drop those along with edges from
            # nodes that never made it into the IR.
            if source_id is None or source_id == target_id or (source_id, target_id) in seen_edges:
                continue
            seen_edges.add((source_id, target_id))
            graph.edges.append(Edge(source=source_id, target=target_id))
    return graph


def _leaf_module(model: nn.Module, entries: list[tuple[str, object]]) -> tuple[str, nn.Module | None]:
    """Resolve the leaf ``nn.Module`` a node was lowered from, if it was lowered from one.

    When the node came from functional code instead, the returned path is still the innermost recorded one, which is
    what scope resolution needs.

    Args:
        model: The original module, whose submodule paths ``nn_module_stack`` refers to.
        entries: The node's ``nn_module_stack`` entries.

    Returns:
        A ``(path, module)`` pair; ``module`` is ``None`` when the node came from functional code, not a leaf layer.
    """
    if not entries:
        return "", None
    path = entries[-1][0]
    try:
        module = model.get_submodule(path)
    except AttributeError:  # a path export synthesized that the original model doesn't have
        return path, None
    return path, module if _LEAF_PROBE.is_leaf_module(module, path) else None


def _scope(entries: list[tuple[str, object]]) -> tuple[str | None, str | None]:
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


def _stack(fx_node: torch.fx.Node) -> list[tuple[str, object]]:
    """Read ``nn_module_stack`` as ``(path, class)`` entries, dropping the root module fx never records."""
    stack = fx_node.meta.get("nn_module_stack")
    return [entry for entry in stack.values() if entry[0]] if stack else []


def _label(target: object) -> str:
    """Name a functional node the way the fx frontend would, e.g. ``aten.relu.default`` becomes ``"relu"``."""
    name = getattr(target, "__name__", None) or str(target)
    return name.split(".")[0]  # ATen overloads report as "relu.default" / "flatten.using_ints"


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


def _io_shape(fx_node: torch.fx.Node) -> tuple[int, ...] | None:
    """Read the shape of a graph input or output node.

    Export leaves the ``output`` node without ``val`` metadata of its own, so a single-return model's output shape is
    read off the node feeding it — matching what fx's shape propagation records there.

    Args:
        fx_node: The ``placeholder`` or ``output`` node.

    Returns:
        The shape, or ``None`` when the model takes or returns something other than one tensor.
    """
    shape = _shape(fx_node)
    if shape is not None or fx_node.op != "output":
        return shape
    inputs = fx_node.all_input_nodes
    return _shape(inputs[0]) if len(inputs) == 1 else None


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


def _unique(candidate: str, used: set[str]) -> str:
    """Return ``candidate``, suffixed if needed, so node ids stay unique across both naming schemes."""
    node_id = candidate
    suffix = 1
    while node_id in used:
        node_id = f"{candidate}_{suffix}"
        suffix += 1
    used.add(node_id)
    return node_id
