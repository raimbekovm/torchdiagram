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

# fx's own leaf-module rule ("a torch.nn built-in that isn't a Sequential"), reused verbatim so that both frontends
# draw the same box for the same layer instead of each inventing its own notion of a leaf.
LEAF_PROBE = torch.fx.Tracer()

# Names a tensor subscript can arrive under. A reader writes one thing, `x[:, 0]`, but each tracer lowers it its own
# way: fx keeps the `operator.getitem` the syntax desugars to, while export lowers it to whichever ATen overload fits
# the subscript — `slice` for a range, `select` for an index, `index` for a tensor of indices. Left alone, the same
# model draws a box reading `getitem` under one frontend and `slice` under the other.
_INDEXING = frozenset({"getitem", "slice", "select", "index"})

# Names a dtype cast can arrive under, for the same reason. `x.float()` is a method of its own to fx and one more
# overload of `to` to export, which lowers every cast the same way, so a model that casts anything draws a box reading
# `float` on one side and `to` on the other.
_CASTS = frozenset(
    {"bfloat16", "bool", "byte", "char", "double", "float", "half", "int", "long", "short", "type", "type_as"}
)


def normalize_label(name: str) -> str:
    """Map a functional op's name to the one both frontends agree on, e.g. every form of subscript to ``index``."""
    if name in _INDEXING:
        return "index"
    return "to" if name in _CASTS else name


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
    """Ids of the kept nodes feeding ``fx_node``, looking through dropped ones only when nothing else feeds it.

    Looking through unconditionally reinstates the very dependency the plumbing pass exists to remove. ViT's
    ``cls_token.expand(x.shape[0], -1, -1)`` takes its data from the class token and only its batch size from ``x``;
    walking through the dropped shape lookup draws an arrow from the patch features into the class token, which says the
    token is computed from them. When a node has a kept input of its own, that input is the answer.
    """
    direct = [owner[upstream] for upstream in fx_node.all_input_nodes if upstream in owner]
    if direct:
        return direct
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


def continues_subscript(fx_node: torch.fx.Node, label: str, indexing: dict[torch.fx.Node, Node]) -> Node | None:
    """The box ``fx_node`` belongs to when it is another step of a subscript already being drawn.

    Both tracers split a subscript, and they split it in different places. ``x[:, 0, :-1]`` is one node to fx and two to
    export, which lowers a multi-axis subscript per axis; ``x[0][1]`` is one node to export, which cannot see the two
    subscripts, and two to fx. Neither split is recoverable from the other graph — the ATen ops behind ``x[0, 1]`` and
    ``x[0][1]`` are identical — so the two are made to agree by collapsing both: a run of indexing ops that came from
    one source line and feed nothing but each other draws as one box, whichever tracer produced it.

    Subscripts on separate lines keep a box each on both sides, which is what a reader wrote.

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


def append_outputs(
    graph: Graph,
    owner: dict[torch.fx.Node, str],
    results: Sequence[tuple[str | None, torch.fx.Node]],
    shape: Callable[[torch.fx.Node], tuple[int, ...] | None],
    used_ids: set[str],
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
        used_ids: Ids already handed out. An output box is named after the return it carries, and a model is free to
            hold a submodule called ``output``, so the two have to be kept apart.
    """
    for key, producer in results:
        node_id, label = _output_identity(key)
        node_id = unique_id(node_id, used_ids)
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


def parameter_node(model: torch.nn.Module, target: str, shape: tuple[int, ...] | None) -> Node:
    """Build the box for a learned tensor the model uses directly in ``forward()``.

    A ``Conv2d``'s weight belongs inside its layer's box and never surfaces. A class token, a position embedding, or a
    causal mask referenced straight from ``forward()`` has nowhere else to live: dropping it leaves the addition it
    feeds looking unary, and leaves the reader to guess where a tensor with no incoming edge came from.

    Args:
        model: The module the attribute belongs to.
        target: Dotted attribute path, e.g. ``"cls_token"`` or ``"blocks.0.mask"``.
        shape: The tensor's shape, when known.

    Returns:
        The IR node, named after the attribute and marked ``parameter`` or ``buffer`` by what it actually is.
    """
    attribute: object = model
    for part in target.split("."):
        attribute = getattr(attribute, part, None)
    op = "parameter" if isinstance(attribute, torch.nn.Parameter) else "buffer"
    return Node(id=target.replace(".", "_"), op=op, label=target.rpartition(".")[2], output_shape=shape)


def class_name(cls: object) -> str:
    """Name the class recorded on an ``nn_module_stack`` entry, which fx stores as an object and export as a string."""
    return cls.rsplit(".", 1)[-1] if isinstance(cls, str) else getattr(cls, "__name__", str(cls))


def record_scopes(entries: Sequence[tuple[str, object]], scopes: dict[str, str]) -> None:
    """Note every container on a node's ``nn_module_stack``, not just the one it reports as its scope.

    A level that owns no operation of its own — a bare ``nn.Sequential`` between two custom modules — is never any
    node's ``scope``, so a map built from the nodes alone has a hole exactly there, and aggregation dead-ends at it. The
    stack knows the whole chain, so recording all of it fills the hole in.

    Args:
        entries: The node's ``nn_module_stack`` entries as ``(path, class)`` pairs.
        scopes: Accumulating path-to-class-name map. Modified in place.
    """
    for path, cls in entries:
        if path:
            scopes[path] = class_name(cls)


def unique_id(candidate: str, used: set[str]) -> str:
    """Return ``candidate``, suffixed if needed, so node ids stay unique; records it in ``used``."""
    node_id = candidate
    suffix = 1
    while node_id in used:
        node_id = f"{candidate}_{suffix}"
        suffix += 1
    used.add(node_id)
    return node_id


def leaf_root_graph(model: torch.nn.Module, args: tuple[torch.Tensor, ...] | None, *, name: str) -> Graph:
    """Draw a model that is itself a single leaf layer as the one box it is.

    Neither tracer treats the root module as a layer: fx never applies its leaf rule to the model it was handed, so it
    descends into ``Linear.forward`` and reports ``linear``, an op name with no configuration; export records nothing on
    ``nn_module_stack`` above the root and lands in the same place. Pointing the tool at ``nn.LSTM`` or
    ``nn.TransformerEncoderLayer`` to see what one looks like then gives a page of functional ops with no layer name on
    any of them, while the same layer nested one level down draws a proper box.

    Built directly rather than traced, so both frontends give the same answer for free.

    Args:
        model: The leaf layer being diagrammed.
        args: One example tensor per ``forward()`` argument, or ``None`` to skip shape annotation.
        name: Diagram title.

    Returns:
        A graph of one input node per argument, the layer, and one output node per returned tensor.
    """
    kind = type(model).__name__
    extra = model.extra_repr()
    layer = Node(id=kind.lower(), op=kind.lower(), label=kind, params={"config": extra} if extra else {})
    graph = Graph(name=name)

    for index, tensor in enumerate(args or (None,)):
        node_id = "input" if index == 0 else f"input_{index}"
        shape = None if tensor is None else tuple(tensor.shape)
        graph.nodes.append(Node(id=node_id, op="input", label="input", output_shape=shape))
        graph.edges.append(Edge(source=node_id, target=layer.id))
    graph.nodes.append(layer)

    results = _leaf_root_results(model, args)
    layer.output_shape = results[0][1] if len(results) == 1 else None
    used_ids = {node.id for node in graph.nodes} | {layer.id}
    for key, shape in results:
        node_id, label = _output_identity(key)
        node_id = unique_id(node_id, used_ids)
        graph.nodes.append(Node(id=node_id, op="output", label=label, output_shape=shape))
        graph.edges.append(Edge(source=layer.id, target=node_id))
    return graph


def _leaf_root_results(
    model: torch.nn.Module, args: tuple[torch.Tensor, ...] | None
) -> list[tuple[str | None, tuple[int, ...] | None]]:
    """Run a leaf layer once to learn what it returns, as ``(key, shape)`` pairs."""
    if args is None:
        return [(None, None)]
    with torch.no_grad():
        result = model(*args)
    if isinstance(result, dict):
        structure, names, values = "dict", list(result), list(result.values())
    elif isinstance(result, (tuple, list)):
        structure, names, values = "tuple", None, list(result)
    else:
        structure, names, values = None, None, [result]
    keys = result_keys(len(values), structure, names)
    shapes = [tuple(value.shape) if isinstance(value, torch.Tensor) else None for value in values]
    return list(zip(keys, shapes, strict=True))


def qualify_labels(nodes: list[Node], paths: dict[str, str]) -> None:
    """Disambiguate layers that would otherwise draw identical boxes, using the attribute each was traced from.

    A transformer's token and position embeddings are both an ``nn.Embedding`` at the same level of the model, so the
    class name alone labels two boxes the same and says nothing about which is which. Where several nodes in one scope
    share a label, each is qualified with its attribute name — ``Embedding (tok_emb)`` — and layers that are alone under
    their label keep the plain class name, which is what most of a diagram is.

    Several boxes can also be one attribute rather than several: a siamese encoder applies one layer to two inputs, and
    tied embeddings run one weight matrix twice. The data flow really does pass through twice, so both boxes are drawn,
    but they are numbered ``Linear (a, call 1)`` / ``(a, call 2)`` and cross-referenced in ``params["shared_with"]`` so
    the diagram does not claim two sets of weights where the model has one.

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
        if len(group) < 2:
            continue  # a layer alone under its label needs nothing to tell it apart from
        by_attribute: dict[str, list[Node]] = {}
        for node in group:
            by_attribute.setdefault(scope_leaf(paths[node.id]), []).append(node)
        for name, calls in by_attribute.items():
            if len(calls) == 1:
                calls[0].label = f"{calls[0].label} ({name})"
                continue
            for index, node in enumerate(calls, start=1):
                node.label = f"{node.label} ({name}, call {index})"
                node.params = {**node.params, "shared_with": [call.id for call in calls if call is not node]}
