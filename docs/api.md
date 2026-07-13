# API reference

All public names are importable from the top-level package:

```python
import torchdiagram as td

graph = td.trace(model)
td.render(graph, "model.svg")
```

## Functions

### `td.trace(model, example_input=None, *, name=None) -> Graph`

Trace a PyTorch module into a renderable graph.

| Parameter       | Type                   | Description                                                                             |
| --------------- | ---------------------- | --------------------------------------------------------------------------------------- |
| `model`         | `nn.Module`            | Module to trace. `forward()` may be arbitrary fx-traceable code.                        |
| `example_input` | `torch.Tensor \| None` | When given, a forward pass is shape-propagated and every node carries its output shape. |
| `name`          | `str \| None`          | Diagram title. Defaults to the model's class name.                                      |

Raises `torch.fx.proxy.TraceError` (and related fx exceptions) when the model is not symbolically traceable; see [tracing limitations](usage.md#tracing-limitations).

### `td.render(graph, path) -> Path`

Validate `graph` and write it to `path`, selecting the format from the file extension.

| Extension       | Format                         |
| --------------- | ------------------------------ |
| `.svg`          | SVG image                      |
| `.tex`, `.tikz` | Standalone TikZ/LaTeX document |

Returns the written path. Raises `ValueError` for an unsupported extension or an invalid graph (see `Graph.validate`).

### `td.aggregate_blocks(graph, *, min_repeats=2) -> Graph`

Collapse scoped blocks — repeated ones (e.g. ResNet layers) and singleton ones (e.g. a transformer's attention or MLP sub-block) — into single labeled nodes, recursively from the deepest nesting outward.

| Parameter     | Type    | Description                                                                                         |
| ------------- | ------- | --------------------------------------------------------------------------------------------------- |
| `graph`       | `Graph` | Graph to transform. Not mutated.                                                                    |
| `min_repeats` | `int`   | Minimum run length, in groups, required to merge multiple groups into one badged node. Default `2`. |

Groups nodes by the module scope they were traced from (see `Node.scope`), processing the deepest scope nesting first. A group that's single-entry/single-exit always collapses into one synthetic node with `op="block"` and `params={"repeats": ..., "block_class": ..., "ops_per_repeat": ...}` — even if it occurs only once, e.g. an attention or MLP block that appears exactly once per transformer layer, in which case `label` is just the class name (`"Attention"`) with `repeats=1`. Consecutive groups that additionally share a `scope_class` and an identical internal structure are merged into a single node instead, labeled e.g. `"BasicBlock ×5"`, provided the run has at least `min_repeats` groups; shorter runs still collapse, just one node per group. Groups that aren't single-entry/single-exit are left untouched — a branch whose output is tapped elsewhere stays expanded rather than being merged incorrectly. Because collapsing runs deepest-first, a repeated outer block (e.g. a transformer block containing a singleton attention and MLP) is compared against its siblings using its own short, already-collapsed node sequence, which is what lets it merge into `"TransformerBlock ×N"` even though its inner sub-blocks are not themselves repeated. Pure function; returns a new, validated graph. See [aggregating repeated blocks](usage.md#aggregating-repeated-blocks).

### `td.to_svg(graph) -> str`

Render `graph` as a self-contained SVG document string. Pure function; does not validate the graph or touch the filesystem.

### `td.to_tikz(graph) -> str`

Render `graph` as a standalone LaTeX/TikZ document string. Labels are escaped for LaTeX. Pure function; does not validate the graph or touch the filesystem.

## Data classes

The intermediate representation lives in `torchdiagram.graph` and has no torch dependency.

### `td.Graph`

An ordered model graph. Node order in `nodes` is the topological (execution) order; renderers lay nodes out in this order.

| Field   | Type         | Default   | Description               |
| ------- | ------------ | --------- | ------------------------- |
| `name`  | `str`        | `"model"` | Diagram title.            |
| `nodes` | `list[Node]` | `[]`      | Nodes in execution order. |
| `edges` | `list[Edge]` | `[]`      | Directed data-flow edges. |

**`Graph.validate() -> None`** — raises `ValueError` on duplicate node ids or edges that reference unknown node ids. Called automatically by `render()`.

### `td.Node`

A single block in the diagram: a layer, a function call, or a graph input/output.

| Field          | Type                      | Default | Description                                                                                        |
| -------------- | ------------------------- | ------- | -------------------------------------------------------------------------------------------------- |
| `id`           | `str`                     | —       | Unique identifier, referenced by edges.                                                            |
| `op`           | `str`                     | —       | Normalized operation kind, e.g. `"conv2d"`, `"add"`, `"input"`, `"output"`.                        |
| `label`        | `str`                     | —       | Human-readable text shown on the diagram.                                                          |
| `params`       | `dict[str, Any]`          | `{}`    | Layer configuration; the tracer stores the module's `extra_repr()` under the `"config"` key.       |
| `output_shape` | `tuple[int, ...] \| None` | `None`  | Output tensor shape, populated when `trace()` receives an `example_input`.                         |
| `scope`        | `str \| None`             | `None`  | Dotted path of the immediate custom-container module this node was traced from, e.g. `"layer1.0"`. |
| `scope_class`  | `str \| None`             | `None`  | Class name of that container, e.g. `"BasicBlock"`, or `None` alongside `scope`.                    |

`op` values produced by the tracer: `"input"` and `"output"` for graph boundaries, the lowercased class name for submodule calls (`"conv2d"`, `"linear"`, `"maxpool2d"`, ...), and the function or method name for functional ops (`"relu"`, `"add"`, `"flatten"`, `"view"`, ...).

**`Node.is_io -> bool`** — `True` for graph input/output nodes (`op` is `"input"` or `"output"`); both renderers use it to pick the distinct I/O box style.

### `td.Edge`

A directed data-flow edge between two nodes.

| Field    | Type  | Description            |
| -------- | ----- | ---------------------- |
| `source` | `str` | Id of the source node. |
| `target` | `str` | Id of the target node. |

## Constants

### `td.__version__`

The installed package version as a string.
