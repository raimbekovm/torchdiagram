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

| Field          | Type                      | Default | Description                                                                                  |
| -------------- | ------------------------- | ------- | -------------------------------------------------------------------------------------------- |
| `id`           | `str`                     | —       | Unique identifier, referenced by edges.                                                      |
| `op`           | `str`                     | —       | Normalized operation kind, e.g. `"conv2d"`, `"add"`, `"input"`, `"output"`.                  |
| `label`        | `str`                     | —       | Human-readable text shown on the diagram.                                                    |
| `params`       | `dict[str, Any]`          | `{}`    | Layer configuration; the tracer stores the module's `extra_repr()` under the `"config"` key. |
| `output_shape` | `tuple[int, ...] \| None` | `None`  | Output tensor shape, populated when `trace()` receives an `example_input`.                   |

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
