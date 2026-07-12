# User guide

## Tracing a model

`torchdiagram.trace()` accepts any `nn.Module` whose `forward()` can be handled by `torch.fx.symbolic_trace`:

```python
import torchdiagram as td

graph = td.trace(model)
```

The trace records the actual data flow of `forward()`, so the diagram is derived from the code rather than from a separate, manually maintained description. The following constructs are captured:

- **Submodule calls** (`self.conv(x)`) become nodes labeled with the layer class name (`Conv2d`, `Linear`, ...). The layer configuration reported by `extra_repr()` — kernel size, feature counts, and so on — is stored in `node.params["config"]`.
- **Function calls** (`torch.relu(x)`, `x + y`, `torch.flatten(x, 1)`) become nodes named after the function (`relu`, `add`, `flatten`).
- **Method calls** (`x.view(...)`, `x.mean(...)`) become nodes named after the method.
- **Model inputs and outputs** become dedicated `input` and `output` nodes.
- **Residual connections and parallel branches** appear as additional edges; a node may have any number of incoming and outgoing edges.

Parameter and buffer accesses (`get_attr` in fx terms) are internal plumbing and are excluded from the diagram.

### Shape annotations

Passing an example input runs shape propagation over the traced graph and stores the output shape of every node:

```python
graph = td.trace(model, torch.randn(1, 3, 224, 224))
```

Renderers display these shapes as a second line inside each block (for example `1×64×56×56`). Without an example input, nodes carry no shape information and the diagram shows structure only.

The example input must match what `forward()` expects. For models whose input is not a float tensor (for example, embedding indices), construct an appropriate tensor such as `torch.zeros(1, 128, dtype=torch.long)`.

### Diagram title

The graph name defaults to the model's class name and can be overridden:

```python
graph = td.trace(model, name="Encoder v2")
```

### Tracing limitations

`torch.fx` symbolic tracing executes `forward()` with proxy values, which imposes the standard fx restrictions:

- **Data-dependent control flow** (`if x.sum() > 0:`, loops whose length depends on tensor values) raises a `TraceError`. A fallback frontend based on `torch.export` is planned; see [design.md](design.md).
- **Non-tensor containers with dynamic contents** and some dynamic Python features inside `forward()` may not be traceable.

Models that are fully defined in terms of submodule calls, tensor functions, and tensor methods — which covers most convolutional and transformer architectures — trace without modification.

## Rendering

### Choosing a format

`render()` writes a graph to a file and selects the format from the file extension:

```python
td.render(graph, "model.svg")  # SVG
td.render(graph, "model.tex")  # TikZ (standalone LaTeX document)
td.render(graph, "model.tikz")  # TikZ, same content as .tex
```

An unsupported extension raises `ValueError`. The graph is validated before writing; a graph with duplicate node ids or edges that reference missing nodes is rejected.

To obtain the output as a string instead of a file, call the format functions directly:

```python
svg_text = td.to_svg(graph)
tikz_text = td.to_tikz(graph)
```

### SVG output

The SVG renderer produces a self-contained document with no external references. Nodes are laid out in a single vertical column in execution order; adjacent nodes are connected with straight arrows, and skip connections are routed as curves to the right of the column. Input and output nodes are styled distinctly from computation blocks.

The file can be opened in any browser, embedded in HTML or Markdown, and edited in vector graphics software such as Inkscape.

### TikZ output

The TikZ renderer produces a standalone LaTeX document:

- Compile it directly: `pdflatex model.tex` (or `tectonic model.tex`).
- Or copy the `tikzpicture` environment into an existing document. The required libraries are loaded with `\usetikzlibrary{arrows.meta,positioning}`.

Node styles are defined once at the top of the picture (`block`, `io`, `arrow`), so the appearance of the whole diagram can be adjusted by editing those three style definitions. Labels are escaped for LaTeX; characters such as `_`, `&`, and `%` in layer names render literally.

## Working with the graph IR

`trace()` returns a `Graph` — a plain dataclass containing `Node` and `Edge` lists with no torch dependency. It can be inspected and modified before rendering:

```python
graph = td.trace(model)

# Rename a node's displayed label
for node in graph.nodes:
    if node.op == "conv2d":
        node.label = f"Conv {node.params['config'].split(',')[1].strip()}"

# Drop nodes you do not want in the figure (e.g. dropout at inference time)
dropped = {n.id for n in graph.nodes if n.op == "dropout"}
graph.nodes = [n for n in graph.nodes if n.id not in dropped]
graph.edges = [e for e in graph.edges if e.source not in dropped and e.target not in dropped]
```

Note that removing a node also removes its edges; reconnect the neighbors explicitly if the data flow should remain visually continuous.

Graphs can also be constructed entirely by hand, which is useful for schematic figures that do not correspond to runnable code:

```python
graph = td.Graph(
    name="pipeline",
    nodes=[
        td.Node(id="in", op="input", label="tokens"),
        td.Node(id="enc", op="encoder", label="Encoder"),
        td.Node(id="out", op="output", label="logits"),
    ],
    edges=[td.Edge("in", "enc"), td.Edge("enc", "out")],
)
td.render(graph, "pipeline.svg")
```

`Graph.nodes` order is the layout order: renderers draw nodes top-to-bottom in list order, so hand-built graphs should list nodes in execution order.
