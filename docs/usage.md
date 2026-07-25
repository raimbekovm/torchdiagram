# User guide

## Tracing a model

`torchdiagram.trace()` accepts any `nn.Module` whose `forward()` can be handled by `torch.fx.symbolic_trace`, and falls back to a `torch.export` frontend for the rest:

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

### Data-dependent control flow

`torch.fx` symbolic tracing executes `forward()` with proxy values, so it cannot evaluate a branch that depends on tensor values (`if x.sum() > 0:`, loops whose length depends on the data). When that happens, `trace()` falls back to a second frontend built on `torch.export`, which runs the model on the example input and records the branch that input actually takes:

```python
graph = td.trace(model, torch.randn(1, 3, 224, 224))
# UserWarning: ... specialized on the example input: branches not taken by this input
# are absent from the diagram
```

The warning matters: the resulting diagram is a **specialization**, not a complete picture of the model. Branches the example input does not take are missing, and a different input can produce a different diagram. Choose an example input that exercises the path you want to publish.

The fallback needs an example input, since `torch.export` traces by running the model. Calling `td.trace(model)` on an untraceable model with no example input raises `TraceError` with a hint to supply one.

Both frontends emit the same IR, so shapes, `scope`, `aggregate_blocks()`, and every renderer behave identically either way. On a model both can trace, they produce identical graphs. Pin one with `backend=`:

```python
graph = td.trace(model, example_input, backend="fx")  # never fall back
graph = td.trace(model, example_input, backend="export")  # skip fx entirely
```

### Other tracing limitations

- **Non-tensor containers with dynamic contents** and some dynamic Python features inside `forward()` may not be traceable by either frontend.
- A model the `torch.export` frontend also cannot handle surfaces the original `TraceError` from fx, since that error describes the model rather than the fallback.

Models that are fully defined in terms of submodule calls, tensor functions, and tensor methods — which covers most convolutional and transformer architectures — trace without modification.

### Aggregating repeated blocks

Deep models trace to hundreds of nodes — a ResNet-50 is 177, and every transformer block adds dozens more for
its attention math alone. `aggregate_blocks()` collapses scoped blocks into single labeled nodes before
rendering, recursing from the deepest nesting outward:

```python
graph = td.trace(model, example_input)
graph = td.aggregate_blocks(graph)
td.render(graph, "model.svg")
```

It groups nodes by the submodule they were traced from (`Node.scope`/`scope_class`, set by `trace()`). A
group that's structurally clean (single entry, single exit) always collapses to one node — `op="block"`,
`params={"repeats": ..., "block_class": ..., "ops_per_repeat": ...}` — even if it only occurs once, so a
transformer's attention or MLP sub-block collapses to a plain `"Attention"`/`"MLP"` node the same way a
repeated ResNet block does. Consecutive groups that additionally share a class and internal structure merge
into a single node instead, labeled e.g. `"BasicBlock ×5"`. Because the deepest scopes collapse first, a
transformer block's own node sequence becomes short and uniform once its attention/MLP contents are
collapsed, which is what lets the stack of blocks itself then merge into one `"TransformerBlock ×N"` node —
no attention-specific code involved, just the same scope+structure rule applied one nesting level at a time.
This outer merge relies on the block having at least one op of its own at that scope (a residual `add` is the
common case); a container that does nothing but call its children collapses its contents fine but won't merge
across repeats itself — see [design.md](design.md#block-aggregation).

Only exact structural matches merge. A stage whose first block differs from the rest — for example a ResNet
stage's first `Bottleneck`, which has an extra downsample convolution on the shortcut — collapses to its own
single node rather than merging with its neighbors; the remaining uniform blocks merge into one. This is
expected, not a bug: it keeps the diagram from misrepresenting a block that isn't actually identical to its
neighbors, while still compacting it to one glyph.

`min_repeats` (default `2`) sets how many consecutive matching blocks are required before they merge into one
badged node; below that, each still collapses individually, just without the `×N` badge:

```python
graph = td.aggregate_blocks(graph, min_repeats=3)  # only merge runs of 3 or more into one node
```

`aggregate_blocks()` is a pure function — it returns a new graph and does not modify its input — and is opt-in: `trace()` and `render()` never call it implicitly.

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

### Theming

`render()`, `to_svg()`, and `to_tikz()` all take a `theme` keyword — a `Theme` value carrying the diagram's colors and typography. Three presets ship with the package:

```python
td.render(graph, "model.svg")  # td.DEFAULT — soft blue, the implicit default
td.render(graph, "model.svg", theme=td.MONOCHROME)  # grayscale, for print figures
td.render(graph, "model.svg", theme=td.DARK)  # for dark backgrounds
```

Build your own by copying a preset and overriding fields with `dataclasses.replace`:

```python
from dataclasses import replace

theme = replace(td.DEFAULT, block_fill="#fff3cd", block_stroke="#e0a800", edge_color="#555555")
td.render(graph, "model.svg", theme=theme)
```

All colors are hex strings; see [the `Theme` field table](api.md#tdtheme) for the full list. A `Theme` is frozen, so the shared presets can't be mutated in place — always build a new one with `replace`.

The same theme drives both renderers, so an SVG preview and its TikZ counterpart match. Two caveats follow from the backends differing:

- `font_family` applies to SVG only. TikZ uses the LaTeX document's font; set it in your `.tex` preamble instead.
- `corner_radius` and `stroke_width` are interpreted in each backend's native unit (pixels for SVG, points for TikZ), so they're a close visual match rather than an exact one.

Geometry — node spacing and box size — is not themeable yet; SVG measures in pixels and TikZ in millimeters, so a single shared value would have no honest unit.

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
