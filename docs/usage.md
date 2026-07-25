# User guide

## Tracing a model

`torchdiagram.trace()` accepts any `nn.Module` whose `forward()` can be handled by `torch.fx.symbolic_trace`, and falls back to a `torch.export` frontend for the rest:

```python
import torchdiagram as td

graph = td.trace(model)
```

The trace records the actual data flow of `forward()`, so the diagram is derived from the code rather than from a separate, manually maintained description. The following constructs are captured:

- **Submodule calls** (`self.conv(x)`) become nodes labeled with the layer class name (`Conv2d`, `Linear`, ...). The layer configuration reported by `extra_repr()` — kernel size, feature counts, and so on — is stored in `node.params["config"]`. Where two layers of the same class sit at the same level of the model, the class name alone would label both boxes identically, so each is qualified with the attribute it was traced from: a transformer's token and position embeddings draw as `Embedding (tok_emb)` and `Embedding (pos_emb)`. A layer that is alone under its label keeps the plain class name.
- **Function calls** (`torch.relu(x)`, `x + y`, `torch.flatten(x, 1)`) become nodes named after the function (`relu`, `add`, `flatten`).
- **Method calls** (`x.view(...)`, `x.mean(...)`) become nodes named after the method.
- **Model inputs and outputs** become dedicated `input` and `output` nodes.
- **Residual connections and parallel branches** appear as additional edges; a node may have any number of incoming and outgoing edges.

Two kinds of node are internal plumbing and are excluded from the diagram: parameter and buffer accesses (`get_attr` in fx terms), and code that computes with a tensor's metadata rather than with the tensor. The second kind covers `x.shape[1]`, `x.size(0)`, and the arithmetic built on them — an attention head's `c // self.heads` is three nodes of it — none of which is an architecture step. What consumes them stays: `torch.arange(x.shape[1])` produces a real tensor, so the range is drawn, connected to the input the shape was read from.

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

Only exact matches merge, and the badge is a claim the transform has to be able to back. Two neighboring groups
merge into one `×N` node only when three things agree: their op sequence and internal wiring, the configuration
of every layer in them, and, for any block already collapsed inside them, everything that block collapsed. The
last two matter more than they sound. A VGG stage running 64 channels and the next one running 128 have the same
op sequence, so comparing structure alone would badge them `VGGBlock ×2` although they are different sizes; and
once each stage's inner `nn.Sequential` has collapsed to one node, a two-convolution stage and a
three-convolution one read as the same two-node sequence, so comparing only what is visible after collapsing
would merge those too. Both cases stay separate.

A stage whose first block differs from the rest — for example a ResNet stage's first `Bottleneck`, which has an
extra downsample convolution on the shortcut — likewise collapses to its own single node rather than merging
with its neighbors; the remaining uniform blocks merge into one. This is expected, not a bug: it keeps the
diagram from misrepresenting a block that isn't actually identical to its neighbors, while still compacting it
to one glyph.

Collapsed blocks are labeled with the class of the module they came from. `nn.Sequential`, `nn.ModuleList`, and
`nn.ModuleDict` are the exception: their class name describes a container rather than a computation, so a block
collapsed from one is labeled with the attribute holding it — a `self.classifier = nn.Sequential(...)` head draws
as `classifier`, not as `Sequential`.

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
td.render(graph, "model.png")  # PNG, needs the optional rasterizer
```

An unsupported extension raises `ValueError`. The graph is validated before writing; a graph with duplicate node ids or edges that reference missing nodes is rejected.

To obtain the output in memory instead of a file, call the format functions directly:

```python
svg_text = td.to_svg(graph)
tikz_text = td.to_tikz(graph)
png_bytes = td.to_png(graph)
```

### SVG output

The SVG renderer produces a self-contained document with no external references. Nodes are laid out in a single vertical column in execution order; adjacent nodes are connected with straight arrows, and skip connections are routed as curves to the right of the column. Input and output nodes are styled distinctly from computation blocks.

The file can be opened in any browser, embedded in HTML or Markdown, and edited in vector graphics software such as Inkscape.

### TikZ output

The TikZ renderer produces a standalone LaTeX document:

- Compile it directly: `pdflatex model.tex` (or `tectonic model.tex`).
- Or copy the `tikzpicture` environment into an existing document. The required libraries are loaded with `\usetikzlibrary{arrows.meta,positioning}`.

Node styles are defined once at the top of the picture (`block`, `io`, `arrow`), so the appearance of the whole diagram can be adjusted by editing those three style definitions. Labels are escaped for LaTeX; characters such as `_`, `&`, and `%` in layer names render literally.

### PNG output

PNG is the SVG rendering rasterized, for places that won't display vector graphics — a slide deck, an issue thread, a chat message. It needs one optional dependency, a prebuilt [resvg](https://github.com/linebender/resvg) wheel that pulls in no system libraries:

```bash
pip install 'torchdiagram[png]'
```

Without it, `to_png()` and `render(..., "model.png")` raise `ImportError` naming the extra; the vector formats are unaffected either way. On macOS with Python 3.10 there is no prebuilt wheel and pip falls back to building it from source, which needs a Rust toolchain; Python 3.11 and newer are covered.

`scale` multiplies the SVG's own pixel dimensions, so a 400×600 diagram becomes an 800×1200 image at the default `2.0`:

```python
td.render(graph, "model.png", scale=3.0)  # 3× resolution, e.g. for a slide
png_bytes = td.to_png(graph, scale=1.0)  # 1:1 with the SVG
```

Two properties are worth knowing before embedding the result. The background stays transparent, as in the SVG, so the diagram sits on whatever page it lands on — pair the `DARK` preset with a dark page and `DEFAULT` or `MONOCHROME` with a light one. And label text is typeset with the system fonts matching the theme's `font_family`; in a minimal container with no fonts installed the boxes and arrows still render but the text is silently dropped, so prefer SVG or TikZ there.

### Theming

`render()`, `to_svg()`, `to_tikz()`, and `to_png()` all take a `theme` keyword — a `Theme` value carrying the diagram's colors and typography. Three presets ship with the package:

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

The same theme drives every renderer, so an SVG preview, its PNG rasterization, and its TikZ counterpart match. Two caveats follow from the backends differing:

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
