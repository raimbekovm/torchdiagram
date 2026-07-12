# Design

## Pipeline

```text
nn.Module ──▶ frontend (trace.py) ──▶ IR (graph.py) ──▶ transforms (planned) ──▶ renderers (svg, tikz)
```

Each stage only knows about the one after it, and everything downstream of the frontend is torch-free.

## Decisions

### Why `torch.fx` for tracing

The core promise of the tool is "diagram from arbitrary `forward()` code", and `torch.fx.symbolic_trace` is the lowest-friction way to get a faithful dataflow graph out of one: it captures residual connections, parallel branches, and functional ops without requiring the user to modify the model or run real data through it. Shape annotation uses `torch.fx.passes.shape_prop.ShapeProp` with a user-supplied example input, so shapes are optional rather than mandatory.

Known limitation: symbolic tracing fails on data-dependent control flow (`if x.sum() > 0:`). The planned fallback is a second frontend based on `torch.export` (which traces with fake tensors and supports more programs), selected automatically when `symbolic_trace` raises. Both frontends will emit the same IR, so nothing downstream changes.

### Why a framework-agnostic IR

`torchdiagram.graph` (`Graph` / `Node` / `Edge`) deliberately imports nothing from torch:

- Renderers can be developed and tested with hand-built graphs — no model, no tracing, no torch import cost.
- Users can post-process a traced graph (rename labels, drop nodes) before rendering with plain dataclass manipulation.
- A future non-PyTorch frontend (ONNX, Keras) only has to target the IR.

Node order in `Graph.nodes` is the topological/execution order, which is what `torch.fx` emits; renderers rely on it for layout instead of re-deriving it.

### Why two renderers, and why these two

- **SVG** is the preview path: viewable in any browser or editor, with no toolchain required to produce or inspect a draft.
- **TikZ** is the publication path: a standalone document that compiles as-is, whose `tikzpicture` can be pasted into a paper. Vector output, editable by the user afterwards.

Renderers are pure functions `Graph -> str`; `render()` only dispatches on file extension and writes the file. A new format (PNG via resvg, Mermaid, draw.io XML) is one module plus one dictionary entry.

### Planned: block aggregation

Modern architectures are too deep to render layer-by-layer (ResNet-50 is 177 traced nodes). The plan follows the approach validated by Net2Vis (arXiv:1902.04394): detect repeated subgraph patterns and collapse them into a single labeled block with a repetition count, targeting ~20 glyphs for a ResNet-50. This will be a pure IR→IR transform between the frontend and the renderers.

## Non-goals

The project is deliberately scoped to code-to-diagram generation for static figures:

- **Prompt-to-diagram AI generation** — generating figures from natural-language descriptions is a different problem with a different toolchain; torchdiagram only draws graphs derived from real model code.
- **Interactive editors and 3D visualization** — the output is a static vector figure; interactive editing is left to the SVG/TikZ tools users already have.
- **Training-time visualization** (activations, gradients, metrics) — torchdiagram draws the architecture, not the training process.
