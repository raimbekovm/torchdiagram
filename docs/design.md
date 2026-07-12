# Design

## Pipeline

```text
nn.Module ──▶ frontend (trace.py) ──▶ IR (graph.py) ──▶ transforms (transforms.py) ──▶ renderers (svg, tikz)
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

### Block aggregation

Modern architectures are too deep to render layer-by-layer (ResNet-50 is 177 traced nodes). `aggregate_blocks()` (`transforms.py`) is a pure IR→IR transform, following the spirit of Net2Vis (arXiv:1902.04394): detect repeated block patterns and collapse them into a single labeled node with a repetition count (`"BasicBlock ×5"`).

Detection is a hybrid, not pure subgraph isomorphism:

- **Scope, not just topology, drives grouping.** `trace()` records each node's immediate custom-container ancestor (`Node.scope`/`scope_class`) from fx's `nn_module_stack`, which is populated for every node kind — not just `call_module` — so functional ops called from inside a submodule's `forward()` (a residual `add`, say) still resolve to that submodule. Consecutive nodes sharing a scope form a candidate group; this turns "find repeated subgraphs" into a linear scan instead of general subgraph isomorphism, and gives free, accurate labels.
- **Structure still decides whether groups actually match.** Two groups only merge if they share a `scope_class` _and_ an identical structural signature (op sequence, internal edge topology, external-entry positions). This is what correctly leaves a stage's non-uniform first block (e.g. ResNet's downsample `Bottleneck`) unmerged — it never repeats with a matching signature, so it stays expanded rather than being silently mis-collapsed with the uniform blocks around it.
- **Eligibility is single-entry/single-exit.** A group only merges if exactly one external node feeds into it and its only external exit originates from its last member — this holds for the common case (a block on the trunk, skip connection included) and rules out merging when a group's output is tapped elsewhere.

On real ResNet-50, this takes 177 traced nodes to 61 (each stage's uniform tail collapses to one node; the differing first block per stage stays expanded) — a meaningful reduction, short of the ~20-glyph aspiration from full recursive collapsing.

Deferred to a future iteration:

- **Recursive/hierarchical aggregation** — collapsing a whole stage-of-stages (e.g. also merging `layer1`/`layer2`/`layer3`/`layer4` into one node), which is what would close the gap to ~20 glyphs.
- **Singleton scope collapsing** — a stage's one-off downsample block currently stays fully expanded; collapsing it to a single node too (without a repetition badge) would compact ResNet-scale diagrams further.
- **Non-contiguous repetition** — only runs of _consecutive_ same-scope groups are considered.
- A structural-only fallback for frontends that don't populate `scope` — graphs without scope info simply don't aggregate; this degrades gracefully rather than erroring.

## Non-goals

The project is deliberately scoped to code-to-diagram generation for static figures:

- **Prompt-to-diagram AI generation** — generating figures from natural-language descriptions is a different problem with a different toolchain; torchdiagram only draws graphs derived from real model code.
- **Interactive editors and 3D visualization** — the output is a static vector figure; interactive editing is left to the SVG/TikZ tools users already have.
- **Training-time visualization** (activations, gradients, metrics) — torchdiagram draws the architecture, not the training process.
