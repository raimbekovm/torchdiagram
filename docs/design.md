# Design

## Pipeline

```text
nn.Module ──▶ frontend (trace.py, export_trace.py, frontend.py) ──▶ IR (graph.py) ──▶ transforms (transforms.py) ──▶ renderers (svg, tikz, png)
```

Each stage only knows about the one after it, and everything downstream of the frontend is torch-free.

## Decisions

### Why `torch.fx` for tracing

The core promise of the tool is "diagram from arbitrary `forward()` code", and `torch.fx.symbolic_trace` is the lowest-friction way to get a faithful dataflow graph out of one: it captures residual connections, parallel branches, and functional ops without requiring the user to modify the model or run real data through it. Shape annotation uses `torch.fx.passes.shape_prop.ShapeProp` with a user-supplied example input, so shapes are optional rather than mandatory.

Known limitation: symbolic tracing fails on data-dependent control flow (`if x.sum() > 0:`). That case is covered by a second frontend, selected automatically when `symbolic_trace` raises; see below.

### The `torch.export` fallback

Plain `torch.export` does **not** fix data-dependent control flow. It traces with fake tensors, which carry shapes but no values, so `if x.sum() > 0:` hits the same wall fx does — measured on torch 2.13:

| API                                 | `if x.sum() > 0:`                                   |
| ----------------------------------- | --------------------------------------------------- |
| `torch.export.export(strict=True)`  | fails — `Unsupported: Data-dependent branching`     |
| `torch.export.export(strict=False)` | fails — `GuardOnDataDependentSymNode`               |
| `make_fx(tracing_mode="real")`      | fails — `aten._local_scalar_dense`                  |
| `torch.export.draft_export`         | works — real-tensor propagation resolves the branch |

Only `draft_export` gets through, by propagating real tensors alongside the fake ones and **specializing** on the example input. That is a real semantic difference, not an implementation detail: the resulting graph describes the branch that one input takes, and a different input can yield a different diagram. `trace()` says so with a `UserWarning` rather than quietly presenting a partial trace as the model.

The frontend therefore tries the sound path first (`export(strict=False)`, no warning when it succeeds) and only then `draft_export`. Soundness is read from `draft_export`'s own report; a missing report is treated as unsound rather than assumed fine.

Three properties of export graphs make this a small module rather than a second tracer:

- **Module identity survives.** Export emits ATen ops, so the naive reading gives `aten.convolution.default` where fx gives `Conv2d`. But `nn_module_stack` is populated on every node, and walking it with fx's own `Tracer.is_leaf_module` predicate reproduces fx's labels exactly. The walk has to go **outermost-first**, which is the direction fx applies the same predicate while tracing: the first leaf on the path stops the descent, and nothing inside it is visited. Asking about the innermost entry instead makes every torch.nn descendant of a torch.nn composite qualify on its own, so an `nn.TransformerEncoderLayer` comes out as one box _and_ as its nine children in the same diagram. One layer can lower to many ATen ops (`nn.MultiheadAttention` becomes 28), so nodes sharing a leaf path collapse into one box.
- **Shapes are free.** Every node carries `meta["val"]`, so this frontend needs no `ShapeProp` pass. The `output` node is the exception — export leaves it without `val` and flattens the returned structure away, so each output box takes its shape from the node feeding it and its name from the module's output pytree spec.
- **Guard plumbing has a clean signature.** `draft_export` records the specialized condition as `aten.item` / `sym_ite` / `operator.ge` / `aten._assert_scalar` nodes, plus a synthetic `_guards_fn` module. None of them produce tensor data, so one test — is `meta["val"]` a tensor, or a tuple containing one — drops all of it without touching a real node. The tuple case matters for genuinely multi-output ops (`torch.max(dim=...)`, `split`, `topk`).

    Dropping only those leaves a stub, though: the condition the model really did compute (`x.sum() > 0` lowers to `sum` then `gt` then `ne`) is tensor-valued, and once its consumers are gone it trails off the side of the diagram as a dangling chain that corresponds to nothing in the architecture. So the rule extends one step: a node whose consumers were _all_ dropped as plumbing is plumbing too, applied in reverse topological order so one pass clears a whole chain. A node with **no** consumers at all is deliberately kept — that is dead code the model genuinely contains, which the fx frontend draws, and dropping it here would make the two frontends disagree.

The same `nn_module_stack` walk yields `scope`/`scope_class`, so `aggregate_blocks()` works on export graphs unchanged. On the models both frontends can trace, they emit identical graphs, which is asserted in the test suite.

Subscripting a tensor is where holding that equality takes the most work, because the two tracers disagree a level above naming: fx records the Python operator, export records the ATen op it became. `x[:, 0]` is `operator.getitem` to one and `aten.select.int` to the other, and `x[..., :-1]` is `aten.slice.Tensor`. Two rules close the gap. Every form of subscript is labeled `index` on both sides, since the level a diagram reads at is "this indexes the tensor", not which overload did it. And both frontends collapse a run of indexing ops that came from one source line and feed nothing but each other into a single box.

Collapsing has to happen on both sides because each tracer splits a subscript the other keeps whole, and neither split is recoverable from the other's graph. Export lowers one subscript per axis, so `x[:, 0, :-1]` becomes a `select` then a `slice` against fx's single node. fx splits a chained subscript, so `x[0][1]` is two `getitem` nodes against export's one — and export cannot recover the two, because `x[0, 1]` lowers to exactly the same pair of `select` ops on the same line, and the graph carries no column information to separate them. Collapsing both leaves one box for one indexing step, whichever tracer produced it, and leaves subscripts written on separate lines as a box each, which is what the model was written as.

fx does not record a source line by default, and turning on `Tracer.record_stack_traces` to get one captures and _formats_ a whole traceback per node. Tracing a GPT-2 goes from 18 ms to 511 ms — the formatting reads every source file it names, which is warm on a second trace and cold on the only one a CLI run performs. `_LineTracer` walks the frames without formatting them instead, which costs nothing measurable (19 ms). It records the whole chain of model frames rather than the innermost one, because a subscript is free to live in a helper — a submodule applied twice, a function called from two places — and every call of it then reports the same innermost line. Two separate operations would fold into one box, which reads as the second submodule doing nothing and the first producing a shape it never produced. The call site is what separates them, and it is what `torch.export` records.

Holding that equality is what forced the fx frontend to learn the same lesson about metadata. Export resolves `x.shape[1]` into symbolic shapes and drops the result by the tensor-valued test above; fx keeps it as Python-level `getattr`, `getitem`, and `floordiv` nodes, which used to leave a hand-written transformer block at 27 nodes against export's 22, and put a row of boxes reading `getattr` above a GPT diagram. fx has no `meta["val"]` to test, so it classifies the same nodes by what they ask for instead: a read of a metadata attribute or method (`shape`, `dtype`, `size()`, `numel()`), plus any plain Python arithmetic or indexing that consumes nothing else. The propagation stops at the first torch call, which is the point — `torch.arange(x.shape[1])` produces a tensor and belongs in the diagram. Both frontends then reconnect what they kept across what they dropped (`frontend.build_edges`), so the range still shows as depending on the input rather than floating loose.

The mirror of that is a tensor built from nothing. `torch.arange(8)` hands fx no proxy to trace through, so fx runs the call while tracing and keeps the result as a graph attribute: the diagram reads `_tensor_constant0`, a name torch invented, where export reads `arange`, and whatever the model computes from that tensor is folded away with it and never drawn at all. Nothing links the attribute back to the call, so there is no recovering it after the fact. The ten factories a model plausibly writes in `forward()` are held back for the duration of a trace instead, each recording a node rather than running — but only when the model's own code is what called it, since torch builds tensors constantly on its way down to an operation and handing one of those a proxy breaks a trace over something the model never wrote. A model can also want the tensor rather than the operation, iterating a range or reading an element out of it to branch on, and no proxy stands in for that; there is no telling which model that is in advance, so a trace that fails with the factories held back is retried without them and draws exactly what it drew before. `torch.tensor` is left out on purpose: export lifts a literal into a constant of its own rather than keeping the call, so tracing it would trade one disagreement between the frontends for another. Not folding constants also makes what the model does to them visible, which is how dtype casts came to be labeled `to` on both sides — fx keeps `x.float()` as a method of its own where export lowers every cast the same way.

Reconnecting only works where the dependency survived tracing, and on the export side it does not: `torch.arange(idx.shape[1])` is evaluated during export, and the ATen graph holds `arange(16)` with nothing tying it to the placeholder. The dependency is recoverable, just not from that graph — exporting again with every input dimension marked `Dim.AUTO` leaves the read standing as a `sym_size` node naming the placeholder it came from. A second export costs as much as the first, so it is attempted only for a node that ended up with no incoming edge at all, which is the symptom and is rare: one of the sixteen reference architectures, and no cost for the other fifteen. `Dim.AUTO` specializes back any dimension the model turns out to need fixed, so the dynamic export is not a claim that the model is shape-generic — it is the same trace with as little constant folding as torch will agree to.

What comes back is the _placeholder_, which is not always the box fx points at. `torch.arange(h.shape[1])`, where `h` is an activation, gives fx an arrow from the layer that produced `h`, because its edge rebuilder stops at the nearest box it kept; export gives an arrow from the input, because the symbol for `h`'s length is the input's length and torch records the read against the input. Neither side can be moved to the other: the export graph has no trace of `h` left, and making fx walk past a box it kept would break the case `frontend._sources` exists for. Both arrows are true, and they point at different ends of the same dependency.

One practical wrinkle: `draft_export`'s real-tensor logging calls `inspect.getsourcelines` on the user frame, so a model defined in a REPL or notebook, where no source file exists, can fail with `OSError`. It is caught with the other export failures, and the original fx error — which describes the model rather than the fallback — is what reaches the user.

### Why a framework-agnostic IR

`torchdiagram.graph` (`Graph` / `Node` / `Edge`) deliberately imports nothing from torch:

- Renderers can be developed and tested with hand-built graphs — no model, no tracing, no torch import cost.
- Users can post-process a traced graph (rename labels, drop nodes) before rendering with plain dataclass manipulation.
- A future non-PyTorch frontend (ONNX, Keras) only has to target the IR.

Node order in `Graph.nodes` is the topological/execution order, which is what `torch.fx` emits; renderers rely on it for layout instead of re-deriving it.

### Why these renderers

- **SVG** is the preview path: viewable in any browser or editor, with no toolchain required to produce or inspect a draft.
- **TikZ** is the publication path: a standalone document that compiles as-is, whose `tikzpicture` can be pasted into a paper. Vector output, editable by the user afterwards.
- **PNG** is the paste-anywhere path, for the surfaces that won't display vector graphics at all: slide decks, issue threads, chat.

Renderers are pure functions of the IR; `render()` only dispatches on file extension and writes the file. A new text format (Mermaid, draw.io XML) is one module plus one dictionary entry.

PNG is deliberately not a renderer of its own: it rasterizes the SVG rather than laying the graph out again, so the two can never drift apart, and every layout fix lands in one place. Rasterization goes through the `resvg-py` wheel, chosen over cairosvg because it needs no system libraries, and over an SVG-to-PDF converter because the output has to be embeddable as an image. It is an **optional** dependency behind the `png` extra: the formats a paper or README usually wants need nothing beyond the standard library, and `to_png()` fails with an `ImportError` naming the extra rather than making every install carry a rasterizer. It is also the one format with a `scale` knob, since a raster image, unlike SVG and TikZ, has to commit to a resolution.

### Block aggregation

Modern architectures are too deep to render layer-by-layer (ResNet-50 is 177 traced nodes). `aggregate_blocks()` (`transforms.py`) is a pure IR→IR transform, following the spirit of Net2Vis (arXiv:1902.04394): detect repeated block patterns and collapse them into a single labeled node with a repetition count (`"BasicBlock ×5"`).

Detection is a hybrid, not pure subgraph isomorphism:

- **Scope, not just topology, drives grouping.** `trace()` records each node's immediate custom-container ancestor (`Node.scope`/`scope_class`) from fx's `nn_module_stack`, which is populated for every node kind — not just `call_module` — so functional ops called from inside a submodule's `forward()` (a residual `add`, say) still resolve to that submodule. Consecutive nodes sharing a scope form a candidate group; this turns "find repeated subgraphs" into a linear scan instead of general subgraph isomorphism, and gives free, accurate labels.
- **Structure still decides whether groups actually match.** Two groups only merge if they share a `scope_class` _and_ an identical structural signature (op sequence, internal edge topology, external-entry positions). This is what correctly leaves a stage's non-uniform first block (e.g. ResNet's downsample `Bottleneck`) unmerged — it never repeats with a matching signature, so it stays expanded rather than being silently mis-collapsed with the uniform blocks around it.

- **The signature has to reach through what was already collapsed.** A `×N` badge asserts that N interchangeable blocks were found, and two things that look alike after collapsing need not be alike before it. Per-node keys therefore carry the layer's `extra_repr()` configuration, so a VGG stage at 64 channels does not merge with the next one at 128; and a collapsed block is keyed not by its class but by a digest of the signature it was built from, recorded as the run is synthesized and looked up one level out. Without the digest, a stage holding two convolutions and a stage holding three both reduce to `[block:Sequential, maxpool2d]` once their inner `nn.Sequential` collapses, and VGG-16 badges as `VGGBlock ×5` — 13 convolutions of three different widths compressed into one node that claims they repeat. The digest is what makes the deepest-first pass order safe rather than lossy.

- **Container classes don't name a block.** A block collapsed from an `nn.Sequential`, `nn.ModuleList`, or `nn.ModuleDict` is labeled with the attribute holding it (`classifier`) rather than the container's class name, which describes the data structure and not the computation. `block_class` in `params` keeps the real class, so structural identity is unaffected.
- **Eligibility is single-entry/single-exit.** A group only merges if exactly one external node feeds into it and its only external exit originates from its last member — this holds for the common case (a block on the trunk, skip connection included) and rules out merging when a group's output is tapped elsewhere.
- **Eligible groups always collapse, even alone.** A group that passes the eligibility check collapses into a single node whether or not it repeats; only groups that additionally share a `scope_class` and structural signature with their neighbors get _merged_ into one badged node (`"BasicBlock ×5"`) instead of each becoming its own unbadged node. This is what lets a stage's non-uniform first block (e.g. ResNet's downsample `Bottleneck`) collapse to a single node too, without being silently merged with the uniform blocks around it.
- **Collapsing runs deepest-scope-first, one depth of nesting per pass.** A node's scope nesting depth is read off its dotted path (`"blocks.0.attn"` is deeper than `"blocks.0"`). `aggregate_blocks()` processes the deepest depth first and works outward, assigning each newly collapsed node the scope of its parent container before moving up. This is what makes transformer blocks collapse well: an attention or MLP sub-block occurs only once per transformer block, so it collapses on the deep pass (unbadged, by the point above); the transformer block's own node sequence is then short and uniform across the stack, so the next pass up finds it repeats and merges it into one `"TransformerBlock ×N"` node — no attention-specific code involved, just the same scope+structure rules applied recursively, provided the parent container qualifies for the next bullet.

On real ResNet-50, this takes 177 traced nodes to 17 (each stage's uniform tail merges into one badged node; the differing first block per stage collapses to its own unbadged node) — beyond the ~20-glyph aspiration for full recursive collapsing.

Deferred to a future iteration:

- **Non-contiguous repetition** — only runs of _consecutive_ same-scope groups are considered.
- **A level that owns no operation is still a level.** A synthetic node's `scope_class` for the next pass cannot be looked up from other nodes in this pass, because a container that does nothing but call its children (`forward(self, x): return self.mlp(self.attn(x))`, or a bare `nn.Sequential`) is never any node's `scope`. The map would have a hole exactly there, the lookup would come back `None`, and the walk up the tree would end one level short — a DenseNet showing eight flat `DenseLayer` boxes and no `DenseBlock`. The frontends therefore record the whole `nn_module_stack` chain in `Graph.scopes`, not just each node's immediate parent, so the tree the transform walks is complete.

    Completing the tree exposes two rules the walk needs. A run of groups merges only among **siblings**: `blocks.0.layers` and `blocks.1.layers` look identical and are not repeats of each other at that level — the repetition is one level up, where their parents are, and merging them would file one node under an arbitrary parent. And a bare container holding nothing but already-collapsed blocks is **unwrapped** rather than collapsed: replacing `BasicBlock ×4` with a box named `layer1` after the attribute holding it says strictly less. A container holding raw operations still collapses, since `classifier` is the only name those three layers have.

- A structural-only fallback for frontends that don't populate `scope` — graphs without scope info simply don't aggregate; this degrades gracefully rather than erroring.

### Theming

`Theme` (`theme.py`) is a frozen, torch-free dataclass — like the IR — that every renderer accepts as an optional keyword. It is deliberately **color-first**: colors are the one part of the visual contract that maps cleanly onto both backends (SVG writes them literally; TikZ emits a `\definecolor` per field and references it by name), so a single `Theme` value drives an SVG preview and its TikZ counterpart to the same look. This also unified the two renderers' defaults: TikZ previously used its own grayscale palette (`black!60`/`black!5`), and now shares `DEFAULT`'s blue palette with SVG — the old grayscale look is available as the `MONOCHROME` preset.

Two length fields (`corner_radius`, `stroke_width`) are carried too, each applied in the backend's native unit (pixels in SVG, points in TikZ) — a documented approximation rather than an exact cross-backend match. Sharing `DEFAULT` also nudges the TikZ default geometry to match SVG's: block corners move from `2pt` to `6pt` and the outline gains an explicit `1.2pt` width. `font_family` is SVG-only, since LaTeX font selection is a preamble/package concern.

Deferred: **themeable geometry** (node spacing, box size). SVG lays out in pixels and TikZ in millimeters, so a shared "node distance" field would have no single honest unit; the spacing constants stay renderer-internal until there's a reason to abstract them. Frontends and renderers that don't recognize a field simply ignore it — the same graceful-degradation stance as the rest of the pipeline.

## Non-goals

The project is deliberately scoped to code-to-diagram generation for static figures:

- **Prompt-to-diagram AI generation** — generating figures from natural-language descriptions is a different problem with a different toolchain; torchdiagram only draws graphs derived from real model code.
- **Interactive editors and 3D visualization** — the output is a static vector figure; interactive editing is left to the SVG/TikZ tools users already have.
- **Training-time visualization** (activations, gradients, metrics) — torchdiagram draws the architecture, not the training process.
