# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `torchdiagram.trace()`: trace any fx-traceable `nn.Module` (residuals, branches, functional ops) into a framework-agnostic graph IR, with optional output-shape annotation from an example input. Parameter plumbing and metadata computations (`x.shape[1]`, `x.size(0)`, and the arithmetic over them) are left out of the diagram, with the data flow reconnected across them; layers of the same class at the same level are labeled apart by attribute, e.g. `Embedding (tok_emb)`.
- `torchdiagram.aggregate_blocks()`: collapse scoped blocks into single labeled nodes, recursively from the deepest nesting outward, badging runs of matching blocks as `BasicBlock ×5`. Blocks merge only when their layer configuration and their already-collapsed contents match, so stages differing in width or depth stay separate; blocks collapsed from an `nn.Sequential` and friends are named after the attribute holding them.
- `torch.export` fallback frontend for models `torch.fx` cannot trace, such as those with data-dependent control flow. Selected automatically when symbolic tracing fails, or pinned with `trace(..., backend="export")` and `--backend`. The fallback specializes on the example input and warns that untaken branches are absent from the diagram; both frontends emit the same IR.
- Aggregation reaches two shapes it used to miss: a run of identical leaf layers held directly in an `nn.Sequential` is badged `TransformerEncoderLayer ×4` rather than folded into one unnamed box, and a container level no operation was traced from — a bare `nn.Sequential` between two custom modules — no longer stops the walk up the module tree, so a `DenseNet` shows `DenseBlock` instead of eight flat `DenseLayer` boxes. `Graph.scopes` records the module chain this needs.
- Layers returning `(output, state)` — `nn.LSTM`, `nn.GRU`, `nn.RNN`, `nn.MultiheadAttention` — draw as one box carrying the shape of the output the rest of the model consumes. The fx frontend no longer leaves the tuple unpacking on the diagram as a live `getitem` plus a dead stub, and the export frontend no longer annotates the box with its last internal op's shape, which for an LSTM is the final hidden state rather than the output sequence.
- Both frontends draw a torch-provided composite layer (`nn.TransformerEncoderLayer`, `nn.LSTM`, `nn.MultiheadAttention`) as the single box fx always did, instead of the export frontend drawing it as a box and as its children at once. Tensor subscripts are labeled `index` either way, whichever ATen op the expression lowered to, and a model that is itself a leaf layer draws that layer with its configuration rather than the functional ops fx descends into.
- Multi-input and multi-output models: `trace()` takes a tuple of example inputs, one per `forward()` argument, and the CLI's `--input-shape` is repeatable. Each returned tensor gets its own output node carrying its own shape, labeled by the tuple index or dict key it was returned under (`output[0]`, `output[logits]`); a single return keeps the plain `output` node.
- SVG renderer with curved routing for skip/residual edges.
- TikZ renderer producing a standalone compilable LaTeX document.
- PNG renderer: `torchdiagram.to_png()` rasterizes the SVG output through the optional `resvg-py` dependency, installed with `pip install 'torchdiagram[png]'`. Resolution is set by `scale`, on `render()` and the CLI's `--scale`.
- `torchdiagram.render()` dispatching on output file extension (`.svg`, `.tex`, `.tikz`, `.png`).
- CLI: `torchdiagram package.module:Model -o model.svg --input-shape 1,3,224,224`.
- Documentation under `docs/`: getting started, user guide, CLI reference, API reference, and design notes.
