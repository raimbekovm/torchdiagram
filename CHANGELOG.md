# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `torchdiagram.trace()`: trace any fx-traceable `nn.Module` (residuals, branches, functional ops) into a framework-agnostic graph IR, with optional output-shape annotation from an example input. Parameter plumbing and metadata computations (`x.shape[1]`, `x.size(0)`, and the arithmetic over them) are left out of the diagram, with the data flow reconnected across them; layers of the same class at the same level are labeled apart by attribute, e.g. `Embedding (tok_emb)`.
- `torchdiagram.aggregate_blocks()`: collapse scoped blocks into single labeled nodes, recursively from the deepest nesting outward, badging runs of matching blocks as `BasicBlock ×5`. Blocks merge only when their layer configuration and their already-collapsed contents match, so stages differing in width or depth stay separate; blocks collapsed from an `nn.Sequential` and friends are named after the attribute holding them.
- `torch.export` fallback frontend for models `torch.fx` cannot trace, such as those with data-dependent control flow. Selected automatically when symbolic tracing fails, or pinned with `trace(..., backend="export")` and `--backend`. The fallback specializes on the example input and warns that untaken branches are absent from the diagram; both frontends emit the same IR.
- SVG renderer with curved routing for skip/residual edges.
- TikZ renderer producing a standalone compilable LaTeX document.
- PNG renderer: `torchdiagram.to_png()` rasterizes the SVG output through the optional `resvg-py` dependency, installed with `pip install 'torchdiagram[png]'`. Resolution is set by `scale`, on `render()` and the CLI's `--scale`.
- `torchdiagram.render()` dispatching on output file extension (`.svg`, `.tex`, `.tikz`, `.png`).
- CLI: `torchdiagram package.module:Model -o model.svg --input-shape 1,3,224,224`.
- Documentation under `docs/`: getting started, user guide, CLI reference, API reference, and design notes.
