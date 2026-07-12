# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `torchdiagram.trace()`: trace any fx-traceable `nn.Module` (residuals, branches, functional ops) into a framework-agnostic graph IR, with optional output-shape annotation from an example input.
- SVG renderer with curved routing for skip/residual edges.
- TikZ renderer producing a standalone compilable LaTeX document.
- `torchdiagram.render()` dispatching on output file extension (`.svg`, `.tex`, `.tikz`).
- CLI: `torchdiagram package.module:Model -o model.svg --input-shape 1,3,224,224`.
- Documentation under `docs/`: getting started, user guide, CLI reference, API reference, and design notes.
