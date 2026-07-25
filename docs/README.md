# torchdiagram documentation

torchdiagram generates architecture block diagrams directly from PyTorch models. It traces an `nn.Module` with `torch.fx`, converts the traced graph into a small framework-agnostic intermediate representation, and renders that representation as an SVG image or a standalone TikZ/LaTeX document.

## Contents

| Document                              | Description                                                                                            |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [Getting started](getting-started.md) | Requirements, installation, and a first diagram.                                                       |
| [User guide](usage.md)                | Tracing models, shape annotations, output formats, and working with the graph IR.                      |
| [CLI reference](cli.md)               | The `torchdiagram` command: arguments, model specification, and examples.                              |
| [API reference](api.md)               | Public Python API: `trace`, `render`, `to_svg`, `to_tikz`, and the `Graph`/`Node`/`Edge` data classes. |
| [Design](design.md)                   | Internal architecture: the pipeline, the IR, and the reasoning behind the main technical decisions.    |

## Overview

The library consists of three parts:

1. **Tracer** (`torchdiagram.trace`) — symbolically traces a model's `forward()` with `torch.fx` and emits a graph of layers, function calls, and data-flow edges. Residual connections, parallel branches, and functional ops are captured as part of the trace. An optional example input enables output-shape annotation on every node. Models `torch.fx` cannot trace, such as those branching on tensor values, fall back to a `torch.export` frontend that emits the same graph.
2. **Intermediate representation** (`torchdiagram.graph`) — plain dataclasses (`Graph`, `Node`, `Edge`) with no torch dependency. Traced graphs can be inspected and edited before rendering, and graphs can also be constructed by hand.
3. **Renderers** (`torchdiagram.renderers`) — pure functions from a `Graph` to a string. Two formats are provided: SVG (self-contained, viewable in any browser or editor) and TikZ (a standalone LaTeX document suitable for publications). Producing either format requires no LaTeX toolchain.

A command-line interface wraps the full pipeline: `torchdiagram package.module:Model -o model.svg`.
