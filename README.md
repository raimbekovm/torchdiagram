# torchdiagram

> Architecture block diagrams generated directly from PyTorch models.

[![CI](https://github.com/raimbekovm/torchdiagram/actions/workflows/ci.yml/badge.svg)](https://github.com/raimbekovm/torchdiagram/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/raimbekovm/torchdiagram/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/raimbekovm/torchdiagram/blob/main/LICENSE)

torchdiagram traces an `nn.Module` with `torch.fx` and renders the resulting computation graph as a block diagram. Because the diagram is derived from the traced `forward()`, it reflects the actual data flow of the model — including residual connections, parallel branches, and functional ops — rather than a manually maintained description of it.

Two output formats are supported:

- **SVG** — a self-contained image, viewable in any browser or editor.
- **TikZ** — a standalone LaTeX document suitable for publications; it compiles as-is, and its `tikzpicture` environment can be copied into an existing paper.

Generating either format requires no LaTeX toolchain.

## Usage

```python
import torch
from torch import nn

import torchdiagram as td


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv2(torch.relu(self.conv1(x))) + x)


graph = td.trace(ResidualBlock(), torch.randn(1, 64, 56, 56))
td.render(graph, "block.svg")  # SVG image
td.render(graph, "block.tex")  # standalone TikZ/LaTeX document
```

The example input is optional; when provided, every node in the diagram is annotated with its output shape.

The same pipeline is available from the command line:

```bash
torchdiagram my_models:ResidualBlock -o block.svg --input-shape 1,64,56,56
```

## Installation

Not on PyPI yet — install from source:

```bash
pip install git+https://github.com/raimbekovm/torchdiagram
```

Requires Python 3.10+ and PyTorch 2.0+.

## Documentation

- [Getting started](docs/getting-started.md) — installation and a first diagram.
- [User guide](docs/usage.md) — tracing, shape annotations, output formats, and editing the graph before rendering.
- [CLI reference](docs/cli.md) — the `torchdiagram` command.
- [API reference](docs/api.md) — the public Python API.
- [Design](docs/design.md) — internal architecture and technical decisions.

## How it works

```text
nn.Module ──▶ torch.fx trace ──▶ framework-agnostic IR ──▶ SVG / TikZ renderer
```

The tracer converts the fx graph into a small intermediate representation (`Graph` / `Node` / `Edge` dataclasses with no torch dependency). Renderers are pure functions over that IR, so traced graphs can be inspected or edited before rendering, and graphs can also be built by hand.

## Status & roadmap

Pre-alpha. The core pipeline works end-to-end:

- [x] Trace any fx-traceable `nn.Module` (residuals, branches, functional ops)
- [x] Output-shape annotations via a single example input
- [x] SVG renderer with skip-edge routing
- [x] TikZ renderer (standalone compilable document)
- [x] CLI (`torchdiagram pkg.module:Model -o out.svg`)
- [ ] Block aggregation — collapse repeated layers so deep networks render compactly
- [ ] Presets for attention/transformer blocks
- [ ] Styling/theme API
- [ ] Fallback tracer for data-dependent control flow (`torch.export`)
- [ ] PNG export
- [ ] PyPI release

**Known limitation:** `torch.fx` symbolic tracing cannot handle data-dependent control flow (e.g. `if x.sum() > 0:` inside `forward()`). This is the standard fx restriction; a fallback tracer is on the roadmap.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and guidelines. Bug reports with a minimal model snippet are especially useful.

## License

[MIT](LICENSE)
