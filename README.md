# torchdiagram

> Architecture block diagrams generated directly from PyTorch models.

[![CI](https://github.com/raimbekovm/torchdiagram/actions/workflows/ci.yml/badge.svg)](https://github.com/raimbekovm/torchdiagram/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/raimbekovm/torchdiagram/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/raimbekovm/torchdiagram/blob/main/LICENSE)

torchdiagram traces an `nn.Module` with `torch.fx` and renders the resulting computation graph as a block diagram. Because the diagram is derived from the traced `forward()`, it reflects the actual data flow of the model — including residual connections, parallel branches, and functional ops — rather than a manually maintained description of it. Models that `torch.fx` cannot trace, such as those branching on tensor values, fall back to a `torch.export` frontend automatically.

Three output formats are supported:

- **SVG** — a self-contained image, viewable in any browser or editor.
- **TikZ** — a standalone LaTeX document suitable for publications; it compiles as-is, and its `tikzpicture` environment can be copied into an existing paper.
- **PNG** — the SVG rasterized, for slides, issue threads, and anywhere else vector graphics aren't displayed. Needs one optional dependency (`pip install 'torchdiagram[png]'`).

Generating any of them requires no LaTeX toolchain.

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
td.render(graph, "block.png")  # rasterized image
```

The example input is optional; when provided, every node in the diagram is annotated with its output shape.

For deep models, `td.aggregate_blocks(graph)` collapses runs of repeated blocks (e.g. ResNet layers) into a single labeled node before rendering — see [the user guide](docs/usage.md#aggregating-repeated-blocks).

The same pipeline is available from the command line:

```bash
torchdiagram my_models:ResidualBlock -o block.svg --input-shape 1,64,56,56
```

## Installation

Not on PyPI yet — install from source:

```bash
pip install git+https://github.com/raimbekovm/torchdiagram

# with PNG output
pip install 'torchdiagram[png] @ git+https://github.com/raimbekovm/torchdiagram'
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
nn.Module ──▶ torch.fx trace (or torch.export) ──▶ framework-agnostic IR ──▶ SVG / TikZ / PNG renderer
```

The tracer converts the fx graph into a small intermediate representation (`Graph` / `Node` / `Edge` dataclasses with no torch dependency). Renderers are pure functions over that IR, so traced graphs can be inspected or edited before rendering, and graphs can also be built by hand.

## Status & roadmap

Pre-alpha. The core pipeline works end-to-end:

- [x] Trace any fx-traceable `nn.Module` (residuals, branches, functional ops)
- [x] Output-shape annotations via a single example input
- [x] SVG renderer with skip-edge routing
- [x] TikZ renderer (standalone compilable document)
- [x] CLI (`torchdiagram pkg.module:Model -o out.svg`)
- [x] Block aggregation — collapse repeated layers so deep networks render compactly
- [x] Presets for attention/transformer blocks
- [x] Styling/theme API
- [x] Fallback tracer for data-dependent control flow (`torch.export`)
- [x] PNG export
- [ ] PyPI release

**Known limitation:** a model that branches on tensor values (`if x.sum() > 0:`) is traced by the `torch.export` fallback, which specializes on the example input. The diagram then shows the branch that input takes and omits the others, and `trace()` warns when this happens.

## FAQ

**Does torchdiagram require a LaTeX installation?**
No. Generating SVG, PNG, or TikZ output needs no LaTeX toolchain; LaTeX is only required if you want to compile the generated `.tex` file yourself with `pdflatex` or `tectonic`.

**Does tracing a model require a GPU?**
No. `torch.fx` symbolic tracing and shape propagation both run on CPU tensors; the example input passed to `trace()` only needs to match the shape and dtype `forward()` expects.

**Can it diagram models with residual connections or branching, like ResNet or a Transformer block?**
Yes. Because the diagram comes from a real `torch.fx` trace of `forward()`, residual connections, parallel branches, and functional ops (`torch.relu`, `x + y`, `x.view(...)`) are captured as part of the graph rather than requiring manual annotation.

**What happens if my model has data-dependent control flow (`if x.sum() > 0:`)?**
`trace()` falls back to a second frontend built on `torch.export`, which runs the model on the example input and records the branch that input actually takes. The diagram is therefore a specialization: branches the input does not take are absent, and `trace()` emits a warning saying so. This fallback needs an example input, since `torch.export` traces by running the model. Both frontends emit the same IR, so shapes, block aggregation, and every renderer behave identically either way.

**Can it export the diagram as a PNG image?**
Yes. `td.render(graph, "model.png")` writes a PNG, and the CLI does the same for a `.png` output path, with `--scale` controlling the resolution. PNG is the SVG rendering rasterized, so it matches the vector output exactly. It is the one format with an extra dependency: install it with `pip install 'torchdiagram[png]'`.

**Can I edit the diagram after generating it?**
Yes, at two levels: the intermediate `Graph` returned by `trace()` is a plain dataclass you can modify (rename labels, drop nodes) before rendering, and the rendered SVG is editable in any vector graphics editor while the TikZ output is plain LaTeX you can edit directly.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and guidelines. Bug reports with a minimal model snippet are especially useful.

## License

[MIT](LICENSE)
