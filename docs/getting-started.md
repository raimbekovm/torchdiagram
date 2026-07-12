# Getting started

## Requirements

- Python 3.10 or newer
- PyTorch 2.0 or newer

## Installation

The package is not yet published on PyPI. Install from source:

```bash
pip install git+https://github.com/raimbekovm/torchdiagram
```

Or with uv:

```bash
uv add git+https://github.com/raimbekovm/torchdiagram
```

Installing torchdiagram does not install or require a LaTeX distribution. LaTeX is only needed if you want to compile the generated `.tex` files yourself.

## A first diagram

Define a model (or import an existing one) and pass it to `trace()`:

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

`block.svg` can be opened directly in a browser or embedded in HTML/Markdown. `block.tex` compiles as-is with `pdflatex` or `tectonic`, and its `tikzpicture` environment can be copied into an existing LaTeX document.

The second argument to `trace()` is optional. When provided, torchdiagram runs shape propagation and annotates every node with its output shape; without it, the diagram shows the layer structure only.

## From the command line

The same pipeline is available without writing any code:

```bash
torchdiagram my_models:ResidualBlock -o block.svg --input-shape 1,64,56,56
```

`my_models:ResidualBlock` is an import path: the module `my_models` (a file `my_models.py` in the current directory, or any importable module) and the attribute `ResidualBlock` inside it. See the [CLI reference](cli.md) for the accepted forms.

## Next steps

- [User guide](usage.md) — shape annotations, output formats, editing the graph before rendering, and current tracing limitations.
- [API reference](api.md) — the full public API.
