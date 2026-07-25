"""Publication-ready neural-network architecture diagrams straight from PyTorch code."""

from importlib.metadata import PackageNotFoundError, version

from torchdiagram.graph import Edge, Graph, Node
from torchdiagram.renderers import render, to_svg, to_tikz
from torchdiagram.theme import DARK, DEFAULT, MONOCHROME, Theme
from torchdiagram.trace import trace
from torchdiagram.transforms import aggregate_blocks

try:
    __version__ = version("torchdiagram")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = [
    "DARK",
    "DEFAULT",
    "MONOCHROME",
    "Edge",
    "Graph",
    "Node",
    "Theme",
    "__version__",
    "aggregate_blocks",
    "render",
    "to_svg",
    "to_tikz",
    "trace",
]
