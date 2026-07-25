"""Command-line interface: ``torchdiagram my_package.models:MyNet -o my_net.svg``."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import warnings

import torch
from torch import nn

from torchdiagram.renderers import render
from torchdiagram.theme import DARK, DEFAULT, MONOCHROME
from torchdiagram.trace import BACKENDS, trace
from torchdiagram.transforms import aggregate_blocks

_THEMES = {"default": DEFAULT, "mono": MONOCHROME, "dark": DARK}


def main(argv: list[str] | None = None) -> None:
    """Run the ``torchdiagram`` command-line entry point.

    Args:
        argv: Command-line arguments to parse, excluding the program name. Defaults to ``sys.argv[1:]`` when ``None``.

    Raises:
        SystemExit: If the model cannot be loaded, ``--input-shape`` is invalid or missing when the chosen backend needs
            it, no tracing frontend can handle the model, the output path has an unsupported extension, or ``.png``
            output was requested without the optional rasterizer installed.
    """
    parser = argparse.ArgumentParser(
        prog="torchdiagram",
        description="Generate an architecture diagram from a PyTorch model.",
    )
    parser.add_argument(
        "model",
        help="Import path to the model as 'package.module:attr'; attr may be an nn.Module instance, "
        "an nn.Module subclass, or a zero-argument factory function.",
    )
    parser.add_argument("-o", "--output", required=True, help="Output file (.svg, .png, .tex, .tikz).")
    parser.add_argument(
        "--input-shape",
        action="append",
        help="Comma-separated input shape, e.g. '1,3,224,224'; enables shape annotations on every node. Repeat it "
        "once per forward() argument for a model taking more than one input.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Collapse runs of repeated, structurally identical blocks (e.g. ResNet layers) into one node.",
    )
    parser.add_argument(
        "--min-repeats",
        type=int,
        default=2,
        help="Minimum run length required to collapse with --aggregate (default: 2).",
    )
    parser.add_argument(
        "--theme",
        choices=sorted(_THEMES),
        default="default",
        help="Color theme for the diagram (default: default).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Pixel scale factor for .png output (default: 2.0); ignored by the vector formats.",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="auto",
        help="Tracing frontend: 'auto' falls back to torch.export when torch.fx cannot trace the model, "
        "'fx' and 'export' pin one (default: auto).",
    )
    args = parser.parse_args(argv)

    model = _load_model(args.model)
    example = _build_example_input(args.input_shape)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        try:
            graph = trace(model, example, backend=args.backend)
        except torch.fx.proxy.TraceError as exc:
            # `from None`: the message carries what the user needs, and a chained traceback through torch internals is
            # the thing these errors exist to replace.
            raise SystemExit(f"error: cannot trace {args.model!r}: {exc}") from None
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from None
    for warning in caught:
        if issubclass(warning.category, UserWarning):
            print(f"note: {warning.message}", file=sys.stderr)
    if args.aggregate:
        graph = aggregate_blocks(graph, min_repeats=args.min_repeats)
    try:
        path = render(graph, args.output, theme=_THEMES[args.theme], scale=args.scale)
    except (ImportError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"wrote {path}")


def _build_example_input(input_shapes: list[str] | None) -> tuple[torch.Tensor, ...] | None:
    """Parse the ``--input-shape`` values into example tensors for shape propagation.

    Args:
        input_shapes: One comma-separated dimension list per ``forward()`` argument, e.g. ``["1,3,224,224"]``, or
            ``None`` to skip shape annotation.

    Returns:
        A randomly initialized tensor per requested shape, in argument order, or ``None`` if no shape was given.

    Raises:
        SystemExit: If a shape is not a comma-separated list of integers, or any dimension is invalid.
    """
    if not input_shapes:
        return None
    tensors = []
    for input_shape in input_shapes:
        try:
            shape = tuple(int(dim.strip()) for dim in input_shape.split(","))
            tensors.append(torch.randn(*shape))
        except (ValueError, RuntimeError) as exc:
            raise SystemExit(f"error: invalid --input-shape {input_shape!r}: {exc}") from exc
    return tuple(tensors)


def _load_model(spec: str) -> nn.Module:
    """Resolve an ``nn.Module`` instance from an import spec.

    Args:
        spec: Import path in the form ``"package.module:attr"``, where ``attr`` is an ``nn.Module`` instance, an
            ``nn.Module`` subclass, or a zero-argument factory function returning one.

    Returns:
        The resolved module instance.

    Raises:
        SystemExit: If ``spec`` is malformed, the module cannot be imported, or ``attr`` does not resolve to an
            ``nn.Module``.
    """
    module_path, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"error: expected 'package.module:attr', got {spec!r}")
    # Console scripts don't put the working directory on sys.path; users expect
    # 'torchdiagram my_models:Net' to work from the directory holding my_models.py.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        obj = getattr(importlib.import_module(module_path), attr)
    except (ImportError, AttributeError) as exc:
        raise SystemExit(f"error: cannot import {spec!r}: {exc}") from exc
    if isinstance(obj, nn.Module):
        return obj
    if callable(obj):
        try:
            instance = obj()
        except Exception as exc:
            raise SystemExit(f"error: {spec!r} could not be called with no arguments: {exc}") from exc
        if isinstance(instance, nn.Module):
            return instance
    raise SystemExit(f"error: {spec!r} is not an nn.Module, an nn.Module subclass, or a factory returning one")
