"""Command-line interface: ``torchdiagram my_package.models:MyNet -o my_net.svg``."""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import torch
from torch import nn

from torchdiagram.renderers import render
from torchdiagram.trace import trace
from torchdiagram.transforms import aggregate_blocks


def main(argv: list[str] | None = None) -> None:
    """Run the ``torchdiagram`` command-line entry point.

    Args:
        argv: Command-line arguments to parse, excluding the program name. Defaults to ``sys.argv[1:]`` when ``None``.

    Raises:
        SystemExit: If the model cannot be loaded, ``--input-shape`` is invalid, tracing fails (e.g. due to
            data-dependent control flow), or the output path has an unsupported extension.
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
    parser.add_argument("-o", "--output", required=True, help="Output file (.svg, .tex, .tikz).")
    parser.add_argument(
        "--input-shape",
        help="Comma-separated input shape, e.g. '1,3,224,224'; enables shape annotations on every node.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Collapse runs of repeated, structurally identical blocks (e.g. ResNet layers) into one node.",
    )
    parser.add_argument(
        "--min-repeats",
        type=int,
        default=None,
        help="Minimum run length required to collapse with --aggregate (default: 2).",
    )
    args = parser.parse_args(argv)

    model = _load_model(args.model)
    example = _build_example_input(args.input_shape)
    try:
        graph = trace(model, example)
    except torch.fx.proxy.TraceError as exc:
        raise SystemExit(f"error: cannot trace {args.model!r}: {exc}") from exc
    if args.aggregate:
        min_repeats_kwargs = {} if args.min_repeats is None else {"min_repeats": args.min_repeats}
        graph = aggregate_blocks(graph, **min_repeats_kwargs)
    try:
        path = render(graph, args.output)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"wrote {path}")


def _build_example_input(input_shape: str | None) -> torch.Tensor | None:
    """Parse ``--input-shape`` into an example tensor for shape propagation.

    Args:
        input_shape: Comma-separated dimensions, e.g. ``"1,3,224,224"``, or ``None`` to skip shape annotation.

    Returns:
        A randomly initialized tensor of the requested shape, or ``None`` if ``input_shape`` is ``None``.

    Raises:
        SystemExit: If ``input_shape`` is not a comma-separated list of integers, or any dimension is invalid.
    """
    if not input_shape:
        return None
    try:
        shape = tuple(int(dim.strip()) for dim in input_shape.split(","))
        return torch.randn(*shape)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: invalid --input-shape {input_shape!r}: {exc}") from exc


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
