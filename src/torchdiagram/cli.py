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


def main(argv: list[str] | None = None) -> None:
    """Run the ``torchdiagram`` command-line entry point.

    Args:
        argv: Command-line arguments to parse, excluding the program name. Defaults to ``sys.argv[1:]`` when ``None``.
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
    args = parser.parse_args(argv)

    model = _load_model(args.model)
    example = None
    if args.input_shape:
        shape = tuple(int(dim.strip()) for dim in args.input_shape.split(","))
        example = torch.randn(*shape)
    path = render(trace(model, example), args.output)
    print(f"wrote {path}")


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
        instance = obj()
        if isinstance(instance, nn.Module):
            return instance
    raise SystemExit(f"error: {spec!r} is not an nn.Module, an nn.Module subclass, or a factory returning one")
