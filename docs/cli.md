# CLI reference

The `torchdiagram` command traces a model given by an import path and writes a rendered diagram.

## Synopsis

```bash
torchdiagram MODEL -o OUTPUT [--input-shape SHAPE] [--aggregate] [--min-repeats N] [--theme NAME]
```

## Arguments

| Argument         | Required | Description                                                                                                                                                      |
| ---------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MODEL`          | yes      | Import path to the model in the form `package.module:attr`.                                                                                                      |
| `-o`, `--output` | yes      | Output file. The format is selected by extension: `.svg`, `.tex`, or `.tikz`.                                                                                    |
| `--input-shape`  | no       | Comma-separated input shape, e.g. `1,3,224,224`. Enables output-shape annotations on every node.                                                                 |
| `--aggregate`    | no       | Collapse scoped blocks — repeated (e.g. ResNet layers) and singleton (e.g. a transformer's attention/MLP sub-block) — into single labeled nodes. Off by default. |
| `--min-repeats`  | no       | Minimum run length required to merge multiple blocks into one badged node with `--aggregate` (default: `2`); shorter runs still collapse individually.           |
| `--theme`        | no       | Color theme: `default` (soft blue), `mono` (grayscale, for print), or `dark` (for dark backgrounds). Default: `default`.                                         |

## Model specification

The `MODEL` argument names a module and an attribute inside it, separated by a colon. The attribute may be:

- an `nn.Module` **instance** (`my_models:net`),
- an `nn.Module` **subclass**, which is instantiated with no arguments (`my_models:ResidualBlock`),
- a **zero-argument factory function** returning an `nn.Module` (`my_models:build_model`).

The current working directory is added to the import path, so a model defined in `./my_models.py` is addressable as `my_models:...` without installing anything. Installed packages work the same way (`torchvision.models:resnet18`).

Models that require constructor arguments cannot be instantiated by the CLI; wrap them in a zero-argument factory function, or use the [Python API](api.md).

## Input shape

When `--input-shape` is given, the CLI creates a random float tensor of that shape with `torch.randn` and uses it for shape propagation. For models that expect non-float inputs (for example, token indices for an embedding layer), use the Python API and pass an appropriate `example_input` to `trace()`.

## Examples

```bash
# Structure-only SVG from a class in the current directory
torchdiagram my_models:ResidualBlock -o block.svg

# Shape-annotated diagram
torchdiagram my_models:ResidualBlock -o block.svg --input-shape 1,64,56,56

# Standalone TikZ document from an installed package
torchdiagram torchvision.models:resnet18 -o resnet18.tex --input-shape 1,3,224,224

# Collapse repeated residual blocks into single labeled nodes
torchdiagram torchvision.models:resnet50 -o resnet50.svg --input-shape 1,3,224,224 --aggregate

# Grayscale figure for a print paper
torchdiagram torchvision.models:resnet18 -o resnet18.tex --input-shape 1,3,224,224 --theme mono
```

## Exit behavior

On success, the command prints the path of the written file and exits with status 0. Import failures, invalid model specifications, and unsupported output extensions terminate with a non-zero status and an error message on stderr.
