# Contributing to torchdiagram

Thanks for helping! Bug reports, docs fixes, and features from the [roadmap](README.md#status--roadmap) are all welcome.

## Development setup

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/raimbekovm/torchdiagram
cd torchdiagram
uv sync
```

On Linux, `uv sync` installs CPU-only PyTorch wheels (configured in `pyproject.toml`) — a diagram tool never needs CUDA.

## Checks

CI runs exactly these commands; run them locally before pushing:

```bash
uv run ruff format --check   # formatting
uv run ruff check            # linting
uv run mypy src              # type checking
uv run pytest                # tests
```

## Guidelines

- Add or update tests for any behavior change.
- Keep `torchdiagram.graph` (the IR) free of torch imports — renderers and hand-built graphs must work without tracing.
- New renderers go in `src/torchdiagram/renderers/` and register their file extension in `renderers/__init__.py`.
- Add a line to `CHANGELOG.md` under **Unreleased** for user-visible changes.

## Reporting bugs

The single most useful bug report is a **minimal `nn.Module` that reproduces the problem** plus the command or code you ran. The issue template asks for exactly that.
