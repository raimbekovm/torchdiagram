"""Tests for the torchdiagram command-line interface."""

import pytest

from torchdiagram.cli import main


def test_cli_end_to_end(tmp_path):
    """The CLI traces a model by import path and writes the rendered file."""
    output = tmp_path / "tiny.svg"
    main(["tests.models:TinyCNN", "-o", str(output), "--input-shape", "1,1,28,28"])
    content = output.read_text()
    assert content.startswith("<svg")
    assert "×28×28" in content  # shape annotations flowed through


def test_cli_rejects_bad_import_path():
    """A non-importable model spec exits with a clear error."""
    with pytest.raises(SystemExit, match="cannot import"):
        main(["tests.models:DoesNotExist", "-o", "out.svg"])


def test_cli_rejects_spec_without_attr():
    """A spec without ':attr' exits with a usage error."""
    with pytest.raises(SystemExit, match=r"expected 'package\.module:attr'"):
        main(["tests.models", "-o", "out.svg"])
