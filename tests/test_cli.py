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


def test_cli_aggregate_flag_collapses_repeated_layers(tmp_path):
    """'--aggregate' collapses a repeated block stack into a single labeled node."""
    output = tmp_path / "stack.svg"
    main(["tests.models:RepeatedBlockStack", "-o", str(output), "--aggregate"])
    content = output.read_text()
    assert "BasicBlock" in content
    assert "×4" in content


def test_cli_omits_aggregation_by_default(tmp_path):
    """Without '--aggregate', the output shows individual layers, not a collapsed block."""
    output = tmp_path / "stack.svg"
    main(["tests.models:RepeatedBlockStack", "-o", str(output)])
    content = output.read_text()
    assert "×4" not in content


def test_cli_rejects_bad_import_path():
    """A non-importable model spec exits with a clear error."""
    with pytest.raises(SystemExit, match="cannot import"):
        main(["tests.models:DoesNotExist", "-o", "out.svg"])


def test_cli_rejects_spec_without_attr():
    """A spec without ':attr' exits with a usage error."""
    with pytest.raises(SystemExit, match=r"expected 'package\.module:attr'"):
        main(["tests.models", "-o", "out.svg"])


def test_cli_rejects_bad_input_shape():
    """A malformed '--input-shape' exits with a clear error instead of a raw traceback."""
    with pytest.raises(SystemExit, match="invalid --input-shape"):
        main(["tests.models:TinyCNN", "-o", "out.svg", "--input-shape", "1,x,3"])


def test_cli_rejects_untraceable_model():
    """A model with data-dependent control flow exits with a clear tracing error."""
    with pytest.raises(SystemExit, match="cannot trace"):
        main(["tests.models:BranchingModel", "-o", "out.svg", "--input-shape", "1,4"])


def test_cli_rejects_factory_that_needs_arguments():
    """A model class requiring constructor arguments exits with a clear error."""
    with pytest.raises(SystemExit, match="could not be called with no arguments"):
        main(["tests.models:NeedsConstructorArgs", "-o", "out.svg"])


def test_cli_rejects_unsupported_output_extension():
    """An unsupported output extension exits with a clear error."""
    with pytest.raises(SystemExit, match="unsupported output format"):
        main(["tests.models:TinyCNN", "-o", "out.png"])
