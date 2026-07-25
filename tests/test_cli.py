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


def test_cli_falls_back_for_untraceable_model(tmp_path, capsys):
    """A model with data-dependent control flow renders through the torch.export fallback, with a note about it."""
    output = tmp_path / "gated.svg"
    main(["tests.models:GatedNet", "-o", str(output), "--input-shape", "1,3,8,8"])
    assert output.read_text().startswith("<svg")
    assert "specialized on the example input" in capsys.readouterr().err


def test_cli_rejects_untraceable_model_without_input_shape():
    """Without '--input-shape' the fallback cannot run, so an untraceable model exits with a hint to supply one."""
    with pytest.raises(SystemExit, match="cannot trace"):
        main(["tests.models:GatedNet", "-o", "out.svg"])


def test_cli_backend_fx_disables_the_fallback():
    """'--backend fx' pins symbolic tracing, so an untraceable model exits instead of falling back."""
    with pytest.raises(SystemExit, match="cannot trace"):
        main(["tests.models:GatedNet", "-o", "out.svg", "--input-shape", "1,3,8,8", "--backend", "fx"])


def test_cli_backend_export_requires_input_shape():
    """'--backend export' without '--input-shape' exits with a clear error rather than a traceback."""
    with pytest.raises(SystemExit, match="requires an example input"):
        main(["tests.models:TinyCNN", "-o", "out.svg", "--backend", "export"])


def test_cli_rejects_factory_that_needs_arguments():
    """A model class requiring constructor arguments exits with a clear error."""
    with pytest.raises(SystemExit, match="could not be called with no arguments"):
        main(["tests.models:NeedsConstructorArgs", "-o", "out.svg"])


def test_cli_rejects_unsupported_output_extension():
    """An unsupported output extension exits with a clear error."""
    with pytest.raises(SystemExit, match="unsupported output format"):
        main(["tests.models:TinyCNN", "-o", "out.pdf"])


def test_cli_writes_png(tmp_path):
    """A '.png' output path is rasterized rather than written as text."""
    output = tmp_path / "tiny.png"
    main(["tests.models:TinyCNN", "-o", str(output), "--scale", "1"])
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_cli_theme_flag_applies_preset(tmp_path):
    """'--theme dark' renders with the dark preset's palette, not the default one."""
    import torchdiagram as td

    output = tmp_path / "tiny.svg"
    main(["tests.models:TinyCNN", "-o", str(output), "--theme", "dark"])
    content = output.read_text()
    # block_stroke is unique to DARK ("#81a1c1"); block_fill would also appear under DEFAULT.
    assert td.DARK.block_stroke in content
    assert td.DEFAULT.block_stroke not in content


def test_cli_rejects_unknown_theme():
    """An unknown '--theme' value exits with a usage error."""
    with pytest.raises(SystemExit):
        main(["tests.models:TinyCNN", "-o", "out.svg", "--theme", "bogus"])
