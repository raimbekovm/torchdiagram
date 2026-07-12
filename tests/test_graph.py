"""Tests for the Graph/Node/Edge intermediate representation."""

import pytest

import torchdiagram as td


def test_validate_accepts_a_consistent_graph():
    """A graph whose edges only reference existing node ids passes validation."""
    graph = td.Graph(nodes=[td.Node(id="a", op="input", label="a"), td.Node(id="b", op="output", label="b")])
    graph.edges.append(td.Edge("a", "b"))
    graph.validate()  # must not raise


def test_validate_rejects_duplicate_node_ids():
    """Two nodes sharing an id fail validation with the offending id named."""
    graph = td.Graph(nodes=[td.Node(id="a", op="x", label="x"), td.Node(id="a", op="y", label="y")])
    with pytest.raises(ValueError, match=r"duplicate node ids: \['a'\]"):
        graph.validate()


def test_validate_rejects_edge_to_unknown_node():
    """An edge referencing a node id absent from ``nodes`` fails validation."""
    graph = td.Graph(nodes=[td.Node(id="a", op="x", label="x")], edges=[td.Edge("a", "ghost")])
    with pytest.raises(ValueError, match="unknown node id: 'ghost'"):
        graph.validate()


def test_node_is_io_true_for_graph_boundaries():
    """Input and output nodes report is_io as True."""
    assert td.Node(id="a", op="input", label="input").is_io
    assert td.Node(id="b", op="output", label="output").is_io


def test_node_is_io_false_for_regular_ops():
    """Layer and functional-op nodes report is_io as False."""
    assert not td.Node(id="c", op="conv2d", label="Conv2d").is_io
    assert not td.Node(id="d", op="add", label="add").is_io
