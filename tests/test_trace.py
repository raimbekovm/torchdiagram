"""Tests for torch.fx tracing into the torchdiagram IR."""

import torch

import torchdiagram as td
from tests.models import RepeatedBlockStack, ResidualBlock, TinyCNN


def test_trace_produces_valid_graph_with_io_nodes():
    """A traced model yields a valid graph bracketed by input and output nodes."""
    graph = td.trace(TinyCNN())
    graph.validate()
    ops = [node.op for node in graph.nodes]
    assert ops[0] == "input"
    assert ops[-1] == "output"
    assert "conv2d" in ops
    assert "linear" in ops


def test_trace_records_layer_config():
    """Layer hyperparameters (extra_repr) are captured in node params."""
    graph = td.trace(TinyCNN())
    conv = next(node for node in graph.nodes if node.op == "conv2d")
    assert "kernel_size" in conv.params["config"]


def test_shapes_annotated_when_example_input_given():
    """Passing an example input shape-propagates output shapes onto nodes."""
    graph = td.trace(TinyCNN(), torch.randn(1, 1, 28, 28))
    conv = next(node for node in graph.nodes if node.op == "conv2d")
    assert conv.output_shape == (1, 8, 28, 28)


def test_shapes_absent_without_example_input():
    """Without an example input, no node carries a shape annotation."""
    graph = td.trace(TinyCNN())
    assert all(node.output_shape is None for node in graph.nodes)


def test_residual_add_has_two_incoming_edges():
    """A skip connection produces an add node with two incoming edges."""
    graph = td.trace(ResidualBlock())
    graph.validate()
    add = next(node for node in graph.nodes if node.op == "add")
    incoming = [edge for edge in graph.edges if edge.target == add.id]
    assert len(incoming) == 2


def test_graph_name_defaults_to_class_name():
    """Graph name defaults to the model class name and honors an override."""
    assert td.trace(TinyCNN()).name == "TinyCNN"
    assert td.trace(TinyCNN(), name="custom").name == "custom"


def test_trace_records_scope_for_nested_submodules():
    """Nodes traced from inside a custom submodule carry its dotted path and class name."""
    graph = td.trace(RepeatedBlockStack(2))
    nested = next(node for node in graph.nodes if node.id == "layer1_0_conv1")
    assert nested.scope == "layer1.0"
    assert nested.scope_class == "BasicBlock"
    root = next(node for node in graph.nodes if node.id == "stem")
    assert root.scope is None
    assert root.scope_class is None
