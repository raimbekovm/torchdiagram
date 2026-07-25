"""Tests for tracing into the torchdiagram IR, through both the torch.fx and torch.export frontends."""

import pytest
import torch
from torch import nn

import torchdiagram as td
from tests.models import (
    ComposedEncoder,
    CroppedHead,
    DualEncoder,
    GatedNet,
    PyramidHeads,
    RepeatedBlockStack,
    ResidualBlock,
    ShapeMath,
    TinyCNN,
    UnusedBranch,
)


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


def test_fallback_traces_data_dependent_control_flow():
    """A model fx cannot trace still yields a valid graph, via the torch.export frontend."""
    with pytest.warns(UserWarning, match="specialized on the example input"):
        graph = td.trace(GatedNet(), torch.randn(1, 3, 8, 8))
    graph.validate()
    labels = [node.label for node in graph.nodes]
    assert labels[0] == "input"
    assert labels[-1] == "output"
    assert "Conv2d" in labels
    assert "Linear" in labels


def test_fallback_keeps_the_branch_the_example_input_takes():
    """The fallback specializes: only the executed branch reaches the diagram, and guard plumbing does not."""
    model = GatedNet()
    with pytest.warns(UserWarning):
        positive = td.trace(model, torch.full((1, 3, 8, 8), 1.0))
    with pytest.warns(UserWarning):
        negative = td.trace(model, torch.full((1, 3, 8, 8), -1.0))
    # GatedNet negates only when the input sums to zero or less, so each input traces to a different graph.
    assert "neg" not in [node.op for node in positive.nodes]
    assert "neg" in [node.op for node in negative.nodes]
    # Neither the guard machinery nor the branch condition feeding it ('x.sum() > 0' lowers to sum/gt/ne) belongs in
    # an architecture diagram; only the ops that carry the model's output should survive.
    assert [node.op for node in positive.nodes] == [
        "input",
        "conv2d",
        "relu",
        "adaptiveavgpool2d",
        "flatten",
        "linear",
        "output",
    ]


def test_shape_plumbing_is_not_drawn():
    """Reading a tensor's shape is not an architecture step, so none of it reaches the diagram."""
    graph = td.trace(ShapeMath(), torch.randn(1, 8, 4))
    ops = [node.op for node in graph.nodes]
    assert ops == ["input", "arange", "embedding", "add", "linear", "output"]


def test_shape_plumbing_keeps_the_flow_connected():
    """Dropping the shape lookup must not orphan what consumed it: the range still depends on the input."""
    graph = td.trace(ShapeMath(), torch.randn(1, 8, 4))
    graph.validate()
    source = next(node for node in graph.nodes if node.op == "input")
    arange = next(node for node in graph.nodes if node.op == "arange")
    assert td.Edge(source.id, arange.id) in graph.edges


def test_duplicate_layer_labels_are_qualified_by_attribute():
    """Two layers of one class in one scope are named apart; a layer alone under its label keeps the bare class name."""
    labels = [node.label for node in td.trace(ResidualBlock()).nodes]
    assert "Conv2d (conv1)" in labels
    assert "Conv2d (conv2)" in labels
    # TinyCNN holds one convolution and one linear layer, so neither has anything to be told apart from.
    assert "Conv2d" in [node.label for node in td.trace(TinyCNN()).nodes]


def test_backend_fx_does_not_fall_back():
    """'backend="fx"' pins symbolic tracing, so an untraceable model still raises."""
    with pytest.raises(torch.fx.proxy.TraceError):
        td.trace(GatedNet(), torch.randn(1, 3, 8, 8), backend="fx")


def test_fallback_needs_an_example_input():
    """Without an example input the fallback cannot run, and the error says so."""
    with pytest.raises(torch.fx.proxy.TraceError, match="Pass an example input"):
        td.trace(GatedNet())


def test_export_backend_requires_an_example_input():
    """'backend="export"' is rejected without an example input, since torch.export traces with real inputs."""
    with pytest.raises(ValueError, match="requires an example input"):
        td.trace(TinyCNN(), backend="export")


def test_unknown_backend_is_rejected():
    """An unrecognized backend name fails fast rather than silently picking a frontend."""
    with pytest.raises(ValueError, match="unknown backend"):
        td.trace(TinyCNN(), torch.randn(1, 1, 28, 28), backend="onnx")


def test_several_example_inputs_reach_a_multi_input_model():
    """A model taking two tensors takes a tuple of them, and every node still gets a shape."""
    graph = td.trace(DualEncoder(), (torch.randn(1, 4), torch.randn(1, 8)))
    graph.validate()
    inputs = [node for node in graph.nodes if node.op == "input"]
    assert [node.output_shape for node in inputs] == [(1, 4), (1, 8)]
    assert all(node.output_shape is not None for node in graph.nodes)


def test_each_returned_tensor_gets_its_own_output_node():
    """A model returning several tensors draws one output box per tensor, each with its own shape."""
    graph = td.trace(PyramidHeads(), torch.randn(1, 3, 8, 8))
    graph.validate()
    outputs = [node for node in graph.nodes if node.op == "output"]
    assert [node.label for node in outputs] == ["output[0]", "output[1]"]
    assert [node.output_shape for node in outputs] == [(1, 2, 8, 8), (1, 2, 4, 4)]
    # Three arrows into one box would say the levels merge; each prediction leaves on its own.
    assert all(len([edge for edge in graph.edges if edge.target == node.id]) == 1 for node in outputs)


def test_single_return_keeps_one_plain_output_node():
    """A model returning one tensor is unchanged: one box labeled 'output', not 'output[0]'."""
    outputs = [node for node in td.trace(TinyCNN()).nodes if node.op == "output"]
    assert [node.label for node in outputs] == ["output"]


@pytest.mark.parametrize(
    ("model", "example"),
    [
        (TinyCNN(), torch.randn(1, 1, 28, 28)),
        (ResidualBlock(), torch.randn(1, 4, 8, 8)),
        (RepeatedBlockStack(2), torch.randn(1, 3, 8, 8)),
        (UnusedBranch(), torch.randn(1, 4)),
        (DualEncoder(), (torch.randn(1, 4), torch.randn(1, 8))),
        (PyramidHeads(), torch.randn(1, 3, 8, 8)),
        (ComposedEncoder(), torch.randn(1, 5, 8)),
        (CroppedHead(), torch.randn(1, 3, 5)),
    ],
)
def test_both_frontends_emit_the_same_ir(model, example):
    """On a model both frontends can trace, they produce identical graphs — the contract renderers rely on."""
    by_fx = td.trace(model, example, backend="fx")
    by_export = td.trace(model, example, backend="export")

    def fields(graph):
        return [
            (node.id, node.op, node.label, node.params, node.output_shape, node.scope, node.scope_class)
            for node in graph.nodes
        ]

    assert fields(by_fx) == fields(by_export)
    assert sorted((e.source, e.target) for e in by_fx.edges) == sorted((e.source, e.target) for e in by_export.edges)


def test_torch_composite_layer_draws_one_box_in_both_frontends():
    """A torch.nn composite is a leaf to both frontends: one box, not a box plus its nine children."""
    example = torch.randn(1, 5, 8)
    for backend in ("fx", "export"):
        labels = [node.label for node in td.trace(ComposedEncoder(), example, backend=backend).nodes]
        assert sum(label.startswith("TransformerEncoderLayer") for label in labels) == 2
        # The children of an encoder layer are only ever drawn if the descent didn't stop at the layer itself.
        assert "MultiheadAttention" not in labels


def test_subscripting_a_tensor_gets_one_label_from_both_frontends():
    """`x[:, 0, :-1]` is one operation to a reader, whichever way the tracer lowered it."""
    example = torch.randn(1, 3, 5)
    for backend in ("fx", "export"):
        ops = [node.op for node in td.trace(CroppedHead(), example, backend=backend).nodes]
        assert "index" in ops
        assert "getitem" not in ops
        assert "slice" not in ops


def test_a_leaf_layer_traced_on_its_own_keeps_its_class_and_config():
    """Pointing the tool at a bare layer draws that layer, not the functional ops fx descends into."""
    graph = td.trace(nn.Linear(4, 6), torch.randn(1, 4))
    graph.validate()
    layer = graph.nodes[1]
    assert layer.label == "Linear"
    assert "in_features=4" in layer.params["config"]
    assert [node.output_shape for node in graph.nodes] == [(1, 4), (1, 6), (1, 6)]


def test_a_leaf_layer_traced_on_its_own_needs_no_example_input():
    """The layer box is drawn from the module itself, so it works without shapes, like every other model."""
    labels = [node.label for node in td.trace(nn.Conv2d(1, 8, 3)).nodes]
    assert labels == ["input", "Conv2d", "output"]


def test_export_graphs_still_aggregate():
    """Scope survives the export frontend, so block aggregation works on its graphs unchanged."""
    graph = td.trace(RepeatedBlockStack(4), torch.randn(1, 3, 8, 8), backend="export")
    collapsed = td.aggregate_blocks(graph)
    collapsed.validate()
    assert "BasicBlock ×4" in [node.label for node in collapsed.nodes]


def test_trace_records_scope_for_nested_submodules():
    """Nodes traced from inside a custom submodule carry its dotted path and class name."""
    graph = td.trace(RepeatedBlockStack(2))
    nested = next(node for node in graph.nodes if node.id == "layer1_0_conv1")
    assert nested.scope == "layer1.0"
    assert nested.scope_class == "BasicBlock"
    root = next(node for node in graph.nodes if node.id == "stem")
    assert root.scope is None
    assert root.scope_class is None
