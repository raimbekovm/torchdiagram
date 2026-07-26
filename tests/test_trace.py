"""Tests for tracing into the torchdiagram IR, through both the torch.fx and torch.export frontends."""

import pytest
import torch
from torch import nn

import torchdiagram as td
from tests.models import (
    CastRange,
    ChainedSubscript,
    ComposedEncoder,
    CroppedHead,
    DualEncoder,
    FixedRange,
    GatedNet,
    LoopedRange,
    NamedOutput,
    NestedReturn,
    NumberedReuse,
    OutputOnlyTagger,
    PyramidHeads,
    RepeatedBlockStack,
    RepeatedCall,
    ResidualBlock,
    SequenceTagger,
    ShapeMath,
    SiameseTower,
    SlicedTwice,
    StateOnlyTagger,
    SteppedSubscripts,
    SubscriptedLayer,
    TinyCNN,
    TokenPrefix,
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
        (TokenPrefix(), torch.randn(1, 3, 4)),
        (SiameseTower(), torch.randn(1, 4)),
        (CroppedHead(), torch.randn(1, 3, 5)),
        (ChainedSubscript(), torch.randn(3, 4, 4)),
        (SteppedSubscripts(), torch.randn(3, 4, 4)),
        (ShapeMath(), torch.randn(1, 8, 4)),
        (SlicedTwice(), torch.randn(6, 4, 4)),
        (FixedRange(), torch.randn(1, 8, 4)),
        (CastRange(), torch.randn(1, 4)),
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


def test_a_layer_returning_a_tuple_is_one_box_carrying_its_real_shape():
    """`out, _ = self.rnn(x)` is a layer, not a layer plus two boxes of Python syntax."""
    example = torch.randn(1, 5, 4)
    for backend in ("fx", "export"):
        graph = td.trace(SequenceTagger(), example, backend=backend)
        graph.validate()
        assert [node.label for node in graph.nodes] == ["input", "LSTM", "Linear", "output"]
        # (4, 1, 6) is the final hidden state, which is a real tensor the layer produces but not the one that leaves.
        assert next(node for node in graph.nodes if node.label == "LSTM").output_shape == (1, 5, 12)


def test_subscripting_a_layers_tensor_output_stays_on_the_diagram():
    """`self.conv(x)[0]` indexes a tensor; only unpacking a tuple return folds into the layer."""
    graph = td.trace(SubscriptedLayer(), torch.randn(2, 3, 8, 8))
    assert [node.op for node in graph.nodes] == ["input", "conv2d", "index", "output"]


def test_tuple_unpacking_folds_without_an_example_input():
    """With no shapes to propagate, a layer selected from at two indices is still a tuple return."""
    assert [node.label for node in td.trace(SequenceTagger()).nodes] == ["input", "LSTM", "Linear", "output"]


def test_a_parameter_used_in_forward_gets_its_own_box():
    """A class token is a leaf of the data flow with nowhere else to live, so it is drawn rather than dropped."""
    for backend in ("fx", "export"):
        graph = td.trace(TokenPrefix(), torch.randn(1, 3, 4), backend=backend)
        graph.validate()
        incoming = {node.id: [edge.source for edge in graph.edges if edge.target == node.id] for node in graph.nodes}
        assert [node.label for node in graph.nodes if node.op == "parameter"] == ["cls_token", "pos_embed"]
        # An addition of a tensor and a learned embedding has two operands, not one.
        assert sorted(incoming["add"]) == ["cat", "pos_embed"]


def test_a_parameters_consumer_is_not_wired_to_whatever_supplied_a_shape():
    """`cls_token.expand(x.shape[0], ...)` takes its data from the token; the batch size is not a data edge."""
    graph = td.trace(TokenPrefix(), torch.randn(1, 3, 4))
    expand = next(node for node in graph.nodes if node.op == "expand")
    assert [edge.source for edge in graph.edges if edge.target == expand.id] == ["cls_token"]


def test_a_layer_applied_twice_is_marked_as_one_set_of_weights():
    """Two boxes for one layer is the honest picture of the data flow; claiming two sets of weights is not."""
    for backend in ("fx", "export"):
        graph = td.trace(SiameseTower(), torch.randn(1, 4), backend=backend)
        shared = [node for node in graph.nodes if "shared_with" in node.params]
        assert [node.label for node in shared] == ["Linear (encode, call 1)", "Linear (encode, call 2)"]
        assert shared[0].params["shared_with"] == [shared[1].id]


def test_a_mismatched_example_input_is_reported_in_the_users_terms():
    """The model traced fine; the tensor was wrong. The error says which layer rejected it and what it expected."""
    with pytest.raises(ValueError) as failure:
        td.trace(TinyCNN(), torch.randn(1, 3, 28, 28))
    message = str(failure.value)
    assert "example input (1, 3, 28, 28) does not run through this model" in message
    assert "layer 'conv' (Conv2d)" in message
    assert "1 channels, but got 3 channels" in message


def test_a_mismatched_example_input_does_not_print_a_stack(capsys):
    """Shape propagation prints the stack itself from inside torch; the error line is the whole output."""
    with pytest.raises(ValueError):
        td.trace(TinyCNN(), torch.randn(1, 3, 28, 28))
    assert "Traceback" not in capsys.readouterr().err


def test_an_output_box_does_not_collide_with_a_submodule_called_output():
    """The output box is named after the return it carries, and a model may hold a submodule of that name."""
    for backend in ("fx", "export"):
        graph = td.trace(NamedOutput(), torch.randn(1, 4), backend=backend)
        graph.validate()  # would raise "duplicate node ids: ['output']"
        assert [node.label for node in graph.nodes if node.op == "output"] == ["output"]


def test_a_nested_return_structure_is_flattened():
    """`return h, (a, b)` returns three tensors; taking the nested tuple at face value drops two of them."""
    for backend in ("fx", "export"):
        graph = td.trace(NestedReturn(), torch.randn(1, 4), backend=backend)
        graph.validate()
        outputs = [node for node in graph.nodes if node.op == "output"]
        assert [node.label for node in outputs] == ["output[0]", "output[1]", "output[2]"]
        # Every head has somewhere to go; a dropped return leaves its producer dangling.
        assert all(any(edge.source == node.id for edge in graph.edges) for node in graph.nodes if not node.is_io)


def test_a_layer_applied_twice_in_a_row_is_two_boxes_in_both_frontends():
    """Nothing separates the two calls, so the run of ATen ops is unbroken; the call key is what tells them apart."""
    for backend in ("fx", "export"):
        labels = [node.label for node in td.trace(RepeatedCall(), torch.randn(1, 4), backend=backend).nodes]
        assert labels == ["input", "Linear (fc, call 1)", "Linear (fc, call 2)", "output"]


def test_nested_tuple_unpacking_folds_into_the_layer():
    """`_, (h, c) = self.rnn(x)` selects out of the state tuple, one level below what the first fold handled."""
    for backend in ("fx", "export"):
        graph = td.trace(StateOnlyTagger(), torch.randn(1, 5, 4), backend=backend)
        graph.validate()
        # Both selections out of the layer are folded away; the one left is `hidden[-1]`, a subscript of a tensor.
        assert [node.label for node in graph.nodes] == ["input", "LSTM", "index", "Linear", "output"]


def test_tuple_unpacking_folds_the_same_way_with_and_without_shapes():
    """--input-shape is documented as adding annotations; it must not also change how many boxes there are."""
    with_shapes = [node.label for node in td.trace(OutputOnlyTagger(), torch.randn(1, 5, 4)).nodes]
    without = [node.label for node in td.trace(OutputOnlyTagger()).nodes]
    assert with_shapes == without == ["input", "LSTM", "Linear", "output"]


def test_a_reused_container_at_a_numeric_path_gets_the_same_ids_from_both_frontends():
    """Fx renumbers off its own node names and strips a trailing `_<digits>`; ids have to come from the path."""
    example = torch.randn(1, 4)
    by_fx = [node.id for node in td.trace(NumberedReuse(), example, backend="fx").nodes]
    by_export = [node.id for node in td.trace(NumberedReuse(), example, backend="export").nodes]
    assert by_fx == by_export


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


def test_chained_subscripts_on_one_line_draw_one_box_in_both_frontends():
    """`x[0][1]` is one indexing step to a reader; fx splits it in two and export cannot, so both collapse it."""
    example = torch.randn(3, 4, 4)
    for backend in ("fx", "export"):
        ops = [node.op for node in td.trace(ChainedSubscript(), example, backend=backend).nodes]
        assert ops.count("index") == 1, backend


def test_subscripts_a_line_apart_keep_a_box_each_in_both_frontends():
    """The collapse follows the source line, so two subscripts the model wrote as two steps stay two boxes."""
    example = torch.randn(3, 4, 4)
    for backend in ("fx", "export"):
        ops = [node.op for node in td.trace(SteppedSubscripts(), example, backend=backend).nodes]
        assert ops.count("index") == 2, backend


def test_a_subscript_feeding_two_readers_is_not_collapsed():
    """A result used more than once is a step of its own: collapsing it would drop an edge the model really has."""

    class Fanout(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 2)

        def forward(self, x):
            group = x[0]
            return self.proj(group[1]) + self.proj(group[2])

    graph = td.trace(Fanout(), torch.randn(3, 4, 4), backend="fx")
    assert [node.op for node in graph.nodes].count("index") == 3


def test_a_range_sized_by_the_input_depends_on_it_in_both_frontends():
    """Export folds `x.shape[1]` into a constant, so the dependency is recovered rather than read off the graph."""
    example = torch.randn(1, 8, 4)
    for backend in ("fx", "export"):
        graph = td.trace(ShapeMath(), example, backend=backend)
        source = next(node for node in graph.nodes if node.op == "input")
        arange = next(node for node in graph.nodes if node.op == "arange")
        assert td.Edge(source.id, arange.id) in graph.edges, backend


def test_a_range_of_a_fixed_size_is_left_unconnected_in_both_frontends():
    """The recovery reports what the model reads a size from, and a hardcoded length reads nothing."""
    for backend in ("fx", "export"):
        graph = td.trace(FixedRange(), torch.randn(1, 8, 4), backend=backend)
        arange = next(node for node in graph.nodes if node.op == "arange")
        assert [edge for edge in graph.edges if edge.target == arange.id] == []


def test_a_tensor_built_from_nothing_is_an_operation_in_both_frontends():
    """A range with no traced input is still something the model does, not a constant that was always there.

    fx has no proxy to trace such a call through, so left alone it runs the call and keeps the result as a tensor
    attribute, drawing a buffer box under a name torch invented. Holding the factory back makes it a node again.
    """
    for backend in ("fx", "export"):
        graph = td.trace(FixedRange(), torch.randn(1, 8, 4), backend=backend)
        assert [node.label for node in graph.nodes if node.op not in ("input", "output")] == [
            "arange",
            "Embedding",
            "add",
            "Linear",
        ]


def test_a_dtype_cast_draws_the_same_box_in_both_frontends():
    """`x.float()` is a method of its own to fx and one more overload of `to` to export; the diagram says `to`."""
    for backend in ("fx", "export"):
        graph = td.trace(CastRange(), torch.randn(1, 4), backend=backend)
        assert [node.label for node in graph.nodes if node.op == "to"] == ["to"]


def test_a_range_the_model_iterates_still_traces():
    """A factory held back as a node breaks a model that wants the tensor itself, so that trace is retried without it.

    Iterating a range is the plainest case: there is no symbolic stand-in for one, and a model that traced before the
    factories were held back has to keep tracing.
    """
    graph = td.trace(LoopedRange(), torch.randn(1, 4), backend="fx")
    graph.validate()
    assert [node.label for node in graph.nodes if node.op == "add"] == ["add", "add"]


def test_one_subscript_in_a_submodule_applied_twice_stays_two_boxes():
    """Two calls of a slicing submodule are two operations, though both were written on the submodule's one line.

    The source line alone cannot tell them apart, which is why the fold keys on the whole chain of model frames: the
    call site differs even when the subscript does not. Collapsing them would attribute the second call's output to the
    first submodule and leave the second doing nothing at all.
    """
    example = torch.randn(6, 4, 4)
    for backend in ("fx", "export"):
        graph = td.trace(SlicedTwice(), example, backend=backend)
        index = [node for node in graph.nodes if node.op == "index"]
        assert [(node.scope, node.output_shape) for node in index] == [
            ("first", (5, 4, 4)),
            ("second", (4, 4, 4)),
        ], backend
