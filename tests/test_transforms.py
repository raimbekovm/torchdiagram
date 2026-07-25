"""Tests for the block-aggregation IR-to-IR transform."""

import torch

import torchdiagram as td
from tests.models import (
    ClassifierHead,
    ComposedEncoder,
    DelegatingStack,
    MixedDepthStack,
    NonUniformBlockStack,
    RepeatedBlockStack,
    ThreeLevelNet,
    TiedStack,
    TinyCNN,
    TransformerStack,
    WideningStack,
)
from torchdiagram.transforms import aggregate_blocks


def _block_graph(num_groups: int, scope_class: str = "Block") -> td.Graph:
    """Hand-build a graph with ``num_groups`` identical two-node scoped groups in a chain."""
    nodes = [td.Node(id="in", op="input", label="input")]
    edges = []
    previous = "in"
    for i in range(num_groups):
        a, b = f"g{i}a", f"g{i}b"
        nodes.append(td.Node(id=a, op="conv2d", label="Conv2d", scope=f"layer.{i}", scope_class=scope_class))
        nodes.append(td.Node(id=b, op="relu", label="relu", scope=f"layer.{i}", scope_class=scope_class))
        edges.append(td.Edge(previous, a))
        edges.append(td.Edge(a, b))
        previous = b
    nodes.append(td.Node(id="out", op="output", label="output"))
    edges.append(td.Edge(previous, "out"))
    return td.Graph(nodes=nodes, edges=edges)


def test_aggregate_collapses_direct_construction():
    """Three matching scoped groups collapse into one synthetic block node."""
    graph = _block_graph(3)
    result = aggregate_blocks(graph)
    ops = [node.op for node in result.nodes]
    assert ops == ["input", "block", "output"]
    block = result.nodes[1]
    assert block.label == "Block ×3"
    assert block.params == {"repeats": 3, "block_class": "Block", "ops_per_repeat": 2}


def test_aggregate_collapses_repeated_blocks_from_traced_model():
    """A traced stack of 4 identical BasicBlocks collapses to a single block node."""
    graph = td.trace(RepeatedBlockStack(4))
    result = aggregate_blocks(graph)
    blocks = [node for node in result.nodes if node.op == "block"]
    assert len(blocks) == 1
    assert blocks[0].label == "BasicBlock ×4"
    assert blocks[0].params["repeats"] == 4


def test_aggregate_preserves_output_shape_on_synthetic_node():
    """The synthetic node's output shape matches the traced shape of the last repeated node."""
    graph = td.trace(RepeatedBlockStack(4), torch.randn(1, 3, 8, 8))
    last_member_shape = next(node for node in graph.nodes if node.id.startswith("relu_7")).output_shape
    result = aggregate_blocks(graph)
    block = next(node for node in result.nodes if node.op == "block")
    assert block.output_shape == last_member_shape


def test_aggregate_is_pure():
    """The input graph's nodes and edges are unchanged after aggregation."""
    graph = td.trace(RepeatedBlockStack(4))
    node_ids_before = [node.id for node in graph.nodes]
    edges_before = [(edge.source, edge.target) for edge in graph.edges]
    aggregate_blocks(graph)
    assert [node.id for node in graph.nodes] == node_ids_before
    assert [(edge.source, edge.target) for edge in graph.edges] == edges_before


def test_aggregate_respects_min_repeats():
    """Below min_repeats, matching groups still collapse individually — they just don't merge into one ×N node."""
    graph = td.trace(RepeatedBlockStack(2))
    unmerged = aggregate_blocks(graph, min_repeats=3)
    blocks = [node for node in unmerged.nodes if node.op == "block"]
    assert len(blocks) == 2
    assert all(block.label == "BasicBlock" for block in blocks)
    merged = aggregate_blocks(graph, min_repeats=2)
    merged_blocks = [node for node in merged.nodes if node.op == "block"]
    assert len(merged_blocks) == 1
    assert merged_blocks[0].label == "BasicBlock ×2"


def test_aggregate_leaves_non_repeated_graph_unchanged():
    """A model with no scoped repetition passes through with identical node and edge counts."""
    graph = td.trace(TinyCNN())
    result = aggregate_blocks(graph)
    assert len(result.nodes) == len(graph.nodes)
    assert len(result.edges) == len(graph.edges)


def test_aggregate_partial_match_collapses_only_uniform_suffix():
    """A stage's non-uniform first block collapses to its own node; the uniform rest merges into one ×N node."""
    graph = td.trace(NonUniformBlockStack(3))
    result = aggregate_blocks(graph)
    ops = [node.op for node in result.nodes]
    assert ops.count("block") == 2
    blocks = [node for node in result.nodes if node.op == "block"]
    singleton = next(block for block in blocks if block.params["repeats"] == 1)
    merged = next(block for block in blocks if block.params["repeats"] > 1)
    assert singleton.params["block_class"] == "_DownsampleBlock"
    assert merged.label == "BasicBlock ×2"


def test_aggregate_result_passes_validate():
    """The transform's output is internally consistent."""
    graph = td.trace(RepeatedBlockStack(4))
    aggregate_blocks(graph).validate()  # must not raise


def test_aggregate_collapses_nested_transformer_blocks():
    """Attention/MLP singleton-collapse first; the now-uniform TransformerBlock scope then merges across repeats."""
    graph = td.trace(TransformerStack(4), torch.randn(1, 4, 32))
    result = aggregate_blocks(graph)
    ops = [node.op for node in result.nodes]
    assert ops == ["input", "block", "output"]
    block = result.nodes[1]
    assert block.label == "TransformerBlock ×4"
    assert block.params["repeats"] == 4


def test_aggregate_does_not_merge_stages_of_different_depth():
    """Two stages whose collapsed contents differ stay separate, however alike they look once collapsed.

    Both stages are a Sequential of convolutions followed by a pool, so once the Sequential becomes one node each stage
    reads as the same two-node sequence. Only what the Sequential collapsed tells them apart, and a merge here would
    badge a two-convolution stage and a three-convolution one as the same block repeated.
    """
    graph = td.trace(MixedDepthStack(), torch.randn(1, 4, 8, 8))
    result = aggregate_blocks(graph)
    blocks = [node for node in result.nodes if node.op == "block"]
    assert len(blocks) == 2
    assert all("×" not in block.label for block in blocks)


def test_aggregate_does_not_merge_stages_of_different_width():
    """Identically structured stages running different channel counts are not repeats of each other."""
    graph = td.trace(WideningStack(), torch.randn(1, 3, 8, 8))
    result = aggregate_blocks(graph)
    blocks = [node for node in result.nodes if node.op == "block"]
    assert len(blocks) == 2
    assert all(block.params["repeats"] == 1 for block in blocks)


def test_aggregate_names_container_blocks_after_their_attribute():
    """A collapsed ``nn.Sequential`` is labeled with the attribute holding it, not with the container's class name."""
    graph = td.trace(ClassifierHead(), torch.randn(1, 3, 8, 8))
    result = aggregate_blocks(graph)
    labels = [node.label for node in result.nodes]
    assert "classifier" in labels
    assert "Sequential" not in labels


def test_aggregate_does_not_merge_blocks_with_different_inner_content():
    """Two groups with the same op sequence but a different collapsed block inside must not be merged together."""
    nodes = [
        td.Node(id="in", op="input", label="input"),
        td.Node(id="s0_ln", op="layernorm", label="LayerNorm", scope="stage.0", scope_class="Stage"),
        td.Node(
            id="s0_blk",
            op="block",
            label="Attention",
            params={"block_class": "Attention"},
            scope="stage.0",
            scope_class="Stage",
        ),
        td.Node(id="s0_add", op="add", label="add", scope="stage.0", scope_class="Stage"),
        td.Node(id="s1_ln", op="layernorm", label="LayerNorm", scope="stage.1", scope_class="Stage"),
        td.Node(
            id="s1_blk",
            op="block",
            label="MLP",
            params={"block_class": "MLP"},
            scope="stage.1",
            scope_class="Stage",
        ),
        td.Node(id="s1_add", op="add", label="add", scope="stage.1", scope_class="Stage"),
        td.Node(id="out", op="output", label="output"),
    ]
    edges = [
        td.Edge("in", "s0_ln"),
        td.Edge("s0_ln", "s0_blk"),
        td.Edge("s0_blk", "s0_add"),
        td.Edge("s0_add", "s1_ln"),
        td.Edge("s1_ln", "s1_blk"),
        td.Edge("s1_blk", "s1_add"),
        td.Edge("s1_add", "out"),
    ]
    graph = td.Graph(nodes=nodes, edges=edges)
    result = aggregate_blocks(graph)
    blocks = [node for node in result.nodes if node.op == "block"]
    assert len(blocks) == 2
    assert all("×" not in block.label for block in blocks)


def test_aggregate_badges_repeated_leaf_layers_in_a_container():
    """N identical layers held directly in an nn.Sequential get a count, the same as N identical custom blocks do."""
    result = aggregate_blocks(td.trace(ComposedEncoder(3), torch.randn(1, 5, 8)))
    labels = [node.label for node in result.nodes]
    assert "TransformerEncoderLayer ×3" in labels
    badged = next(node for node in result.nodes if node.label.endswith("×3"))
    assert badged.params["repeats"] == 3


def test_aggregate_walks_past_a_container_no_operation_was_traced_from():
    """A bare nn.Sequential between two custom levels is not where the walk up the module tree stops."""
    result = aggregate_blocks(td.trace(ThreeLevelNet(), torch.randn(1, 3, 8, 8)))
    result.validate()
    assert [node.label for node in result.nodes] == ["input", "WideningBlock", "AdaptiveAvgPool2d", "output"]


def test_aggregate_counts_repeats_of_a_block_that_only_calls_its_children():
    """A block owning no operation of its own is a level like any other, so its repeats are still counted."""
    result = aggregate_blocks(td.trace(DelegatingStack(), torch.randn(1, 8)))
    assert [node.label for node in result.nodes] == ["input", "DelegatingBlock ×3", "output"]


def test_aggregate_does_not_badge_a_layer_that_is_shared_rather_than_repeated():
    """`Linear ×2` claims two sets of weights; a layer listed twice in one Sequential has one."""
    result = aggregate_blocks(td.trace(TiedStack(), torch.randn(1, 4)))
    assert not any("×" in node.label for node in result.nodes)
