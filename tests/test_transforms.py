"""Tests for the block-aggregation IR-to-IR transform."""

import torch

import torchdiagram as td
from tests.models import NonUniformBlockStack, RepeatedBlockStack, TinyCNN, TransformerStack
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
