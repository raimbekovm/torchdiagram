"""Tests for the block-aggregation IR-to-IR transform."""

import torch

import torchdiagram as td
from tests.models import NonUniformBlockStack, RepeatedBlockStack, TinyCNN
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
    """A run shorter than min_repeats is left uncollapsed."""
    graph = td.trace(RepeatedBlockStack(2))
    unchanged = aggregate_blocks(graph, min_repeats=3)
    assert len(unchanged.nodes) == len(graph.nodes)
    collapsed = aggregate_blocks(graph, min_repeats=2)
    assert any(node.op == "block" for node in collapsed.nodes)


def test_aggregate_leaves_non_repeated_graph_unchanged():
    """A model with no scoped repetition passes through with identical node and edge counts."""
    graph = td.trace(TinyCNN())
    result = aggregate_blocks(graph)
    assert len(result.nodes) == len(graph.nodes)
    assert len(result.edges) == len(graph.edges)


def test_aggregate_partial_match_collapses_only_uniform_suffix():
    """A stage's non-uniform first block stays individually present; the uniform rest collapses."""
    graph = td.trace(NonUniformBlockStack(3))
    result = aggregate_blocks(graph)
    ops = [node.op for node in result.nodes]
    assert ops.count("block") == 1
    assert "downsample" in " ".join(node.id for node in result.nodes if "downsample" in node.id)
    block = next(node for node in result.nodes if node.op == "block")
    assert block.label == "BasicBlock ×2"


def test_aggregate_result_passes_validate():
    """The transform's output is internally consistent."""
    graph = td.trace(RepeatedBlockStack(4))
    aggregate_blocks(graph).validate()  # must not raise
