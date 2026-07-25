"""Small fixture models shared across the test suite."""

import torch
import torch.nn.functional as F
from torch import nn


class TinyCNN(nn.Module):
    """Minimal conv-pool-linear classifier for 28x28 single-channel inputs."""

    def __init__(self) -> None:
        """Initialize conv, pool, and classifier layers."""
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run conv-relu-pool, flatten, then classify."""
        x = self.pool(F.relu(self.conv(x)))
        return self.fc(torch.flatten(x, 1))


class ResidualBlock(nn.Module):
    """Two convolutions with a skip connection — exercises non-sequential data flow."""

    def __init__(self) -> None:
        """Initialize the two convolution layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv-relu-conv, add the residual, and activate."""
        return F.relu(self.conv2(F.relu(self.conv1(x))) + x)


class BasicBlock(nn.Module):
    """Two convolutions with a residual add — the repeatable unit for block-aggregation tests."""

    def __init__(self, channels: int) -> None:
        """Initialize the two convolution layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv-relu-conv, add the residual, and activate."""
        return F.relu(self.conv2(F.relu(self.conv1(x))) + x)


class RepeatedBlockStack(nn.Module):
    """A stem convolution followed by ``num_blocks`` identical ``BasicBlock`` instances."""

    def __init__(self, num_blocks: int = 4) -> None:
        """Initialize the stem convolution and the stack of repeated blocks."""
        super().__init__()
        self.stem = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.layer1 = nn.Sequential(*[BasicBlock(4) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the stem convolution, then the block stack."""
        return self.layer1(self.stem(x))


class NonUniformBlockStack(nn.Module):
    """A stack whose first block has an extra downsample conv; the rest are uniform ``BasicBlock`` instances."""

    class _DownsampleBlock(nn.Module):
        """A ``BasicBlock`` variant with an extra downsample convolution on the residual path."""

        def __init__(self, channels: int) -> None:
            """Initialize the two main convolutions and the extra downsample convolution."""
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.downsample = nn.Conv2d(channels, channels, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply conv-relu-conv, add a downsampled residual, and activate."""
            return F.relu(self.conv2(F.relu(self.conv1(x))) + self.downsample(x))

    def __init__(self, num_blocks: int = 4) -> None:
        """Initialize the stem convolution, the downsample block, and the uniform block stack."""
        super().__init__()
        self.stem = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.layer1 = nn.Sequential(
            self._DownsampleBlock(4),
            *[BasicBlock(4) for _ in range(num_blocks - 1)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the stem convolution, then the non-uniform block stack."""
        return self.layer1(self.stem(x))


class ConvStage(nn.Module):
    """A convolution stack behind a pooling layer — a VGG stage, the shape that hid its contents from aggregation."""

    def __init__(self, in_channels: int, out_channels: int, num_convs: int) -> None:
        """Initialize ``num_convs`` convolutions inside a Sequential, followed by a pool."""
        super().__init__()
        channels = [in_channels] + [out_channels] * num_convs
        self.body = nn.Sequential(*[nn.Conv2d(channels[i], channels[i + 1], 3, padding=1) for i in range(num_convs)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the convolution stack, then pool."""
        return self.pool(self.body(x))


class WideningStack(nn.Module):
    """Two stages with the same op sequence but different channel counts — repeats in shape only, not in fact."""

    def __init__(self) -> None:
        """Initialize two stages of equal depth and unequal width."""
        super().__init__()
        self.stage1 = ConvStage(3, 4, 1)
        self.stage2 = ConvStage(4, 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply both stages in sequence."""
        return self.stage2(self.stage1(x))


class MixedDepthStack(nn.Module):
    """Two identically sized stages holding a different number of convolutions, which only their contents reveal."""

    def __init__(self) -> None:
        """Initialize a two-convolution stage followed by a three-convolution one."""
        super().__init__()
        self.stage1 = ConvStage(4, 4, 2)
        self.stage2 = ConvStage(4, 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply both stages in sequence."""
        return self.stage2(self.stage1(x))


class ClassifierHead(nn.Module):
    """A model whose head is a bare ``nn.Sequential``, whose class name says nothing about what it does."""

    def __init__(self) -> None:
        """Initialize the stem convolution and the Sequential classifier."""
        super().__init__()
        self.stem = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.classifier = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool the stem features to a vector, then classify."""
        return self.classifier(torch.flatten(F.adaptive_avg_pool2d(self.stem(x), 1), 1))


class ShapeMath(nn.Module):
    """Builds a position range from its input's shape — metadata plumbing the diagram must route around, not draw."""

    def __init__(self) -> None:
        """Initialize the position embedding and the output projection."""
        super().__init__()
        self.pos = nn.Embedding(16, 4)
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add a position embedding sized from the input's own length, then project."""
        return self.proj(x + self.pos(torch.arange(x.shape[1])))


class Attention(nn.Module):
    """Multi-head self-attention written out by hand — qkv projection, softmax, no fused kernel or leaf module."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        """Initialize the qkv and output projections."""
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split into heads, attend, merge heads, and project."""
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class MLP(nn.Module):
    """Two-layer feed-forward block with a GELU activation."""

    def __init__(self, dim: int, hidden: int) -> None:
        """Initialize the two linear layers."""
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fc1-GELU-fc2."""
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Pre-norm attention + MLP block with residual connections, the repeatable unit for transformer stacks."""

    def __init__(self, dim: int, heads: int) -> None:
        """Initialize the norms and the attention and MLP sub-modules."""
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and MLP, each with a pre-norm and a residual add."""
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TransformerStack(nn.Module):
    """A stack of identical ``TransformerBlock`` instances — the repeatable unit for nested block-aggregation tests."""

    def __init__(self, num_blocks: int = 4, dim: int = 32, heads: int = 4) -> None:
        """Initialize the stack of transformer blocks."""
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply each block in sequence."""
        for block in self.blocks:
            x = block(x)
        return x


class GatedNet(nn.Module):
    """Real layers around a data-dependent branch — the fx-untraceable case the torch.export frontend covers."""

    def __init__(self) -> None:
        """Initialize the stem convolution, the pooling layer, and the classifier head."""
        super().__init__()
        self.stem = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool the stem features, negate them when the input sums to zero or less, then classify."""
        h = torch.flatten(self.pool(F.relu(self.stem(x))), 1)
        # Branching on the input rather than on activations keeps which branch runs independent of random init.
        if x.sum() > 0:
            return self.head(h)
        return self.head(-h)


class UnusedBranch(nn.Module):
    """Computes a layer it never returns — dead code both frontends must keep drawing, unlike guard plumbing."""

    def __init__(self) -> None:
        """Initialize the returned and the unused linear layers."""
        super().__init__()
        self.used = nn.Linear(4, 4)
        self.unused = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply both layers but return only one."""
        self.unused(x)
        return self.used(x)


class ComposedEncoder(nn.Module):
    """Built out of torch's own composite layers rather than hand-written blocks.

    ``nn.TransformerEncoderLayer`` is a leaf to fx and a stack of nine children to a naive reading of an export graph,
    which is the shape on which the two frontends stopped agreeing.
    """

    def __init__(self, num_blocks: int = 2) -> None:
        """Initialize the input projection and the stack of encoder layers."""
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.blocks = nn.Sequential(
            *[nn.TransformerEncoderLayer(8, 2, 16, batch_first=True) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project, then run the encoder stack."""
        return self.blocks(self.proj(x))


class CroppedHead(nn.Module):
    """Subscripts its input before projecting — one Python expression each tracer lowers its own way."""

    def __init__(self) -> None:
        """Initialize the projection."""
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take the first position, drop its last feature, and project."""
        return self.proj(x[:, 0, :-1])


class DualEncoder(nn.Module):
    """Two towers scored against each other — a model whose ``forward()`` takes more than one tensor."""

    def __init__(self) -> None:
        """Initialize the two encoders."""
        super().__init__()
        self.left = nn.Linear(4, 6)
        self.right = nn.Linear(8, 6)

    def forward(self, left_input: torch.Tensor, right_input: torch.Tensor) -> torch.Tensor:
        """Encode both inputs and score them by a dot product."""
        return (self.left(left_input) * self.right(right_input)).sum(dim=-1)


class PyramidHeads(nn.Module):
    """A detection-style head returning one prediction per level — several tensors, not one."""

    def __init__(self) -> None:
        """Initialize the stem, the downsample, and the two prediction heads."""
        super().__init__()
        self.stem = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.down = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)
        self.head1 = nn.Conv2d(4, 2, kernel_size=1)
        self.head2 = nn.Conv2d(8, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict from both pyramid levels and return both."""
        fine = self.stem(x)
        return self.head1(fine), self.head2(self.down(fine))


class NeedsConstructorArgs(nn.Module):
    """Requires a constructor argument — exercises the CLI's zero-argument-factory error path."""

    def __init__(self, num_classes: int) -> None:
        """Initialize a linear classifier head with ``num_classes`` outputs."""
        super().__init__()
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear classifier."""
        return self.fc(x)
