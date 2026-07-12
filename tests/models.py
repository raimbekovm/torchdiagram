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


class BranchingModel(nn.Module):
    """Branches on a runtime tensor value — not fx-traceable, exercises the CLI's tracing-error path."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` or ``-x`` depending on a data-dependent condition."""
        if x.sum() > 0:
            return x
        return -x


class NeedsConstructorArgs(nn.Module):
    """Requires a constructor argument — exercises the CLI's zero-argument-factory error path."""

    def __init__(self, num_classes: int) -> None:
        """Initialize a linear classifier head with ``num_classes`` outputs."""
        super().__init__()
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear classifier."""
        return self.fc(x)
