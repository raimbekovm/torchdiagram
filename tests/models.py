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
