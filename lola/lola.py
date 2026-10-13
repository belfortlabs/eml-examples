"""The LoLA network this application serves, as a model provider would write it.

The plaintext counterpart of `orion.models.lola.LoLA`, the small MNIST CNN from
the LoLA paper: one convolution block, then two Linear layers, with Quad
activations throughout.
"""

from typing import Annotated

import torch
from belfort_ml import Secret
from torch import nn


class Quad(nn.Module):
    """CKKS evaluates polynomials, so squaring is exact where a ReLU would
    have to be approximated over a calibrated interval."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class LoLA(nn.Module):
    """The trained model: one strided convolution, then two Linear layers.

    The stride-2 2x2 convolution takes an MNIST digit to 5 maps of 14x14, the
    980 values `fc1` reads.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=2, padding=0, stride=2)
        self.bn1 = nn.BatchNorm2d(5)
        self.act1 = Quad()

        self.fc1 = nn.Linear(980, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.act2 = Quad()

        self.fc2 = nn.Linear(100, num_classes)
        self.flatten = nn.Flatten()

    def forward(self, x: Annotated[torch.Tensor, Secret]) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.flatten(x)
        x = self.act2(self.bn2(self.fc1(x)))
        return self.fc2(x)
