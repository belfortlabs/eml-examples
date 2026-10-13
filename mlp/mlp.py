"""The MLP this application serves, as a model provider would write it."""

from typing import Annotated

import torch
from belfort_ml import Secret
from torch import nn


class Quad(nn.Module):
    """CKKS evaluates polynomials, so squaring is exact where a ReLU would
    have to be approximated over a calibrated interval."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class OrionMLP(nn.Module):
    """The trained model: three Linear layers, BatchNorm and Quad between them."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(784, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = Quad()

        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = Quad()

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x: Annotated[torch.Tensor, Secret]) -> torch.Tensor:
        x = self.flatten(x)
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        return self.fc3(x)
