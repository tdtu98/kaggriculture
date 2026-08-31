"""Schedule-only behavior-cloning baseline."""

import torch
from torch import nn

from bc_core.constants import CLOCK_DIM, OPERATIONS


class ClockOnlyModel(nn.Module):
    """Baseline restricted to the eight clock and actor-identity features."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(CLOCK_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(OPERATIONS)),
        )

    def forward(self, clock_features: torch.Tensor) -> torch.Tensor:
        return self.network(clock_features)
