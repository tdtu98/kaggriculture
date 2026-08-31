"""Grid- and state-aware behavior-cloning baseline."""

import torch
from torch import nn


class StateAwareModel(nn.Module):
    """Fixed v0 tile, global-state, and actor-state model."""

    def __init__(self) -> None:
        super().__init__()
        self.tile = nn.Sequential(
            nn.Conv2d(44, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.actor = nn.Sequential(
            nn.Linear(38, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(62, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(448, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 17),
        )

    def forward(
        self,
        grid: torch.Tensor,
        global_features: torch.Tensor,
        actor_features: torch.Tensor,
    ) -> torch.Tensor:
        encoded = torch.cat(
            (
                self.tile(grid),
                self.global_encoder(global_features),
                self.actor(actor_features),
            ),
            dim=1,
        )
        return self.classifier(encoded)
