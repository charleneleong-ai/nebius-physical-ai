"""A learned verifier for best-of-N action chunks, after DiVeR (arXiv 2610.04933).

The policy samples N chunks per observation. A small classifier, trained on rollouts the frozen policy already ran,
scores each candidate by how much it looks like the state-action pairs of successful episodes rather than failed ones,
and the highest-scoring chunk is executed. DiVeR's change to plain outcome classification: each training step is
weighted by how much the policy's own candidates disagree there (`criticality`), so the few decision-critical steps
(pre-grasp, pre-placement) dominate instead of the many routine ones where every candidate is the same.

DiVeR represents a candidate by the action expert's hidden state. Here a candidate is the observation's pooled
vision-language features (`backbone_features`, the same for every candidate), its joints, and the chunk itself relative
to those joints, which needs no change inside the policy. The torch-free parts (`candidate_features`, `criticality`,
`decision_weights`) live in `best_of_n`, so the twin's venv can record verifier data without torch.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn


class Verifier(nn.Module):
    """MLP over standardised candidate features; its logit is the score (the log success-to-failure visitation ratio)."""

    def __init__(self, width: int, hidden: int = 256) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("std", torch.ones(width))
        self.net = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def fit_normaliser(self, x: torch.Tensor) -> "Verifier":
        """Standardise inputs with the training features' per-column mean and spread."""
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0).clamp_min(1e-4))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std).squeeze(-1)

    def pick(self, features: np.ndarray) -> np.ndarray:
        """(B, N, F) candidate features -> (B,) index of the best-scoring candidate per row."""
        with torch.no_grad():
            scores = self(torch.as_tensor(features, device=self.mean.device))
        return scores.argmax(dim=-1).cpu().numpy()

    def save(self, path: Path) -> None:
        torch.save({"width": self.mean.numel(), "hidden": self.net[0].out_features, "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "Verifier":
        saved = torch.load(path, map_location=device)
        verifier = cls(saved["width"], saved["hidden"])
        verifier.load_state_dict(saved["state"])
        return verifier.to(device).eval()
