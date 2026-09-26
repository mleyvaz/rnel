"""From an RNEL tuple to an action (technical note, Section 5.3)."""
from __future__ import annotations

from dataclasses import dataclass

from .tuple import RNELTuple

ACTIONS = {
    "T": "answer: supported",
    "F": "answer: refuted",
    "C": "present both positions and flag the dispute",
    "U": "wait or ask for more specific evidence",
    "N": "question the premise of the claim",
    "G": "retrieve more evidence or abstain",
}


@dataclass
class Policy:
    """Thresholds per component; set them on validation data with the costs of the application."""

    C: float = 0.25
    U: float = 0.30
    N: float = 0.30
    G: float = 0.40
    margin: float = 0.15  # minimum |T - F| to answer

    def decide(self, x: RNELTuple) -> tuple[str, str]:
        for comp in ("C", "N", "U", "G"):
            if getattr(x, comp) >= getattr(self, comp):
                return comp, ACTIONS[comp]
        if abs(x.T - x.F) >= self.margin:
            comp = "T" if x.T > x.F else "F"
            return comp, ACTIONS[comp]
        return "G", ACTIONS["G"]
