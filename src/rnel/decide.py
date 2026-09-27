"""From an RNEL tuple to an action (technical note, Section 5.3).

Evidence gate (v0.1.1). Version 0.1.0 abstained when the ignorance component G was high and otherwise
answered with the side (T or F) that led by the margin. Because G falls whenever *any* report arrives,
a report supporting x could lower G below its threshold and unlock the answer "refuted": with the default
thresholds, 3 opposing reports -> abstain, but 3 opposing + 1 supporting -> "refuted". This violates the
AGM inclusion postulate (evidence for x must never produce belief in not-x). The gate is now applied to
the evidence on the side being asserted: to answer T (or F) the number of supporting (or opposing)
reports must exceed the count the old total-mass gate required, W (1/G_threshold - 1). With evidence on
one side only, the two gates coincide; with mixed evidence the new gate is monotone. Set
``side_gate=False`` to reproduce v0.1.0.
"""
from __future__ import annotations

from dataclasses import dataclass

from .sl import DEFAULT_W
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
    W: float = DEFAULT_W  # prior weight used to build the tuple (recovers report counts from it)
    side_gate: bool = True  # False = v0.1.0 behaviour (total-mass gate, not monotone)

    def required_side_evidence(self) -> float:
        """Reports needed on the asserted side: the count at which the total-mass gate opened in v0.1.0."""
        return self.W * (1.0 / self.G - 1.0) if self.G > 0 else 0.0

    def decide(self, x: RNELTuple) -> tuple[str, str]:
        for comp in ("C", "N", "U"):
            if getattr(x, comp) >= getattr(self, comp):
                return comp, ACTIONS[comp]
        if not self.side_gate:
            if x.G >= self.G:
                return "G", ACTIONS["G"]
            if abs(x.T - x.F) >= self.margin:
                comp = "T" if x.T > x.F else "F"
                return comp, ACTIONS[comp]
            return "G", ACTIONS["G"]
        if abs(x.T - x.F) >= self.margin:
            comp = "T" if x.T > x.F else "F"
            if x.G <= 0:  # dogmatic tuple: no prior weight, counts are unbounded
                return comp, ACTIONS[comp]
            side_count = (x.T if comp == "T" else x.F) * self.W / x.G  # Sigma = W / G
            if side_count > self.required_side_evidence() + 1e-9:
                return comp, ACTIONS[comp]
        return "G", ACTIONS["G"]
