"""Zero-shot RNEL from text (requires transformers): each piece of evidence is read by an NLI model.

Every (evidence, claim) pair gives an SL opinion b = p(entailment), d = p(contradiction),
u = p(neutral). The pieces are then combined into RNEL reports and read with Definition 8.4:
T, F and G from the fused evidence, C from the largest pairwise conflict between pieces.
No training is involved; for a trained reader see rnel.nn.
"""
from __future__ import annotations

from dataclasses import dataclass

from .sl import DEFAULT_W, Opinion
from .tuple import Reports, RNELTuple, fused_contradiction

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"


@dataclass
class PieceReading:
    evidence: str
    entail: float
    contradict: float
    neutral: float


class NLIReader:
    def __init__(self, model: str = DEFAULT_MODEL):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.idx = (labels["entailment"], labels["contradiction"], labels["neutral"])

    def read(self, claim: str, evidence: list[str]) -> list[PieceReading]:
        import torch
        with torch.no_grad():
            enc = self.tok(evidence, [claim] * len(evidence), truncation=True, max_length=256,
                           padding=True, return_tensors="pt")
            p = torch.softmax(self.model(**enc).logits, -1).numpy()
        return [PieceReading(ev, float(r[self.idx[0]]), float(r[self.idx[1]]), float(r[self.idx[2]]))
                for ev, r in zip(evidence, p)]


def reports_from_reading(r: PieceReading, weight: float = 5.0) -> Reports:
    """One piece as `weight` units of evidence split by the NLI probabilities.

    Neutral mass adds no evidence (the text does not bear on the claim), so it shows up as ignorance G.
    NLI gives no signal for "neither" (ill-posed claim) or "undetermined"; those need typed data (rnel.nn).
    """
    return Reports(t=weight * r.entail, f=weight * r.contradict)


def rnel_from_text(reader: NLIReader, claim: str, evidence: list[str], weight: float = 5.0,
                   W: float = DEFAULT_W) -> tuple[RNELTuple, list[PieceReading]]:
    readings = reader.read(claim, evidence)
    reps = [reports_from_reading(r, weight) for r in readings]
    return fused_contradiction(reps, W), readings


def sl_from_text(readings: list[PieceReading], weight: float = 5.0, W: float = DEFAULT_W) -> Opinion:
    """The SL cumulative-fusion opinion of the same pieces, for comparison."""
    r = sum(weight * x.entail for x in readings)
    s = sum(weight * x.contradict for x in readings)
    return Opinion.from_evidence(r, s, W)
