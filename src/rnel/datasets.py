"""Public data sets whose labels map onto RNEL report types (download them yourself; licences apply).

- AVeriTeC (CC BY-NC 4.0): https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/{train,dev}.json
- CLIMATE-FEVER: https://raw.githubusercontent.com/tdiggelm/climate-fever-dataset/main/dataset/climate-fever.jsonl
- ChaosNLI, MNLI part (CC BY-NC 4.0): https://huggingface.co/datasets/tasksource/chaos-mnli-ambiguity
"""
from __future__ import annotations

import json
from pathlib import Path

AVERITEC = {"Supported": "T", "Refuted": "F", "Conflicting Evidence/Cherrypicking": "C", "Not Enough Evidence": "N"}
CLIMATE = {"SUPPORTS": "T", "REFUTES": "F", "DISPUTED": "C", "NOT_ENOUGH_INFO": "N"}
CHAOS = {"e": "T", "c": "F", "n": "N"}


def load_averitec(path: str | Path) -> list[dict]:
    """Claims with question-answer evidence pieces and their RNEL type."""
    out = []
    for x in json.loads(Path(path).read_text(encoding="utf-8")):
        ev = []
        for q in x["questions"]:
            for a in q["answers"]:
                t = f"Q: {q['question']} A: {a['answer']}"
                if a.get("boolean_explanation"):
                    t += f" {a['boolean_explanation']}"
                ev.append(t)
        out.append({"claim": x["claim"], "evidence": ev, "type": AVERITEC[x["label"]]})
    return out


def load_climate_fever(path: str | Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        x = json.loads(line)
        out.append({"claim": x["claim"], "evidence": [e["evidence"] for e in x["evidences"]],
                    "evidence_labels": [e["evidence_label"] for e in x["evidences"]],
                    "type": CLIMATE[x["claim_label"]]})
    return out


def load_chaosnli(path: str | Path, ambiguity_threshold: float = 0.60) -> list[dict]:
    """Items whose most frequent label gets less than the threshold of the votes are typed U."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        x = json.loads(line)
        top = max(x["label_count"]) / sum(x["label_count"])
        out.append({"claim": x["hypothesis"], "evidence": [x["premise"]],
                    "type": "U" if top < ambiguity_threshold else CHAOS[x["majority_label"]],
                    "entropy": x["entropy"]})
    return out
