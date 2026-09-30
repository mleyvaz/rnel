# rnel

**rnel is a Python library for evidential reasoning that keeps contradiction, ignorance and retracted evidence apart.**

A probability of truth cannot say why a system is unsure. Two sources that contradict each other, a single source
with balanced evidence and no evidence at all can all give 0.5. `rnel` implements Refined Neutrosophic Evidential
Logic (RNEL, Smarandache and Leyva-Vázquez), in which these cases get separate components:

| Component | Meaning | Typical action |
|---|---|---|
| **T** | evidence that the claim is true | answer: supported |
| **F** | evidence that the claim is false | answer: refuted |
| **C** | contradiction: strong evidence for both | present both positions |
| **U** | undetermined: "true or false, cannot say which" | wait, ask for specific evidence |
| **N** | neither: the claim is ill-posed | question the premise |
| **G** | ignorance: no evidence | retrieve more or abstain |

With only T and F, RNEL reduces exactly to Subjective Logic (Jøsang) and to evidential deep learning
(Sensoy et al.). See [Concepts](concepts.md).

!!! warning "Academic, experimental software"
    `rnel` is research code that accompanies papers still under review. The API may change between versions.
    The module `rnel.off` is explicitly experimental.

## Installation

The core has no dependencies beyond the Python standard library (Python 3.9 or later).

```bash
pip install git+https://github.com/mleyvaz/rnel             # core
pip install "rnel[nn] @ git+https://github.com/mleyvaz/rnel"  # with the PyTorch models (rnel.nn)
pip install transformers torch                               # for rnel.text (NLI reader)
```

To install a specific version, pin the tag: `pip install git+https://github.com/mleyvaz/rnel@v0.2.0`.

## Modules

| Module | Contents |
|---|---|
| [`rnel.sl`](api/sl.md) | Subjective Logic: opinions, evidence mapping, multiplication, comultiplication, cumulative and averaging fusion, discounting, ageing, degree of conflict |
| [`rnel.tuple`](api/tuple.md) | RNEL reports, the evidence tuple (Definition 8.1), cumulative fusion (Theorem 8.2), fused contradiction (Definition 8.4) |
| [`rnel.operators`](api/operators.md) | Priority-product ∧, ∨, ¬ for (T, I, N, F), (T, I, N, U₁…Uₙ, F) and RNEL; base-rate-calibrated conjunction and disjunction; coarsening to SL |
| [`rnel.ranking`](api/ranking.md) | Total-order ranking cascades: score, accuracy, extended certainty, certainty |
| [`rnel.decide`](api/decide.md) | From a tuple to an action, with a monotone evidence gate |
| [`rnel.off`](api/off.md) | Experimental: signed-evidence ledger, source-level retraction, over/under/off values |
| [`rnel.text`](api/text.md) | Zero-shot RNEL from a claim and evidence texts via an NLI model |
| [`rnel.nn`](api/nn.md) | PyTorch: typed evidential head, loss, per-source evidential network, source conflict |

## Verified

`pytest` runs the results of the accompanying paper as tests (450 tests in version 0.2.0): the worked example of
Proposition 8.3 and Definition 8.4, Theorem 8.2, Theorem 5.2 on random tuples (involution, De Morgan,
multiplicative mass, projection onto SL multiplication and comultiplication), Theorem 6.2 for n = 3, De Morgan and
SL projection for the RNEL operators, and the neural models.

## Links

- Source code: <https://github.com/mleyvaz/rnel>
- Archived release 0.2.0: <https://doi.org/10.5281/zenodo.23040628>
- Evaluation of the neural head on real fact-checking data: <https://github.com/mleyvaz/rnel-neural-factcheck>
- Licence: MIT
