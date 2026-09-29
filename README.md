# rnel: Refined Neutrosophic Evidential Logic

`rnel` computes with typed evidence. A probability of truth cannot say why a system is unsure: two sources that
contradict each other, a single source with balanced evidence, and no evidence at all can all give 0.5. RNEL
(Smarandache and Leyva-Vázquez) keeps them apart:

| Component | Meaning | Typical action |
|---|---|---|
| **T** | evidence that the claim is true | answer: supported |
| **F** | evidence that the claim is false | answer: refuted |
| **C** | contradiction: strong evidence for both | present both positions |
| **U** | undetermined: "true or false, cannot say which" | wait, ask for specific evidence |
| **N** | neither: the claim is ill-posed | question the premise |
| **G** | ignorance: no evidence | retrieve more or abstain |

With only T and F it reduces exactly to Subjective Logic (Jøsang) and to evidential deep learning (Sensoy et al.).

## Install
```bash
pip install git+https://github.com/mleyvaz/rnel             # core, no dependencies
pip install "rnel[nn] @ git+https://github.com/mleyvaz/rnel"  # with PyTorch models
```

## Quick start
```python
from rnel import Reports, rnel_tuple, fused_contradiction
from rnel.decide import Policy

# two confident sources that disagree, and a fair coin: SL fuses both to the same opinion
x = fused_contradiction([Reports(t=10), Reports(f=10)])
coin = fused_contradiction([Reports(t=10, f=10)])
print(x.C, coin.C)            # 0.579 vs 0.0 (Proposition 8.3 / Definition 8.4)
print(Policy().decide(x))     # ('C', 'present both positions and flag the dispute')

# the evidence tuple of Definition 8.1
print(rnel_tuple(Reports(t=3, f=1, c=2, v=1, n=1)).as_dict())
```

From text, with a public NLI model and no training (`pip install transformers torch`):
```python
from rnel.text import NLIReader, rnel_from_text
t, readings = rnel_from_text(NLIReader(), "Coffee increases the risk of heart disease.",
                             ["A cohort found higher coffee intake linked to more heart disease.",
                              "A meta-analysis found moderate coffee lowers the risk of heart disease."])
```
NLI models often label unrelated text as "contradiction", so give only evidence that is about the claim.

Demo app: `streamlit run examples/app.py`.

## Modules
| Module | Contents |
|---|---|
| `rnel.sl` | Subjective Logic: opinions, evidence mapping, multiplication, comultiplication, cumulative and averaging fusion, discounting, ageing, degree of conflict |
| `rnel.tuple` | RNEL reports, the tuple of Definition 8.1, cumulative fusion (Theorem 8.2), fused contradiction (Definition 8.4) |
| `rnel.operators` | Priority-product ∧, ∨, ¬ for (T, I, N, F), (T, I, N, U₁…Uₙ, F) and RNEL, with coarsening to SL |
| `rnel.ranking` | Total-order cascades: score, accuracy, extended certainty, certainty |
| `rnel.decide` | From a tuple to an action (thresholds to be tuned with the costs of each application). Since v0.1.1 the evidence gate is monotone: a report for x never yields "refuted" (see CHANGELOG) |
| `rnel.nn` | PyTorch: typed evidential head, loss, per-source evidential network, source conflict, fused dissonance |
| `rnel.text` | Zero-shot RNEL from a claim and evidence texts via an NLI model |
| `rnel.datasets` | Loaders for AVeriTeC, CLIMATE-FEVER and ChaosNLI with their mapping to RNEL types |
| `rnel.off` | Experimental signed-evidence ledger, source-level retraction and over/under/off values |

## Verified
`pytest` runs the results of the paper as tests (450 tests):
- the worked example of Proposition 8.3 and Definition 8.4;
- Theorem 8.2;
- Theorem 5.2 on random tuples: involution, De Morgan, multiplicative mass, and projection onto SL
  multiplication and comultiplication;
- Theorem 6.2 for n = 3;
- De Morgan and SL projection for the RNEL operators;
- the neural head and the per-source network.

Version 0.2.0 adds base-rate-calibrated conjunction and disjunction and an experimental signed-evidence ledger. The release is archived at https://doi.org/10.5281/zenodo.23040628.

## Evidence on real data
The preregistered evaluation of the neural head on AVeriTeC, CLIMATE-FEVER and ChaosNLI is at
https://github.com/mleyvaz/rnel-neural-factcheck.

## Cite
F. Smarandache, M. Y. Leyva-Vázquez, *The Neutrosophic (T, I, N, F), Multi-Uncertainty, n-Valued Refined, and
Refined Evidential Logics with Neutrosophic Probability as Extensions of Subjective Logic*, manuscript, 2026.

MIT licence.
