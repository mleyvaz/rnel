# rnel: Refined Neutrosophic Evidential Logic

[![Documentation](https://img.shields.io/badge/docs-mleyvaz.github.io%2Frnel-blue)](https://mleyvaz.github.io/rnel/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23040628.svg)](https://doi.org/10.5281/zenodo.23040628)

Documentation: https://mleyvaz.github.io/rnel/

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

## Credal tools for neutrosophic triples
`rnel.neutro_credal` (experimental; `pip install "rnel[credal]"`, needs numpy and scipy) reads a triple (T, I, F) of an
event as the bounds T <= P(A) <= 1 - F of a credal set (Walley's betting reading). Through this reading the
neutrosophic triple inherits Walley's quality checks and tools, applied directly to triples:

| Function | What it does | Result used |
|---|---|---|
| `avoids_sure_loss`, `sure_loss_degree` | is the credal set non-empty; degree (T + F - 1)/2 for one triple | Theorem 3 |
| `is_coherent`, `classify` | every bound attained (LP, or closed form I_i <= Delta <= sum_{j!=i} I_j on singletons) | Theorems 1, 2 |
| `natural_extension`, `coherent_correction` | triple of a composite event ("A or C"); correction with I^E <= I | Theorems 1(c), 2(c) |
| `singleton_intervals_check`, `singleton_natural_extension` | singletons as probability intervals, closed forms | Theorem 2 |
| `idm_triple` | imprecise Dirichlet model as triples | Corollary 2 |
| `interval_dominance`, `maximality`, `e_admissible` | decisions by sets over per-class triples or member distributions | Walley, Levi |
| `to_glut_frame`, `glut_conjunction` | credal set on a frame with a glut atom, including T + F > 1; N-norms as natural extensions | Theorems 8, 9 |
| `sl_retraction` | saturating chart to Subjective Logic | Theorem 5 |
| `credal_diagnosis` | all of the above plus the RNEL state (a, b, u, c) and C* from evidence per source, in one call | |

```python
from rnel import neutro_credal as nc

members = [(0.7, 0.2, 0.1), (0.6, 0.3, 0.1), (0.1, 0.8, 0.1)]   # three classifiers, classes A, B, C
rep = nc.credal_diagnosis(members, labels=["A", "B", "C"], events={"A or C": ["A", "C"]})
rep["triples"]         # A (0.1, 0.6, 0.3), B (0.2, 0.6, 0.2), C (0.1, 0.0, 0.9): intervals [0.1,0.7], [0.2,0.8], [0.1,0.1]
rep["status"]          # 'coherent'
rep["events"]["A or C"]  # (0.2, 0.6, 0.2): P(A or C) in [0.2, 0.8]
rep["decision"]        # ['A', 'B'] (maximality; E-admissibility and interval dominance agree here)
rep["rnel"]            # why: per class a (support), b (rejection), u (absence), c (conflict between sources); C*

nc.sure_loss_degree((0.7, 0.1, 0.6))       # 0.15 = (T + F - 1) / 2
nc.to_glut_frame((0.8, 0.3, 0.7)).represents()   # True: the glut frame represents T + F > 1
```
The same call accepts a fitted `rnel.credal.CredalEnsemble` (`credal_diagnosis(ensemble=ens, X=X)`), using the leaf
counts of each tree as evidence per source. A longer walk-through is in `examples/neutro_credal_demo.py`. Theorem
numbers refer to Leyva-Vázquez and Smarandache, *Neutrosophic probability and credal sets: exact reduction theorems
and a glut lifting* (manuscript, 2026); the tests check the closed forms of Theorem 2 against linear programming on
3000 random assessments (16 812 events, maximum error 3.3e-16) and reproduce Table 2 and counterexamples C1-C4.

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
| `rnel.neutro_credal` | Experimental: Walley's tools (sure loss, coherence, natural extension, IDM, decisions by sets, glut lifting) on neutrosophic triples |

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
