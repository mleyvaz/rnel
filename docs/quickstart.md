# Quickstart

Every example on this page runs as shown against `rnel` 0.2.0; the printed values are in the comments.
The blocks build on each other, so run them in order in one session.

## 1. Contradiction is not a fair coin

Two confident sources that disagree, and one source whose evidence is balanced, give the same Subjective Logic
opinion after fusion. RNEL keeps them apart with the contradiction component C (Proposition 8.3 and
Definition 8.4).

```python
from rnel import Reports, rnel_tuple, fused_contradiction, sl_opinion

dispute = fused_contradiction([Reports(t=10), Reports(f=10)])  # two sources that disagree
coin = fused_contradiction([Reports(t=10, f=10)])              # one source, balanced evidence

print(sl_opinion(Reports(t=10) + Reports(f=10)) == sl_opinion(Reports(t=10, f=10)))  # True
print(round(dispute.C, 3), round(coin.C, 3))                   # 0.579 0.0
```

## 2. The evidence tuple

`Reports` counts reports of each type about a proposition x; `rnel_tuple` turns them into the tuple
(T, C, U, N, G, F) of Definition 8.1. Each component is its count divided by the total count plus the prior
weight W (default 2), and G = W / (total + W).

```python
t = rnel_tuple(Reports(t=3, f=1, c=2, v=1, n=1))
print({k: round(v, 3) for k, v in t.as_dict().items()})
# {'T': 0.3, 'C': 0.2, 'U': 0.1, 'N': 0.1, 'G': 0.2, 'F': 0.1}

print(t.to_sl())  # coarsening to SL: T -> belief, F -> disbelief, the rest -> uncertainty
# Opinion(b=0.3, d=0.1, u=0.6, a=0.5)
```

## 3. From a tuple to an action

`Policy` maps a tuple to one of six actions. The thresholds are defaults; set them on validation data with the
costs of your application.

```python
from rnel.decide import Policy

policy = Policy()
print(policy.decide(dispute))                  # ('C', 'present both positions and flag the dispute')
print(policy.decide(rnel_tuple(Reports())))    # ('G', 'retrieve more evidence or abstain')
print(policy.decide(rnel_tuple(Reports(t=9))))  # ('T', 'answer: supported')
```

## 4. Subjective Logic

`rnel.sl` implements binomial opinions and the Subjective Logic operators that RNEL extends.

```python
from rnel.sl import Opinion, cumulative_fusion, degree_of_conflict, discount

A = Opinion.from_evidence(8, 2)   # 8 reports for, 2 against, W = 2
B = Opinion.from_evidence(1, 9)
print(round(A.projected, 3))                    # 0.75
print(round(degree_of_conflict(A, B), 3))       # 0.405
print(cumulative_fusion(A, B).u == Opinion.from_evidence(9, 11).u)  # True
print(discount(0.5, A))
# Opinion(b=0.3333333333333333, d=0.08333333333333333, u=0.5833333333333333, a=0.5)
```

## 5. Logical operators

Tuples are plain dictionaries for the operators. RNEL negation swaps only T and F: contradiction, undetermined,
neither and ignorance are fixed points. Conjunction is a priority product whose coarsening is SL multiplication.

```python
from rnel.operators import rnel_and, rnel_not, coarsen

p = t.as_dict()
q = rnel_tuple(Reports(t=5, f=1)).as_dict()

print(rnel_not(p) == {**p, "T": p["F"], "F": p["T"]})  # True
r = rnel_and(p, q)
print({k: round(v, 3) for k, v in r.items()})
# {'T': 0.188, 'C': 0.15, 'U': 0.075, 'N': 0.075, 'G': 0.3, 'F': 0.212}
print(tuple(round(v, 3) for v in coarsen(r)))  # (b, u, d) = (0.188, 0.6, 0.212)
```

Base-rate-calibrated versions (`calibrated_and`, `calibrated_or`) reproduce SL multiplication and
comultiplication for arbitrary base rates:

```python
from rnel.operators import calibrated_and, tinf_and
from rnel.sl import multiply

x = {"T": 0.5, "I": 0.3, "N": 0.0, "F": 0.2}
y = {"T": 0.4, "I": 0.4, "N": 0.0, "F": 0.2}
z = calibrated_and(x, y, ax=0.3, ay=0.6, conj=tinf_and)
m = multiply(Opinion(0.5, 0.2, 0.3, 0.3), Opinion(0.4, 0.2, 0.4, 0.6))
print(round(z["T"], 6) == round(m.b, 6), round(z["F"], 6) == round(m.d, 6))  # True True
```

## 6. Ranking

Total-order cascades (score, then accuracy, then extended certainty, then certainty) for (T, I, N, F) tuples.

```python
from rnel.ranking import rank, tinf_key

alternatives = [
    {"T": 0.3, "I": 0.3, "N": 0.2, "F": 0.2},
    {"T": 0.6, "I": 0.2, "N": 0.1, "F": 0.1},
    {"T": 0.6, "I": 0.1, "N": 0.1, "F": 0.2},
]
print(rank(alternatives, key=tinf_key))  # [1, 2, 0]  (indices, best first)
```

## 7. Retracting evidence (experimental)

`rnel.off.EvidenceLedger` records reports per source and lets you retract them. As long as no source retracts
more than it reported, the result is an ordinary SL opinion; removing a source is SL cumulative unfusion.

```python
from rnel.off import EvidenceLedger

# 40 positive reviews and 10 negative; 25 positive ones turn out to be fake
shop = EvidenceLedger().report("shop", 40, 10).retract("shop", for_x=25)
print(shop.net())                                     # (15.0, 10.0)
print(shop.opinion() == Opinion.from_evidence(15, 10))  # True

# dropping a whole source recovers the opinion of the others
L = EvidenceLedger().report("A", 8, 2).report("B", 3, 1).retract_source("B")
print(L.opinion() == Opinion.from_evidence(8, 2))     # True
```

Over-retraction is rejected by default. On request it is returned as a flagged off-opinion, a diagnostic that is
not a probability:

```python
import warnings

bad = EvidenceLedger().report("A", 2, 0).retract("A", for_x=3)
try:
    bad.opinion()
except ValueError:
    print("rejected")                                  # rejected

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    off = bad.opinion(allow_off=True)
print(round(off.b, 3), off.in_simplex, off.beta_admissible)  # -1.0 False False
```

## 8. From text (optional, needs `transformers` and `torch`)

`rnel.text` reads each piece of evidence against a claim with a public NLI model, with no training. The model
(`cross-encoder/nli-deberta-v3-base`) is downloaded on first use.

```python
from rnel.text import NLIReader, rnel_from_text

claim = "Coffee increases the risk of heart disease."
evidence = [
    "A cohort found higher coffee intake linked to more heart disease.",
    "A meta-analysis found moderate coffee lowers the risk of heart disease.",
]
tup, readings = rnel_from_text(NLIReader(), claim, evidence)
print(Policy().decide(tup)[0])  # C
```

NLI models often label unrelated text as "contradiction", so give only evidence that is about the claim.

## 9. Neural evidential head (optional, needs `torch`)

`rnel.nn.RNELHead` maps features to non-negative typed evidence (T, F, C, U, N); `rnel_from_evidence` turns
evidence into the tuple of Definition 8.1 and `typed_evidential_loss` trains it from typed labels.

```python
import torch
from rnel.nn import RNELHead, rnel_from_evidence, typed_evidential_loss

torch.manual_seed(0)
head = RNELHead(in_dim=16)
features = torch.randn(4, 16)
labels = torch.tensor([0, 1, 2, 4])          # indices into (T, F, C, U, N)

evidence = head(features)                    # shape (4, 5)
tuples = rnel_from_evidence(evidence)        # shape (4, 6): T, F, C, U, N, G
print(tuples.shape, bool(torch.allclose(tuples.sum(-1), torch.ones(4))))  # torch.Size([4, 6]) True
loss = typed_evidential_loss(evidence, labels)
loss.backward()
```
