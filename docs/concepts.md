# Concepts

This page summarises the ideas behind `rnel`. The formal development is in the accompanying paper:

- F. Smarandache and M. Y. Leyva-Vázquez, *The Neutrosophic (T, I, N, F), Multi-Uncertainty, n-Valued Refined, and
  Refined Evidential Logics with Neutrosophic Probability as Extensions of Subjective Logic*. Preprint, 2026.
  [doi:10.20944/preprints202609.2584.v1](https://doi.org/10.20944/preprints202609.2584.v1). Under review.

Definition, theorem and section numbers in the docstrings refer to that paper.

## Why typed evidence

Subjective Logic (SL) describes a binary proposition x with an opinion (b, d, u, a): belief, disbelief,
uncertainty and base rate, with b + d + u = 1. It is built from evidence counts r (for x) and s (against x) and a
prior weight W:

$$
b = \frac{r}{r + s + W}, \qquad d = \frac{s}{r + s + W}, \qquad u = \frac{W}{r + s + W}.
$$

The single uncertainty mass u mixes situations that call for different actions. Two sources that contradict each
other, one source with balanced evidence and a total absence of evidence can all end with the same projected
probability P = b + a u. RNEL gives each of these situations its own component.

## The RNEL tuple (T, C, U, N, G, F)

Evidence about x is counted by type in [`Reports`](api/tuple.md): t reports support x, f oppose x, c support both
(contradictory), v say "x or not x, but cannot say which" (undetermined), and n say "neither" (the claim is
ill-posed). With Σ = t + f + c + v + n + W, the tuple of Definition 8.1 is

$$
(T, C, U, N, G, F) = \left(\frac{t}{\Sigma}, \frac{c}{\Sigma}, \frac{v}{\Sigma}, \frac{n}{\Sigma},
\frac{W}{\Sigma}, \frac{f}{\Sigma}\right).
$$

| Component | Meaning |
|---|---|
| T | evidence that x is true |
| F | evidence that x is false |
| C | contradiction: evidence for both |
| U | undetermined: x or not x, cannot say which |
| N | neither: x is ill-posed |
| G | ignorance: the prior weight that no evidence has yet displaced |

Further indeterminacy types can be added through `Reports.extra`. Fusion of independent sources is componentwise
addition of counts (Theorem 8.2), so with only t and f it coincides with SL cumulative fusion.

### Contradiction between sources

Counting alone cannot distinguish "source A says true, source B says false" from "one source is split".
[`fused_contradiction`](api/tuple.md) (Definition 8.4) takes C as the larger of each source's own contradiction
and the largest pairwise SL degree of conflict between the sources. The total may then exceed 1: the tuple is
paraconsistent, not a probability distribution.

## Projection to Subjective Logic

Every RNEL tuple coarsens to an SL opinion: T becomes belief, F becomes disbelief, and all indeterminacy-type
components (C, U, N, G and any extra types) become uncertainty. This is `RNELTuple.to_sl` and
`rnel.operators.coarsen`. The operators are designed to commute with this projection:

- RNEL negation swaps T and F and leaves C, U, N and G fixed (as in Belnap's four-valued logic).
- The priority-product conjunction and disjunction project onto SL multiplication and comultiplication
  (Theorem 5.2 for (T, I, N, F), Theorem 6.2 for multi-uncertainty tuples).
- The base-rate-calibrated versions (`calibrated_and`, `calibrated_or`, Section 5.2) extend this to arbitrary
  base rates.

With only T and F, RNEL reduces exactly to SL and to evidential deep learning (Sensoy et al., 2018). The test
suite checks these reductions numerically.

## From a tuple to a decision

[`Policy`](api/decide.md) checks C, N and U against thresholds (present both positions, question the premise,
wait for specific evidence), then answers T or F only when one side leads by a margin and has enough evidence of
its own; otherwise it abstains (G). Since version 0.1.1 the evidence gate is monotone: a report supporting x can
never produce the answer "refuted". The default thresholds are placeholders to be set with the costs of the
application.

## Signed evidence and retraction (experimental)

Evidence is sometimes withdrawn: reviews turn out to be fake, a study is retracted, a source is found to be
unreliable. [`rnel.off`](api/off.md) keeps an `EvidenceLedger` of reports and retractions per source.

- While no source retracts more than it reported, net evidence is non-negative and the ledger yields an ordinary
  SL opinion. Removing a source is exactly SL cumulative unfusion.
- A source known to report the opposite of the truth can be inverted: its reports for x count as evidence for not x.
- If a source retracts more than it reported, net evidence becomes negative. The ledger rejects this by default;
  with `allow_off=True` it returns an `OffOpinion`, whose components may leave [0, 1] (over/under/off values).
  Such values are diagnostics of over-retraction or double counting. When the Beta parameters are not positive
  there is no Beta distribution behind them, so they are not probabilities and should not drive decisions.

The module also implements the over/under/off operators of the accompanying work for verification. This part of
the theory is under review and the module is experimental.

## Related material

- Preprint: [doi:10.20944/preprints202609.2584.v1](https://doi.org/10.20944/preprints202609.2584.v1)
- Evaluation of the neural head on AVeriTeC, CLIMATE-FEVER and ChaosNLI:
  <https://github.com/mleyvaz/rnel-neural-factcheck>
- A. Jøsang, *Subjective Logic: A Formalism for Reasoning Under Uncertainty*, Springer, 2016.
- M. Sensoy, L. Kaplan and M. Kandemir, "Evidential deep learning to quantify classification uncertainty",
  NeurIPS 2018.
