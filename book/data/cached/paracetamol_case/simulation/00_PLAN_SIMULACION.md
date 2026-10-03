# Simulation plan (written 2026-10-02 BEFORE running any simulation; ADEMP structure)

## Aim
Test, without the circularity of A3, whether choosing the next study by the dominant named symbol of the typed
decomposition leads to better knowledge of the TRUE causal effect than alternatives. The outcome is measured against
the simulated truth, never against the width of the decomposition.

## Data-generating mechanism (per simulated world)
- True causal log-effect: theta = 0 with prob. 0.5, else theta ~ U(0.10, 0.40).
- Confounding: with prob. pi_conf the world is confounded, bias beta ~ U(0.10, 0.40); otherwise beta = 0.
  Scenarios: pi_conf in {0, 0.25, 0.5, 0.75, 1}.
- Population-model study: estimate ~ N(theta + b, se^2), se ~ U(0.05, 0.15); b = beta, or beta/2 for the 30% of
  studies flagged "extensively adjusted".
- Identification study (sibling-type): estimate ~ N(theta + e, se^2), se = 2.5 x U(0.05, 0.15), residual bias
  e ~ N(0, 0.03^2). With prob. 0.05 an identification study is "bidirectional" (internally mixed direction).
- Initial evidence: k0 ~ Uniform{3..8} studies, each a population model with prob. 0.85, else identification.

## Coding rule (observable quantities only; mirrors the paracetamol coding; never uses theta or beta)
- Identification, bidirectional -> c.
- Identification, 95% CI lower bound > 0 -> t.  Identification, CI upper bound < 0 -> f.
- Identification, CI includes 0 and CI upper bound < 0.20 -> f (precise null opposes P). Otherwise -> v.
- Population model, extensively adjusted, CI lower bound > 0 -> t.
- Population model, not extensively adjusted, CI lower bound > 0 -> A (association without identification).
- Population model, CI includes 0 -> v.  Population model, CI upper bound < 0 -> f.
P = "the exposure causally increases the outcome". W = 2.

## Actions (one round, equal cost)
- MORE: 4 new population-model studies (same distribution).
- LARGER: 2 new population-model studies with half the standard error.
- IDENT: 2 new identification studies.
Typed mapping: dominant I_A or I_C -> IDENT; I_U -> LARGER; I_G -> MORE. Ties: two pre-declared tie rules are both
reported, "ties->IDENT" and "ties->MORE".

## Estimand / outcome (truth-based, computed by a design-aware synthesis that does NOT use the decomposition)
After the action, the synthesis estimate is the inverse-variance (DerSimonian-Laird) pooled estimate of the
identification studies if at least 2 exist, otherwise of all studies. Outcomes:
- O1 absolute error |estimate - theta| (primary);
- O2 verdict loss: verdict "effect" if pooled 95% CI lower > 0, "null" if CI within (-0.10, 0.10), else
  "inconclusive"; loss 0 if correct (effect when theta > 0, null when theta = 0), 1 if wrong conclusive,
  0.3 if inconclusive.

## Policies compared
oracle (best realised action per world, upper bound); typed (two tie rules); always MORE; always LARGER;
always IDENT; random; interval-learned (best action per bin of the observed range of p(P) and N, learned on a
training half of the same scenario: it gets to learn the scenario's confounding prevalence, the typed rule does not).

## Performance measures
Mean O1 and O2 per policy and scenario on the test half, with Monte Carlo SE; share of worlds where the policy picks
the oracle action. 4000 worlds per scenario, seed 20261002.

## What would count against the typed rule (declared now)
- If, in confounded scenarios (pi_conf >= 0.5), typed is not better than always MORE and always LARGER on O1.
- If typed is not at least as good as the interval-learned policy in some scenario.
- Expected (and to be reported, not hidden): with pi_conf = 0 the typed rule should LOSE to MORE/LARGER, because
  identification studies are less precise and there is no bias to remove.
