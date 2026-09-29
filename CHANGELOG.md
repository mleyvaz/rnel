# Changelog

## 0.2.0 — unreleased
- **New experimental module `rnel.off`: signed evidence and over/under/off values.** `EvidenceLedger` records reports
  and retractions per source (`report`, `retract`, `retract_source`, the unfusion of Subjective Logic) and sources
  known to invert their reports (`invert`); an over-retraction is rejected or returned as an `OffOpinion` flagged as
  lying outside the Beta region (r > -aW and s > -(1-a)W). The module also implements the over/under/off operators
  of Smarandache and Leyva-Vázquez (offunion, offintersection, offcomplement, scaled conjunction and disjunction,
  typed discounting with a real trust value, the off-RNEL tuple and the retraction map).
- **New in `rnel.operators`: base-rate-calibrated conjunction and disjunction** (`calibrated_and`, `calibrated_or`),
  Section 5.2 of Smarandache and Leyva-Vázquez. They add to any priority-product operator ((T, I, N, F),
  multi-uncertainty or RNEL) the transfer of mass from indeterminacy to T (or F) that reproduces SL
  multiplication (comultiplication) for arbitrary base rates. Tests: `tests/test_calibrated.py` (equality with SL
  for random base rates in the three families, De Morgan with complemented base rates, reduction to the
  uncalibrated operators at the extreme base rates).
- `scripts/check_off_extensions.py`: numerical checks of the off-extension theorems and worked applications.
- Tests: `tests/test_off.py`.

## 0.1.1 — 2026-09-27
- **Fix (`rnel.decide.Policy`): the evidence gate is now monotone.** In 0.1.0 the policy abstained when the ignorance
  component G was at or above its threshold and otherwise answered with the leading side. Since G falls whenever any
  report arrives, a report *supporting* x could unlock the answer "refuted": with the default policy, 3 opposing
  reports gave "abstain" but 3 opposing + 1 supporting gave "refuted" (and symmetrically). This violates the AGM
  inclusion postulate. The gate now applies to the asserted side: answering T (F) requires more supporting (opposing)
  reports than the count at which the old total-mass gate opened, W(1/G − 1). With one-sided evidence both gates agree;
  on all states with up to 29 reports per side only (1,3) and (3,1) change, from an answer to "abstain".
  `Policy(side_gate=False)` reproduces 0.1.0.
- Tests: `tests/test_decide_monotone.py` (legacy failure reproduced; monotonicity for 18 threshold settings; agreement
  with 0.1.0 on one-sided evidence; fused-sources case; README example unchanged).

## 0.1.0 — 2026-09-26
- First release.
