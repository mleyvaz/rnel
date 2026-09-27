# Changelog

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
