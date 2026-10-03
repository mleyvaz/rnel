# Changelog

## 0.3.0 — 2026-10-03 (local release candidate; not published)
- Consolidates the modules used by *Neutrosophic Evidence*: credal and neutrosophic-credal tools,
  order-free conflict diagnostics, refined neutrosophic statistics, RNEL-MVC and active learning,
  DSmT, copula N-norms, plithogenic operators, and base-rate score functions.
- Adds the explicit affine bridge `I = (1 + eps) / 2`, theorem-labelled tests, and a reproducibility
  runner (`python -m rnel.book regenerate --check`) using only repository-relative paths.
- Adds Python 3.10–3.13 test CI and packaging checks. Publication, tag creation, PyPI upload, and the
  new Zenodo version remain manual owner actions.

## Unreleased (branch feat/dsmt, local only)
- **New experimental module `rnel.dsmt`: Dezert-Smarandache theory on small frames** (standard library only).
  Frames under the free DSm model (hyper-power set, Venn-region codification), Shafer's model and hybrid models
  (`Frame(atoms, model, empty=[...])`, string syntax `"A|B"`, `"A&B"`); rules `conjunctive` (= classic DSm rule on
  the free model, = Smets/TBM with conflict on the empty set otherwise; aliases `dsmc`, `tbm`), `dempster`,
  `yager`, `dubois_prade`, `pcr5` (Smarandache-Dezert general s-source form, canonical-form members, product
  weights), `pcr6` (Martin-Osswald, sum weights), `pcr6_plus` (Dezert, Dezert and Smarandache 2021), `sequential`,
  `combine_all`; decision `betp` (generalised pignistic, DSm cardinality), `dsmp` (DSmP_epsilon), `belief`,
  `plausibility`; the binary-frame dictionary with RNEL: `tuple_to_bba`/`bba_to_tuple` (m(x)=T, m(not x)=F,
  m(x&not x)=C, m(x|not x)=U+G, m(empty)=N), `reports_to_bba`, `opinion_to_bba`, and the credal reading
  `bba_to_triple`/`triple_to_bba`/`belnap_probability` (free binary masses = probabilities on the Belnap frame).
- Tests: `tests/test_dsmt.py` (published examples of Smarandache-Dezert 2005/2006, Martin-Osswald 2006,
  Dezert-Smarandache 2008, Dezert-Dezert-Smarandache 2021, Zadeh's example; properties; optional cross-check with
  evidencelib). Example: `examples/dsmt_demo.py`.

## Unreleased (branch feat/neutro-credal-tools, local only)
- **New experimental module `rnel.neutro_credal`: credal tools for neutrosophic triples** (needs numpy + scipy, extra
  `rnel[credal]`). Under the betting reading T <= P(A) <= 1 - F the triple inherits Walley's checks and tools:
  `avoids_sure_loss`, `sure_loss_degree` (closed form (T + F - 1)/2, Theorem 3), `is_coherent`/`classify` (LP, and the
  closed form of Theorem 2 on singletons), `natural_extension` of composite events as triples, `coherent_correction`
  (I^E <= I, Theorem 1(c)), `singleton_intervals_check`, `singleton_natural_extension`, `singleton_correction`,
  `idm_triple` (Corollary 2), decisions by sets (`interval_dominance`, `maximality`, `e_admissible`), the glut
  lifting `to_glut_frame`/`GlutLifting` (Theorem 8; minimal, Belnap and disjoint frames), `glut_conjunction`
  (N-norms as natural extensions, Theorem 9), `sl_retraction` (Theorem 5) and `credal_diagnosis`, which combines the
  credal checks, natural extension and decision with the RNEL state (a, b, u, c) and C* from evidence per source.
- Tests: `tests/test_neutro_credal.py`; example: `examples/neutro_credal_demo.py`.

## Unreleased (branch feat/credal, local only)
- **New experimental module `rnel.credal`** (needs numpy + scikit-learn, extra `rnel[credal]`).
  `CredalEnsemble` wraps a scikit-learn ensemble (RandomForest, 100 trees, `min_samples_leaf=5` by default) and
  returns per-class lower/upper probabilities over the members (`envelope`: hull, alpha-trimmed as in Nguyen,
  Zhang & Destercke, ECSQARU 2023, doi:10.1007/978-3-031-45608-4_21, or per-class quantiles), plus evidence
  counts (`leaf`, `votes`, `mean`). `to_neutrosophic`/`from_neutrosophic` (T = l, I = u - l, F = 1 - u) are a
  change of representation only. Information beyond the credal set comes from per-view evidence:
  `sup_gap_views`, `c_star_multiclass` (sup-norm extension of C*), `c_star_one_vs_rest`, `evidential_state`
  (a, b, u, c with T = a + c, F = b + c, I = u + c) and `decide` (answer / abstain_seek_data / review_sources /
  reject).
- Tests: `tests/test_credal.py`.

## Unreleased (branch exp/conflict-first-principles, local only)
- **New experimental module `rnel.conflict`: between-source conflict from first principles.** `k_between`
  (K_b = min(M+, M-), the unique measure satisfying ally consolidation, unanimity null and opposed-pair
  additivity), `k_within`, `credal_gap`, multinomial `sup_gap`, the additive state `ConflictState` (R, S, K_w)
  whose fusion is order-free, the normalised component C* = 2 K_b / (R + S + W), `state_from_tuple`
  (identifiability), provenance-aware fusion (`group_counts`, `dependence_interval`, `atom_union`) and the two
  forms of Definition 8.4 for comparison (`def84_sequential`, `def84_maxpair`). Definition 8.4 itself is unchanged.
- Tests: `tests/test_conflict.py`.

## 0.2.0 — 2026-09-29
- Archived release: https://doi.org/10.5281/zenodo.23040628.
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
