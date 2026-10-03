import NeutroEvidence.Credal

/-!
# Retraction of neutrosophic triples to Subjective Logic (book Theorem 5, v13.1)

* (a) `thm5_a` — a normalised binomial triple `(T, I, F)` is an opinion `(b, d, u) = (T, F, I)`;
  the orbit `{b + a u : a ∈ [0,1]}` of the projected probability is `[T, 1 - F]`.
* (b) `thm5_b_*` — the evidential triple of evidence `(r, q)` and weight `W` is the opinion from
  evidence and its interval is the binomial IDM of strength `W` (Corollary 2 with `k = 2`).
* (c) `thm5_c_chart`, `thm5_c_bijOn`, `thm5_c_zone` — the saturating chart, the retraction `ρ`
  as a bijection from `[0,1)²` onto the non-dogmatic opinions, and `T_s + F_s > 1 ↔ r q > W²`.
* (d) `thm5_d_*` — comparison of the evidential interval with the betting interval.
* `thm3_single_event` — betting reading on one event: empty interval iff `T + F > 1`.
-/

namespace NeutroEvidence

/-- Non-negativity and normalisation of an opinion `(b, d, u)` (base rate kept separate). -/
def IsOpinion (o : ℝ × ℝ × ℝ) : Prop :=
  0 ≤ o.1 ∧ 0 ≤ o.2.1 ∧ 0 ≤ o.2.2 ∧ o.1 + o.2.1 + o.2.2 = 1

/-- Non-dogmatic opinions: `u > 0`. -/
def NonDogmatic : Set (ℝ × ℝ × ℝ) :=
  {o | 0 ≤ o.1 ∧ 0 ≤ o.2.1 ∧ 0 < o.2.2 ∧ o.1 + o.2.1 + o.2.2 = 1}

/-- Projected probability `b + a u` of Subjective Logic. -/
def projected (b u a : ℝ) : ℝ := b + a * u

/-! ## Theorem 5(a) -/

/-- The opinion of a normalised binomial triple: `(b, d, u) = (T, F, I)`. -/
theorem thm5_a_opinion {T I F : ℝ} (hT : 0 ≤ T) (hI : 0 ≤ I) (hF : 0 ≤ F)
    (hN : T + I + F = 1) : IsOpinion (T, F, I) :=
  ⟨hT, hF, hI, by simp only; linarith⟩

/-- **Theorem 5(a).** For a normalised triple with `I ≥ 0`, the orbit of the projected
probability over all base rates is the betting interval: `{T + a I : a ∈ [0,1]} = [T, 1 - F]`. -/
theorem thm5_a {T I F : ℝ} (hI : 0 ≤ I) (hN : T + I + F = 1) :
    (projected T I) '' Set.Icc 0 1 = Set.Icc T (1 - F) := by
  have hU : 1 - F = T + I := by linarith
  rw [hU]
  ext x
  simp only [Set.mem_image, Set.mem_Icc, projected]
  constructor
  · rintro ⟨a, ⟨h0, h1⟩, rfl⟩
    exact ⟨by nlinarith, by nlinarith⟩
  · rintro ⟨h0, h1⟩
    rcases hI.eq_or_lt with hI0 | hIpos
    · subst hI0
      exact ⟨0, ⟨le_rfl, zero_le_one⟩, by linarith⟩
    · refine ⟨(x - T) / I, ⟨div_nonneg (by linarith) hIpos.le, ?_⟩, ?_⟩
      · rw [div_le_one hIpos]; linarith
      · rw [div_mul_cancel₀ _ hIpos.ne']; ring

/-- The projected probability for any base rate lies in the betting interval. -/
theorem thm5_a_selection {T I F a : ℝ} (hI : 0 ≤ I) (hN : T + I + F = 1)
    (ha0 : 0 ≤ a) (ha1 : a ≤ 1) : projected T I a ∈ Set.Icc T (1 - F) := by
  rw [← thm5_a hI hN]; exact ⟨a, ⟨ha0, ha1⟩, rfl⟩

/-! ## Theorem 5(b) -/

/-- Opinion from evidence `(r, q)` with prior weight `W`. -/
noncomputable def slOfEvidence (r q W : ℝ) : ℝ × ℝ × ℝ :=
  (r / (r + q + W), q / (r + q + W), W / (r + q + W))

/-- Evidential triple `(T, I, F)` of evidence `(r, q)` with weight `W`. -/
noncomputable def evTriple (r q W : ℝ) : ℝ × ℝ × ℝ :=
  (r / (r + q + W), W / (r + q + W), q / (r + q + W))

/-- **Theorem 5(b), opinion.** The evidential triple is normalised and, read as
`(b, d, u) = (T, F, I)`, it is the opinion from evidence. -/
theorem thm5_b_opinion {r q W : ℝ} (hr : 0 ≤ r) (hq : 0 ≤ q) (hW : 0 < W) :
    (evTriple r q W).1 + (evTriple r q W).2.1 + (evTriple r q W).2.2 = 1 ∧
      ((evTriple r q W).1, (evTriple r q W).2.2, (evTriple r q W).2.1) = slOfEvidence r q W ∧
      IsOpinion (slOfEvidence r q W) := by
  have hpos : 0 < r + q + W := by linarith
  refine ⟨?_, rfl, ?_, ?_, ?_, ?_⟩
  · simp only [evTriple]; field_simp; ring
  · exact div_nonneg hr hpos.le
  · exact div_nonneg hq hpos.le
  · exact div_nonneg hW.le hpos.le
  · simp only [slOfEvidence]; field_simp

/-- Counts on the binary frame: `r` for `true`, `q` for `false`. -/
def binCounts (r q : ℝ) : Bool → ℝ := fun b => if b then r else q

/-- **Theorem 5(b), IDM.** The betting interval `[T, 1 - F]` of the evidential triple is the
binomial imprecise Dirichlet model of strength `W`: its endpoints are the minimum and the
maximum of `P(true)` over the IDM credal set (Corollary 2 with `k = 2`, `s = W`). -/
theorem thm5_b_idm {r q W : ℝ} (hr : 0 ≤ r) (hq : 0 ≤ q) (hW : 0 < W) :
    IsLeast ((fun P => prob P {true}) '' idmCredal (binCounts r q) W) (evTriple r q W).1 ∧
      IsGreatest ((fun P => prob P {true}) '' idmCredal (binCounts r q) W)
        (1 - (evTriple r q W).2.2) := by
  have hn : ∀ x, 0 ≤ binCounts r q x := by intro x; cases x <;> simp [binCounts, hr, hq]
  have hN : ∑ y, binCounts r q y = r + q := by simp [binCounts, add_comm]
  have hpos : 0 < r + q + W := by linarith
  have hne : ({true} : Finset Bool) ≠ Finset.univ := by decide
  have h1 := idm_isLeast hn hW hne
  have h2 := idm_isGreatest hn hW (Finset.singleton_nonempty true)
  simp only [Finset.sum_singleton, binCounts, if_true, hN] at h1 h2
  refine ⟨?_, ?_⟩
  · simpa [evTriple, binCounts, hN] using h1
  · have : 1 - q / (r + q + W) = (r + W) / (r + q + W) := by field_simp; ring
    simp only [evTriple, this]
    simpa [binCounts, hN] using h2

/-! ## Theorem 5(c) -/

/-- The retraction `ρ(T, F) = (T(1-F), F(1-T), (1-T)(1-F)) / (1 - T F)`. -/
noncomputable def rho (x : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (x.1 * (1 - x.2) / (1 - x.1 * x.2), x.2 * (1 - x.1) / (1 - x.1 * x.2),
    (1 - x.1) * (1 - x.2) / (1 - x.1 * x.2))

/-- Inverse of `ρ` on non-dogmatic opinions: `(b, d, u) ↦ (b / (b + u), d / (d + u))`. -/
noncomputable def rhoInv (o : ℝ × ℝ × ℝ) : ℝ × ℝ :=
  (o.1 / (o.1 + o.2.2), o.2.1 / (o.2.1 + o.2.2))

/-- The open unit square `[0,1)²` of the saturating chart. -/
def chartSquare : Set (ℝ × ℝ) := Set.Ico 0 1 ×ˢ Set.Ico 0 1

lemma one_sub_mul_pos {T F : ℝ} (hT : T ∈ Set.Ico (0 : ℝ) 1) (hF : F ∈ Set.Ico (0 : ℝ) 1) :
    0 < 1 - T * F := by
  obtain ⟨hT0, hT1⟩ := hT
  obtain ⟨hF0, hF1⟩ := hF
  nlinarith [mul_le_mul_of_nonneg_left hF1.le hT0]

/-- **Theorem 5(c), chart.** With `T_s = r/(r+W)` and `F_s = q/(q+W)`, the retraction gives the
opinion from evidence. -/
theorem thm5_c_chart {r q W : ℝ} (hr : 0 ≤ r) (hq : 0 ≤ q) (hW : 0 < W) :
    rho (r / (r + W), q / (q + W)) = slOfEvidence r q W := by
  have h1 : r + W ≠ 0 := by positivity
  have h2 : q + W ≠ 0 := by positivity
  have h3 : r + q + W ≠ 0 := by positivity
  have hD : 1 - r / (r + W) * (q / (q + W)) = W * (r + q + W) / ((r + W) * (q + W)) := by
    field_simp; ring
  have h1T : 1 - r / (r + W) = W / (r + W) := by field_simp; ring
  have h1F : 1 - q / (q + W) = W / (q + W) := by field_simp; ring
  simp only [rho, slOfEvidence, hD, h1T, h1F]
  refine Prod.ext ?_ (Prod.ext ?_ ?_) <;> simp only <;> field_simp <;> ring

/-- The chart takes values in `[0,1)²`. -/
theorem thm5_c_chart_mem {r q W : ℝ} (hr : 0 ≤ r) (hq : 0 ≤ q) (hW : 0 < W) :
    (r / (r + W), q / (q + W)) ∈ chartSquare := by
  refine ⟨⟨div_nonneg hr (by linarith), ?_⟩, ⟨div_nonneg hq (by linarith), ?_⟩⟩
  · rw [div_lt_one (by linarith)]; linarith
  · rw [div_lt_one (by linarith)]; linarith

/-- **Theorem 5(c), bijection.** `ρ` is a bijection from `[0,1)²` onto the non-dogmatic
opinions, with inverse `rhoInv`. -/
theorem thm5_c_bijOn : Set.BijOn rho chartSquare NonDogmatic := by
  refine Set.InvOn.bijOn (f' := rhoInv) ⟨?_, ?_⟩ ?_ ?_
  · -- left inverse on the square
    rintro ⟨T, F⟩ ⟨hT, hF⟩
    have hD := one_sub_mul_pos hT hF
    obtain ⟨hT0, hT1⟩ := hT
    obtain ⟨hF0, hF1⟩ := hF
    have h1F : 0 < 1 - F := by linarith
    have h1T : 0 < 1 - T := by linarith
    simp only [rho, rhoInv]
    refine Prod.ext ?_ ?_ <;> simp only
    · rw [← add_div, div_div_div_cancel_right₀ hD.ne']
      have : T * (1 - F) + (1 - T) * (1 - F) = 1 - F := by ring
      rw [this]; field_simp
    · rw [← add_div, div_div_div_cancel_right₀ hD.ne']
      have : F * (1 - T) + (1 - T) * (1 - F) = 1 - T := by ring
      rw [this]; field_simp
  · -- right inverse on the non-dogmatic opinions
    rintro ⟨b, d, u⟩ ⟨hb, hd, hu, hs⟩
    simp only at hb hd hu hs
    have hbu : 0 < b + u := by linarith
    have hdu : 0 < d + u := by linarith
    have hD : 1 - b / (b + u) * (d / (d + u)) = u / ((b + u) * (d + u)) := by
      field_simp
      linear_combination (-(u : ℝ)) * hs
    have h1T : 1 - b / (b + u) = u / (b + u) := by field_simp; ring
    have h1F : 1 - d / (d + u) = u / (d + u) := by field_simp; ring
    simp only [rho, rhoInv, hD, h1T, h1F]
    refine Prod.ext ?_ (Prod.ext ?_ ?_) <;> simp only <;> field_simp <;> ring
  · -- ρ maps the square into the non-dogmatic opinions
    rintro ⟨T, F⟩ ⟨hT, hF⟩
    have hD := one_sub_mul_pos hT hF
    obtain ⟨hT0, hT1⟩ := hT
    obtain ⟨hF0, hF1⟩ := hF
    refine ⟨?_, ?_, ?_, ?_⟩ <;> simp only [rho]
    · exact div_nonneg (mul_nonneg hT0 (by linarith)) hD.le
    · exact div_nonneg (mul_nonneg hF0 (by linarith)) hD.le
    · exact div_pos (mul_pos (by linarith) (by linarith)) hD
    · rw [← add_div, ← add_div, div_eq_one_iff_eq hD.ne']; ring
  · -- the inverse maps the non-dogmatic opinions into the square
    rintro ⟨b, d, u⟩ ⟨hb, hd, hu, hs⟩
    simp only at hb hd hu hs
    refine ⟨⟨div_nonneg hb (by linarith), ?_⟩, ⟨div_nonneg hd (by linarith), ?_⟩⟩
    · simp only [rhoInv]; rw [div_lt_one (by linarith)]; linarith
    · simp only [rhoInv]; rw [div_lt_one (by linarith)]; linarith

/-- **Theorem 5(c), zone.** `T_s + F_s > 1 ↔ r q > W²`. -/
theorem thm5_c_zone {r q W : ℝ} (hr : 0 ≤ r) (hq : 0 ≤ q) (hW : 0 < W) :
    1 < r / (r + W) + q / (q + W) ↔ W ^ 2 < r * q := by
  have h1 : 0 < r + W := by linarith
  have h2 : 0 < q + W := by linarith
  rw [div_add_div _ _ h1.ne' h2.ne', one_lt_div (mul_pos h1 h2)]
  constructor <;> intro h <;> nlinarith

/-! ## Theorem 3 (single event) and Theorem 5(d) -/

/-- **Theorem 3, single event.** Under (D) the betting interval `[T, 1 - F]` of one event is
empty iff `T + F > 1` (sure loss). -/
theorem thm3_single_event (T F : ℝ) : Set.Icc T (1 - F) = ∅ ↔ 1 < T + F := by
  rw [Set.Icc_eq_empty_iff, not_le]
  constructor <;> intro h <;> linarith

/-- **Theorem 5(d), zone.** If `T_s + F_s > 1` the betting interval is empty while the
evidential interval `[b, b + u]` is non-empty. -/
theorem thm5_d_zone {T F : ℝ} (hT : T ∈ Set.Ico (0 : ℝ) 1) (hF : F ∈ Set.Ico (0 : ℝ) 1)
    (h : 1 < T + F) :
    Set.Icc T (1 - F) = ∅ ∧ (Set.Icc (rho (T, F)).1 ((rho (T, F)).1 + (rho (T, F)).2.2)).Nonempty := by
  refine ⟨(thm3_single_event T F).mpr h, ?_⟩
  have hu := (thm5_c_bijOn.mapsTo (show (T, F) ∈ chartSquare from ⟨hT, hF⟩)).2.2.1
  exact Set.nonempty_Icc.mpr (by linarith)

/-- **Theorem 5(d), comparison.** On `[0,1)²`: `b ≤ T_s`, `b + u = 1 - d ≥ 1 - F_s`. -/
theorem thm5_d_compare {T F : ℝ} (hT : T ∈ Set.Ico (0 : ℝ) 1) (hF : F ∈ Set.Ico (0 : ℝ) 1) :
    (rho (T, F)).1 ≤ T ∧ 1 - F ≤ (rho (T, F)).1 + (rho (T, F)).2.2 ∧
      (rho (T, F)).1 + (rho (T, F)).2.2 = 1 - (rho (T, F)).2.1 := by
  have hD := one_sub_mul_pos hT hF
  have hs := (thm5_c_bijOn.mapsTo (show (T, F) ∈ chartSquare from ⟨hT, hF⟩)).2.2.2
  obtain ⟨hT0, hT1⟩ := hT
  obtain ⟨hF0, hF1⟩ := hF
  refine ⟨?_, ?_, by linarith⟩ <;> simp only [rho]
  · rw [div_le_iff₀ hD]
    nlinarith [mul_nonneg (mul_nonneg hT0 hF0) (by linarith : (0 : ℝ) ≤ 1 - T)]
  · rw [← add_div, le_div_iff₀ hD]
    nlinarith [mul_nonneg (mul_nonneg hT0 hF0) (by linarith : (0 : ℝ) ≤ 1 - F)]

/-- **Theorem 5(d), equality case.** `b = T_s` and `b + u = 1 - F_s` hold iff `T_s F_s = 0`. -/
theorem thm5_d_equality {T F : ℝ} (hT : T ∈ Set.Ico (0 : ℝ) 1) (hF : F ∈ Set.Ico (0 : ℝ) 1) :
    ((rho (T, F)).1 = T ∧ (rho (T, F)).1 + (rho (T, F)).2.2 = 1 - F) ↔ T * F = 0 := by
  have hD := one_sub_mul_pos hT hF
  obtain ⟨hT0, hT1⟩ := hT
  obtain ⟨hF0, hF1⟩ := hF
  simp only [rho]
  rw [← add_div, div_eq_iff hD.ne', div_eq_iff hD.ne']
  constructor
  · rintro ⟨h1, -⟩
    have : T * F * (1 - T) = 0 := by linear_combination (-1 : ℝ) * h1
    rcases mul_eq_zero.mp this with h | h
    · exact h
    · exact absurd h (by linarith)
  · intro h
    constructor
    · linear_combination (-T) * h
    · linear_combination (F - 1) * h

/-- **Theorem 5(d), example.** For `(T, I, F) = (0.2, 0.5, 0.3)` the betting interval is
`[0.2, 0.7]`, while the chart applied to `(T_s, F_s) = (0.2, 0.3)` gives `[7/47, 35/47]`
`≈ [0.1489, 0.7447]`. -/
theorem thm5_d_example :
    (rho (1 / 5, 3 / 10)).1 = 7 / 47 ∧ (rho (1 / 5, 3 / 10)).1 + (rho (1 / 5, 3 / 10)).2.2 = 35 / 47 := by
  simp only [rho]; norm_num

end NeutroEvidence
