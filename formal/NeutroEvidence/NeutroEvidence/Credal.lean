import Mathlib

/-!
# Credal reduction of normalised dual neutrosophic probabilities

Finite frame `α`, events are `Finset α`, a probability is a mass function `P : α → ℝ`.

* `NP` — a neutrosophic probability: one triple `(T A, I A, F A)` per event.
* Axioms used only as explicit hypotheses: `Normalised` (N), `Dual` (D), `Boundary` (B),
  `Standard` (components non-negative).
* Betting reading: credal set `{P | T A ≤ P(A) ≤ 1 - F A for every A}`.
* Avoiding sure loss = non-empty credal set (Walley's characterisation on a finite frame,
  taken here as the definition); coherence = non-empty credal set and every lower and every
  upper bound attained by a member of the credal set.

Main results (book numbering, v13.1):
* `NP.reductionEquiv`, `NP.I_eq_upper_sub_lower` — Theorem 1, bijection and `I = U - L`.
* `NP.thm1_a`, `NP.thm1_b`, `NP.thm1_c_*`, `NP.thm1_d_*`, `NP.thm1_e` — Theorem 1 (a)–(e).
* `idm_isLeast`, `idm_isGreatest`, `idm_coherent` — Corollary 2 (imprecise Dirichlet model).
-/

open Finset

set_option linter.unusedSectionVars false

namespace NeutroEvidence

variable {α : Type*} [Fintype α] [DecidableEq α]

/-! ## Probabilities on a finite frame -/

/-- `P` is a probability mass function. -/
def IsProb (P : α → ℝ) : Prop := (∀ x, 0 ≤ P x) ∧ ∑ x, P x = 1

/-- Probability of an event. -/
def prob (P : α → ℝ) (A : Finset α) : ℝ := ∑ x ∈ A, P x

lemma prob_nonneg {P : α → ℝ} (hP : IsProb P) (A : Finset α) : 0 ≤ prob P A :=
  Finset.sum_nonneg fun x _ => hP.1 x

lemma prob_univ {P : α → ℝ} (hP : IsProb P) : prob P univ = 1 := hP.2

lemma prob_empty (P : α → ℝ) : prob P ∅ = 0 := by simp [prob]

lemma prob_compl {P : α → ℝ} (hP : IsProb P) (A : Finset α) :
    prob P Aᶜ = 1 - prob P A := by
  have h := Finset.sum_add_sum_compl A P
  rw [hP.2] at h
  unfold prob; linarith

lemma continuous_prob (A : Finset α) : Continuous (fun P : α → ℝ => prob P A) :=
  continuous_finsetSum A fun i _ => continuous_apply i

/-! ## Credal sets given by lower and upper bounds on events -/

/-- Credal set of a system of bounds `lo A ≤ P(A) ≤ hi A` on all events. -/
def boxCredal (lo hi : Finset α → ℝ) : Set (α → ℝ) :=
  {P | IsProb P ∧ ∀ A, lo A ≤ prob P A ∧ prob P A ≤ hi A}

/-- Coherence of a system of bounds: non-empty credal set and every bound attained. -/
def BoxCoherent (lo hi : Finset α → ℝ) : Prop :=
  (boxCredal lo hi).Nonempty ∧
    ∀ A, (∃ P ∈ boxCredal lo hi, prob P A = lo A) ∧ (∃ P ∈ boxCredal lo hi, prob P A = hi A)

lemma isClosed_boxCredal (lo hi : Finset α → ℝ) : IsClosed (boxCredal lo hi) := by
  have h1 : IsClosed {P : α → ℝ | ∀ x, 0 ≤ P x} := by
    simp only [Set.ofPred_forall]
    exact isClosed_iInter fun x => isClosed_le continuous_const (continuous_apply x)
  have h2 : IsClosed {P : α → ℝ | ∑ x, P x = 1} :=
    isClosed_eq (continuous_finsetSum _ fun i _ => continuous_apply i) continuous_const
  have h3 : IsClosed {P : α → ℝ | ∀ A, lo A ≤ prob P A ∧ prob P A ≤ hi A} := by
    simp only [Set.ofPred_forall, Set.ofPred_and]
    exact isClosed_iInter fun A =>
      (isClosed_le continuous_const (continuous_prob A)).inter
        (isClosed_le (continuous_prob A) continuous_const)
  have : boxCredal lo hi =
      ({P : α → ℝ | ∀ x, 0 ≤ P x} ∩ {P | ∑ x, P x = 1}) ∩
        {P | ∀ A, lo A ≤ prob P A ∧ prob P A ≤ hi A} := by
    ext P; simp [boxCredal, IsProb, and_assoc]
  rw [this]; exact (h1.inter h2).inter h3

lemma isCompact_boxCredal (lo hi : Finset α → ℝ) : IsCompact (boxCredal lo hi) := by
  apply Metric.isCompact_of_isClosed_isBounded (isClosed_boxCredal lo hi)
  rw [Metric.isBounded_iff_subset_closedBall 0]
  refine ⟨1, fun P hP => ?_⟩
  rw [mem_closedBall_zero_iff, pi_norm_le_iff_of_nonneg zero_le_one]
  intro i
  rw [Real.norm_eq_abs, abs_le]
  have hle : P i ≤ ∑ j, P j :=
    Finset.single_le_sum (fun j _ => hP.1.1 j) (Finset.mem_univ i)
  rw [hP.1.2] at hle
  exact ⟨by linarith [hP.1.1 i], hle⟩

/-- On a finite frame the lower envelope over a non-empty credal set is attained. -/
lemma exists_min_prob (lo hi : Finset α → ℝ) (hne : (boxCredal lo hi).Nonempty)
    (A : Finset α) :
    ∃ P ∈ boxCredal lo hi, ∀ Q ∈ boxCredal lo hi, prob P A ≤ prob Q A := by
  obtain ⟨P, hP, hmin⟩ :=
    (isCompact_boxCredal lo hi).exists_isMinOn hne (continuous_prob A).continuousOn
  exact ⟨P, hP, fun Q hQ => isMinOn_iff.mp hmin Q hQ⟩

/-! ## Neutrosophic probabilities -/

/-- A neutrosophic probability on the events of `α`: a triple per event, no axiom imposed. -/
@[ext]
structure NP (α : Type*) where
  T : Finset α → ℝ
  I : Finset α → ℝ
  F : Finset α → ℝ

namespace NP

variable (μ : NP α)

/-- (N) Normalisation. -/
def Normalised : Prop := ∀ A, μ.T A + μ.I A + μ.F A = 1

/-- (D) Complement duality: `NP(Aᶜ) = (F A, I A, T A)`. -/
def Dual : Prop := ∀ A, μ.T Aᶜ = μ.F A ∧ μ.I Aᶜ = μ.I A ∧ μ.F Aᶜ = μ.T A

/-- (B) Boundary: `NP(Ω) = (1,0,0)`, `NP(∅) = (0,0,1)`. -/
def Boundary : Prop :=
  (μ.T univ = 1 ∧ μ.I univ = 0 ∧ μ.F univ = 0) ∧ (μ.T ∅ = 0 ∧ μ.I ∅ = 0 ∧ μ.F ∅ = 1)

/-- Standard (non-negative) components. With (N) they lie in `[0,1]`. -/
def Standard : Prop := ∀ A, 0 ≤ μ.T A ∧ 0 ≤ μ.I A ∧ 0 ≤ μ.F A

/-- Credal set of the betting reading: `T A ≤ P(A) ≤ 1 - F A`. -/
def credal : Set (α → ℝ) := boxCredal μ.T (fun A => 1 - μ.F A)

/-- Avoiding sure loss (finite-frame characterisation: non-empty credal set). -/
def AvoidsSureLoss : Prop := μ.credal.Nonempty

/-- Coherence: every lower bound `T A` and every upper bound `1 - F A` is attained. -/
def Coherent : Prop := BoxCoherent μ.T (fun A => 1 - μ.F A)

end NP

/-! ## Conjugate lower/upper probabilities -/

/-- Conjugate upper probability `U A = 1 - L Aᶜ`. -/
def upper (L : Finset α → ℝ) (A : Finset α) : ℝ := 1 - L Aᶜ

/-- Lower probabilities (with conjugate upper) admitted by Theorem 1:
`L ∅ = 0`, `L Ω = 1`, `L ≥ 0`, `L A + L Aᶜ ≤ 1` (i.e. `L ≤ U`). -/
def IsConjLower (L : Finset α → ℝ) : Prop :=
  L ∅ = 0 ∧ L univ = 1 ∧ (∀ A, 0 ≤ L A) ∧ ∀ A, L A + L Aᶜ ≤ 1

/-- Credal set of a conjugate pair `(L, U)`. -/
def credalL (L : Finset α → ℝ) : Set (α → ℝ) := boxCredal L (upper L)

/-- Walley coherence of the conjugate pair `(L, U)`. -/
def CoherentL (L : Finset α → ℝ) : Prop := BoxCoherent L (upper L)

/-- The upper constraint of a conjugate pair is redundant. -/
lemma credalL_eq (L : Finset α → ℝ) :
    credalL L = {P | IsProb P ∧ ∀ A, L A ≤ prob P A} := by
  ext P
  simp only [credalL, boxCredal, upper, Set.mem_ofPred_eq]
  constructor
  · rintro ⟨hP, h⟩; exact ⟨hP, fun A => (h A).1⟩
  · rintro ⟨hP, h⟩
    refine ⟨hP, fun A => ⟨h A, ?_⟩⟩
    have := h Aᶜ; rw [prob_compl hP] at this; linarith

/-- The neutrosophic probability determined by a lower probability:
`T = L`, `I = U - L`, `F A = L Aᶜ = 1 - U A`. -/
def NP.ofLower (L : Finset α → ℝ) : NP α :=
  ⟨L, fun A => upper L A - L A, fun A => L Aᶜ⟩

namespace NP

variable {μ : NP α}

/-! ## Theorem 1 — the bijection and `I = U - L` -/

/-- **Theorem 1 (reduction), identity part.** Under (N) and (D):
`T = L`, `F A = L Aᶜ = 1 - U A` and `I = U - L`, with `L = T`. -/
theorem I_eq_upper_sub_lower (hN : μ.Normalised) (hD : μ.Dual) (A : Finset α) :
    μ.I A = upper μ.T A - μ.T A ∧ μ.F A = μ.T Aᶜ ∧ μ.F A = 1 - upper μ.T A := by
  have h1 := hN A
  have h2 := (hD A).1
  unfold upper
  refine ⟨by rw [h2]; linarith, h2.symm, by rw [h2]; ring⟩

lemma ofLower_normalised (L : Finset α → ℝ) : (ofLower L).Normalised := by
  intro A; simp only [ofLower, upper]; ring

lemma ofLower_dual (L : Finset α → ℝ) : (ofLower L).Dual := by
  intro A
  refine ⟨?_, ?_, ?_⟩ <;> simp only [ofLower, upper, compl_compl] <;> ring

lemma ofLower_boundary {L : Finset α → ℝ} (hL : IsConjLower L) : (ofLower L).Boundary := by
  obtain ⟨h0, h1, -, -⟩ := hL
  simp only [Boundary, ofLower, upper, compl_univ, compl_empty, h0, h1]
  norm_num

lemma ofLower_standard {L : Finset α → ℝ} (hL : IsConjLower L) : (ofLower L).Standard := by
  obtain ⟨-, -, hpos, hsum⟩ := hL
  intro A
  simp only [ofLower, upper]
  exact ⟨hpos A, by linarith [hsum A], hpos Aᶜ⟩

lemma isConjLower_T (hN : μ.Normalised) (hD : μ.Dual) (hB : μ.Boundary)
    (hS : μ.Standard) : IsConjLower μ.T := by
  refine ⟨hB.2.1, hB.1.1, fun A => (hS A).1, fun A => ?_⟩
  rw [(hD A).1]; linarith [hN A, (hS A).2.1]

/-- **Theorem 1 (reduction), bijection.** `NP ↦ L = T` is a bijection between normalised,
dual, boundary, standard neutrosophic probabilities and lower probabilities `L` with
`L ∅ = 0`, `L Ω = 1`, `L ≥ 0`, `L A + L Aᶜ ≤ 1` (conjugate pairs `(L, U)`, `U A = 1 - L Aᶜ`).
The inverse is `NP.ofLower`: `T = L`, `F A = L Aᶜ = 1 - U A`, `I = U - L`. -/
def reductionEquiv :
    {μ : NP α // μ.Normalised ∧ μ.Dual ∧ μ.Boundary ∧ μ.Standard} ≃
      {L : Finset α → ℝ // IsConjLower L} where
  toFun μ := ⟨μ.1.T, isConjLower_T μ.2.1 μ.2.2.1 μ.2.2.2.1 μ.2.2.2.2⟩
  invFun L := ⟨ofLower L.1, ofLower_normalised L.1, ofLower_dual L.1,
    ofLower_boundary L.2, ofLower_standard L.2⟩
  left_inv μ := by
    obtain ⟨μ, hN, hD, -, -⟩ := μ
    apply Subtype.ext
    ext A
    · rfl
    · exact (I_eq_upper_sub_lower hN hD A).1.symm
    · exact (I_eq_upper_sub_lower hN hD A).2.1.symm
  right_inv L := rfl

/-- Under (D) the credal set of the betting reading is the credal set of `(L, U) = (T, 1 - F)`. -/
lemma credal_eq_credalL (hD : μ.Dual) : μ.credal = credalL μ.T := by
  have : (fun A => 1 - μ.F A) = upper μ.T := by
    funext A; simp only [upper, (hD A).1]
  simp only [credal, credalL, this]

/-- **Theorem 1(a).** Under (D): NP avoids sure loss iff the credal set of `(L, U)` is non-empty
(on a finite frame this is Walley's avoiding sure loss). -/
theorem thm1_a (hD : μ.Dual) : μ.AvoidsSureLoss ↔ (credalL μ.T).Nonempty := by
  rw [AvoidsSureLoss, credal_eq_credalL hD]

/-- **Theorem 1(b).** Under (D): NP is coherent iff `(L, U) = (T, 1 - F)` is coherent in
Walley's sense. -/
theorem thm1_b (hD : μ.Dual) : μ.Coherent ↔ CoherentL μ.T := by
  have : (fun A => 1 - μ.F A) = upper μ.T := by
    funext A; simp only [upper, (hD A).1]
  simp only [Coherent, CoherentL, this]

/-! ## Theorem 1(c) — the neutrosophic natural extension -/

/-- Natural extension `E(A) = min_{P ∈ M} P(A)` (as an infimum; attained, see below). -/
noncomputable def natExt (μ : NP α) (A : Finset α) : ℝ :=
  sInf ((fun P => prob P A) '' μ.credal)

/-- Neutrosophic natural extension `NP_E(A) = (E A, 1 - E A - E Aᶜ, E Aᶜ)`. -/
noncomputable def natExtNP (μ : NP α) : NP α :=
  ⟨μ.natExt, fun A => 1 - μ.natExt A - μ.natExt Aᶜ, fun A => μ.natExt Aᶜ⟩

lemma natExt_le {P : α → ℝ} (hP : P ∈ μ.credal) (A : Finset α) : μ.natExt A ≤ prob P A := by
  apply csInf_le
  · refine ⟨0, ?_⟩
    rintro _ ⟨Q, hQ, rfl⟩
    exact prob_nonneg hQ.1 A
  · exact ⟨P, hP, rfl⟩

lemma le_natExt (hne : μ.AvoidsSureLoss) (A : Finset α) : μ.T A ≤ μ.natExt A := by
  apply le_csInf (hne.image _)
  rintro _ ⟨Q, hQ, rfl⟩
  exact (hQ.2 A).1

lemma natExt_attained (hne : μ.AvoidsSureLoss) (A : Finset α) :
    ∃ P ∈ μ.credal, prob P A = μ.natExt A := by
  obtain ⟨P, hP, hmin⟩ := exists_min_prob _ _ hne A
  refine ⟨P, hP, ?_⟩
  symm
  apply IsLeast.csInf_eq
  exact ⟨⟨P, hP, rfl⟩, by rintro _ ⟨Q, hQ, rfl⟩; exact hmin Q hQ⟩

lemma natExt_nonneg (hne : μ.AvoidsSureLoss) (A : Finset α) : 0 ≤ μ.natExt A := by
  obtain ⟨P, hP, h⟩ := natExt_attained hne A
  rw [← h]; exact prob_nonneg hP.1 A

/-- Under (D), the natural extension has the same credal set. -/
lemma credal_natExtNP (hD : μ.Dual) (hne : μ.AvoidsSureLoss) :
    μ.natExtNP.credal = μ.credal := by
  ext P
  simp only [credal, boxCredal, natExtNP, Set.mem_ofPred_eq]
  constructor
  · rintro ⟨hP, h⟩
    refine ⟨hP, fun A => ⟨le_trans (le_natExt hne A) (h A).1, ?_⟩⟩
    have h1 := (h A).2
    have h2 := le_natExt hne Aᶜ
    rw [(hD A).1] at h2
    linarith
  · rintro ⟨hP, h⟩
    have hPc : P ∈ μ.credal := ⟨hP, h⟩
    refine ⟨hP, fun A => ⟨natExt_le hPc A, ?_⟩⟩
    have := natExt_le hPc Aᶜ
    rw [prob_compl hP] at this
    linarith

/-- **Theorem 1(c).** If NP is dual and avoids sure loss, its neutrosophic natural extension is
normalised, dual, boundary, standard and coherent. -/
theorem thm1_c_properties (hD : μ.Dual) (hne : μ.AvoidsSureLoss) :
    μ.natExtNP.Normalised ∧ μ.natExtNP.Dual ∧ μ.natExtNP.Boundary ∧
      μ.natExtNP.Standard ∧ μ.natExtNP.Coherent := by
  have hcred := credal_natExtNP hD hne
  -- values of `E` on `Ω` and `∅`
  have hEuniv : μ.natExt univ = 1 := by
    obtain ⟨P, hP, h⟩ := natExt_attained hne univ
    rw [← h, prob_univ hP.1]
  have hEempty : μ.natExt ∅ = 0 := by
    obtain ⟨P, hP, h⟩ := natExt_attained hne ∅
    rw [← h, prob_empty]
  have hsum : ∀ A, μ.natExt A + μ.natExt Aᶜ ≤ 1 := by
    intro A
    obtain ⟨P, hP, -⟩ := natExt_attained hne A
    have h1 := natExt_le hP A
    have h2 := natExt_le hP Aᶜ
    rw [prob_compl hP.1] at h2
    linarith
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro A; simp only [natExtNP]; ring
  · intro A
    refine ⟨?_, ?_, ?_⟩ <;> simp only [natExtNP, compl_compl] <;> ring
  · simp only [Boundary, natExtNP, compl_univ, compl_empty, hEuniv, hEempty]; norm_num
  · intro A
    simp only [natExtNP]
    exact ⟨natExt_nonneg hne A, by linarith [hsum A], natExt_nonneg hne Aᶜ⟩
  · refine ⟨?_, fun A => ⟨?_, ?_⟩⟩
    · rw [show boxCredal μ.natExtNP.T (fun A => 1 - μ.natExtNP.F A) = μ.natExtNP.credal
        from rfl, hcred]
      exact hne
    · obtain ⟨P, hP, h⟩ := natExt_attained hne A
      exact ⟨P, by rw [show boxCredal μ.natExtNP.T (fun A => 1 - μ.natExtNP.F A) =
        μ.natExtNP.credal from rfl, hcred]; exact hP, h⟩
    · obtain ⟨P, hP, h⟩ := natExt_attained hne Aᶜ
      refine ⟨P, by rw [show boxCredal μ.natExtNP.T (fun A => 1 - μ.natExtNP.F A) =
        μ.natExtNP.credal from rfl, hcred]; exact hP, ?_⟩
      simp only [natExtNP]
      rw [← h, prob_compl hP.1]; ring

/-- **Theorem 1(c), monotonicity.** Under (N), (D) and avoiding sure loss:
`T ≤ T_E`, `F ≤ F_E` and `I_E ≤ I` — correcting incoherence can only reduce indeterminacy. -/
theorem thm1_c_monotone (hN : μ.Normalised) (hD : μ.Dual) (hne : μ.AvoidsSureLoss)
    (A : Finset α) :
    μ.T A ≤ μ.natExtNP.T A ∧ μ.F A ≤ μ.natExtNP.F A ∧ μ.natExtNP.I A ≤ μ.I A := by
  have h1 := le_natExt hne A
  have h2 := le_natExt hne Aᶜ
  rw [(hD A).1] at h2
  refine ⟨h1, h2, ?_⟩
  simp only [natExtNP]
  linarith [hN A]

/-! ## Theorem 1(e) — two outcomes -/

lemma finset_bool_cases (A : Finset Bool) :
    A = ∅ ∨ A = {true} ∨ A = {false} ∨ A = univ := by
  revert A; decide

/-- The two-point probability with `P true = x`. -/
def bprob (x : ℝ) : Bool → ℝ := fun b => if b then x else 1 - x

lemma prob_bprob (x : ℝ) (A : Finset Bool) :
    prob (bprob x) A = (if true ∈ A then x else 0) + (if false ∈ A then 1 - x else 0) := by
  rcases finset_bool_cases A with rfl | rfl | rfl | rfl <;> simp [prob, bprob]

/-- **Theorem 1(e).** On a two-element frame every normalised, dual, boundary, standard NP is
coherent. -/
theorem thm1_e {μ : NP Bool} (hN : μ.Normalised) (hD : μ.Dual) (hB : μ.Boundary)
    (hS : μ.Standard) : μ.Coherent := by
  have hct : ({true} : Finset Bool)ᶜ = {false} := by decide
  have hcf : ({false} : Finset Bool)ᶜ = {true} := by decide
  set a := μ.T {true}
  set b := μ.T {false}
  have hFt : μ.F {true} = b := by rw [← (hD {true}).1, hct]
  have hFf : μ.F {false} = a := by rw [← (hD {false}).1, hcf]
  have hab : a + b ≤ 1 := by linarith [hN {true}, (hS {true}).2.1]
  have ha : 0 ≤ a := (hS {true}).1
  have hb : 0 ≤ b := (hS {false}).1
  obtain ⟨⟨hTu, -, hFu⟩, ⟨hTe, -, hFe⟩⟩ := hB
  -- every `bprob x` with `a ≤ x ≤ 1 - b` is in the credal set
  have hmem : ∀ x, a ≤ x → x ≤ 1 - b →
      bprob x ∈ boxCredal μ.T (fun A => 1 - μ.F A) := by
    intro x hx1 hx2
    refine ⟨⟨fun c => ?_, ?_⟩, fun A => ?_⟩
    · cases c <;> simp [bprob] <;> linarith
    · simp [bprob]
    · rw [prob_bprob]
      rcases finset_bool_cases A with rfl | rfl | rfl | rfl
      · beta_reduce; rw [hTe, hFe]; simp
      · simp [hFt]; exact ⟨hx1, by linarith⟩
      · simp [hFf]; exact ⟨by linarith, by linarith⟩
      · beta_reduce; rw [hTu, hFu]; simp
  have hlo := hmem a le_rfl (by linarith)
  have hhi := hmem (1 - b) (by linarith) le_rfl
  refine ⟨⟨_, hlo⟩, fun A => ?_⟩
  rcases finset_bool_cases A with rfl | rfl | rfl | rfl
  · exact ⟨⟨_, hlo, by rw [prob_empty, hTe]⟩,
      ⟨_, hlo, by beta_reduce; rw [prob_empty, hFe]; ring⟩⟩
  · exact ⟨⟨_, hlo, by rw [prob_bprob]; simp [a]⟩,
      ⟨_, hhi, by beta_reduce; rw [prob_bprob, hFt]; simp⟩⟩
  · exact ⟨⟨_, hhi, by rw [prob_bprob]; simp [b]⟩,
      ⟨_, hlo, by beta_reduce; rw [prob_bprob, hFf]; simp⟩⟩
  · exact ⟨⟨_, hlo, by rw [prob_univ (hlo.1), hTu]⟩,
      ⟨_, hlo, by beta_reduce; rw [prob_univ (hlo.1), hFu]; ring⟩⟩

/-! ## Theorem 1(d) — (N)+(D)+(B) imply neither avoiding sure loss nor coherence -/

lemma prob_fin3 (P : Fin 3 → ℝ) (A : Finset (Fin 3)) :
    prob P A = (if (0 : Fin 3) ∈ A then P 0 else 0) + (if (1 : Fin 3) ∈ A then P 1 else 0)
      + (if (2 : Fin 3) ∈ A then P 2 else 0) := by
  have h : ∑ i, (if i ∈ A then P i else 0) = ∑ x ∈ A, P x := by
    rw [Finset.sum_ite_mem, Finset.univ_inter]
  rw [prob, ← h, Fin.sum_univ_three]

/-- Lower probability of counterexample C1: `0.4` on singletons, `0.3` on pairs. -/
noncomputable def c1L (A : Finset (Fin 3)) : ℝ :=
  if A.card = 0 then 0 else if A.card = 1 then 2 / 5 else if A.card = 2 then 3 / 10 else 1

/-- Lower probability of counterexample C2 (Table 1): `T{0} = 0.5`, `T{0,1} = 0.2`,
`T{0,2} = 0.5`, `T Ω = 1`, all other events `0`. -/
noncomputable def c2L (A : Finset (Fin 3)) : ℝ :=
  if A = univ then 1 else if A = {0} then 1 / 2 else if A = {0, 1} then 1 / 5
  else if A = {0, 2} then 1 / 2 else 0

lemma finset_fin3_cases (A : Finset (Fin 3)) :
    A = ∅ ∨ A = {0} ∨ A = {1} ∨ A = {2} ∨ A = {0, 1} ∨ A = {0, 2} ∨ A = {1, 2} ∨
      A = univ := by
  revert A; decide

lemma isConjLower_c1L : IsConjLower c1L := by
  refine ⟨by simp [c1L], by simp [c1L], fun A => ?_, fun A => ?_⟩ <;>
    rcases finset_fin3_cases A with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp (config := { decide := true }) [c1L] <;> norm_num

lemma isConjLower_c2L : IsConjLower c2L := by
  refine ⟨by simp (config := { decide := true }) [c2L], by simp [c2L], fun A => ?_,
    fun A => ?_⟩ <;>
    rcases finset_fin3_cases A with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp (config := { decide := true }) [c2L] <;> norm_num

/-- **Counterexample C1 (sure loss).** -/
theorem c1_sure_loss : ¬ (ofLower c1L).AvoidsSureLoss := by
  rintro ⟨P, hP, h⟩
  have h0 := (h {0}).1
  have h1 := (h {1}).1
  have h2 := (h {2}).1
  simp only [ofLower, c1L, Finset.card_singleton, prob, Finset.sum_singleton] at h0 h1 h2
  norm_num at h0 h1 h2
  have hs := hP.2
  rw [Fin.sum_univ_three] at hs
  linarith

/-- **Counterexample C2 (avoids sure loss, incoherent).** -/
theorem c2_asl_incoherent :
    (ofLower c2L).AvoidsSureLoss ∧ ¬ (ofLower c2L).Coherent := by
  constructor
  · refine ⟨![1 / 2, 1 / 4, 1 / 4], ⟨fun x => ?_, ?_⟩, fun A => ?_⟩
    · fin_cases x <;> norm_num
    · rw [Fin.sum_univ_three]; norm_num
    · simp only [ofLower, upper, compl_compl]
      rw [prob_fin3]
      rcases finset_fin3_cases A with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
        simp (config := { decide := true }) [c2L] <;> norm_num
  · rintro ⟨-, h⟩
    obtain ⟨⟨P, hP, hP01⟩, -⟩ := h {0, 1}
    have h0 := (hP.2 {0}).1
    have hP1 := hP.1.1 1
    simp only [ofLower] at hP01 h0
    rw [prob_fin3] at hP01 h0
    simp (config := { decide := true }) [c2L] at hP01 h0
    norm_num at hP01 h0
    linarith

/-- **Theorem 1(d).** (N) + (D) + (B) (with standard components) imply neither avoiding sure
loss nor coherence, already on three outcomes. -/
theorem thm1_d :
    (∃ μ : NP (Fin 3), μ.Normalised ∧ μ.Dual ∧ μ.Boundary ∧ μ.Standard ∧
        ¬ μ.AvoidsSureLoss) ∧
      (∃ μ : NP (Fin 3), μ.Normalised ∧ μ.Dual ∧ μ.Boundary ∧ μ.Standard ∧
        μ.AvoidsSureLoss ∧ ¬ μ.Coherent) :=
  ⟨⟨ofLower c1L, ofLower_normalised _, ofLower_dual _, ofLower_boundary isConjLower_c1L,
      ofLower_standard isConjLower_c1L, c1_sure_loss⟩,
    ⟨ofLower c2L, ofLower_normalised _, ofLower_dual _, ofLower_boundary isConjLower_c2L,
      ofLower_standard isConjLower_c2L, c2_asl_incoherent⟩⟩

end NP

/-! ## Corollary 2 — the evidential triple and the imprecise Dirichlet model -/

section IDM

variable (n : α → ℝ) (s : ℝ)

/-- Credal set of the evidential triple on singletons:
`T x = n x / (N + s)`, `I x = s / (N + s)`, `F x = 1 - T x - I x`, read as
`T x ≤ P x ≤ 1 - F x = (n x + s) / (N + s)`. -/
def idmCredal : Set (α → ℝ) :=
  {P | IsProb P ∧ ∀ x, n x / (∑ y, n y + s) ≤ P x ∧ P x ≤ (n x + s) / (∑ y, n y + s)}

/-- The evidential triple is normalised and its upper bound `1 - F` is `(n x + s)/(N + s)`. -/
lemma evidential_triple (hn : ∀ x, 0 ≤ n x) (hs : 0 < s) (x : α) :
    let N := ∑ y, n y
    let T := n x / (N + s); let I := s / (N + s); let F := (N - n x) / (N + s)
    T + I + F = 1 ∧ 1 - F = (n x + s) / (N + s) := by
  intro N T I F
  have hN : 0 ≤ N := Finset.sum_nonneg fun y _ => hn y
  have hpos : N + s ≠ 0 := by positivity
  constructor
  · simp only [T, I, F]; field_simp; ring
  · simp only [F]; field_simp; ring

variable {n s}

/-- The extreme point that puts the prior strength on outcome `j`. -/
noncomputable def idmVertex (n : α → ℝ) (s : ℝ) (j : α) : α → ℝ :=
  fun x => (n x + if x = j then s else 0) / (∑ y, n y + s)

lemma idmVertex_mem (hn : ∀ x, 0 ≤ n x) (hs : 0 < s) (j : α) :
    idmVertex n s j ∈ idmCredal n s := by
  have hN : 0 ≤ ∑ y, n y := Finset.sum_nonneg fun y _ => hn y
  have hpos : 0 < ∑ y, n y + s := by linarith
  refine ⟨⟨fun x => ?_, ?_⟩, fun x => ⟨?_, ?_⟩⟩
  · unfold idmVertex
    apply div_nonneg _ hpos.le
    have := hn x; split_ifs <;> linarith
  · unfold idmVertex
    rw [← Finset.sum_div, Finset.sum_add_distrib, Finset.sum_ite_eq' univ j (fun _ => s)]
    simp only [Finset.mem_univ, if_true]
    exact div_self hpos.ne'
  · unfold idmVertex
    apply div_le_div_of_nonneg_right _ hpos.le
    split_ifs <;> linarith
  · unfold idmVertex
    apply div_le_div_of_nonneg_right _ hpos.le
    split_ifs <;> linarith

lemma prob_idmVertex (j : α) (A : Finset α) :
    prob (idmVertex n s j) A =
      (∑ x ∈ A, n x + if j ∈ A then s else 0) / (∑ y, n y + s) := by
  unfold prob idmVertex
  rw [← Finset.sum_div, Finset.sum_add_distrib, Finset.sum_ite_eq' A j (fun _ => s)]

/-- **Corollary 2 (IDM), lower envelope.** For `∅ ≠ A ≠ Ω`,
`min_{P ∈ M} P(A) = n_A / (N + s)`. -/
theorem idm_isLeast (hn : ∀ x, 0 ≤ n x) (hs : 0 < s) {A : Finset α} (hA : A ≠ univ) :
    IsLeast ((fun P => prob P A) '' idmCredal n s) ((∑ x ∈ A, n x) / (∑ y, n y + s)) := by
  obtain ⟨j, hj⟩ : ∃ j, j ∉ A := by
    by_contra h; push_neg at h; exact hA (Finset.eq_univ_of_forall h)
  constructor
  · refine ⟨idmVertex n s j, idmVertex_mem hn hs j, ?_⟩
    simp only [prob_idmVertex, hj, if_false, add_zero]
  · rintro _ ⟨P, hP, rfl⟩
    unfold prob
    rw [Finset.sum_div]
    exact Finset.sum_le_sum fun x _ => (hP.2 x).1

/-- **Corollary 2 (IDM), upper envelope.** For `∅ ≠ A ≠ Ω`,
`max_{P ∈ M} P(A) = (n_A + s) / (N + s)`. -/
theorem idm_isGreatest (hn : ∀ x, 0 ≤ n x) (hs : 0 < s) {A : Finset α} (hA : A.Nonempty) :
    IsGreatest ((fun P => prob P A) '' idmCredal n s)
      ((∑ x ∈ A, n x + s) / (∑ y, n y + s)) := by
  obtain ⟨j, hj⟩ := hA
  have hN : 0 ≤ ∑ y, n y := Finset.sum_nonneg fun y _ => hn y
  have hpos : 0 < ∑ y, n y + s := by linarith
  constructor
  · refine ⟨idmVertex n s j, idmVertex_mem hn hs j, ?_⟩
    simp only [prob_idmVertex, hj, if_true]
  · rintro _ ⟨P, hP, rfl⟩
    have hc : (∑ x ∈ Aᶜ, n x) / (∑ y, n y + s) ≤ prob P Aᶜ := by
      unfold prob
      rw [Finset.sum_div]
      exact Finset.sum_le_sum fun x _ => (hP.2 x).1
    rw [prob_compl hP.1] at hc
    have hsplit := Finset.sum_add_sum_compl A n
    have key : (∑ x ∈ A, n x + s) / (∑ y, n y + s) =
        1 - (∑ x ∈ Aᶜ, n x) / (∑ y, n y + s) := by
      rw [eq_sub_iff_add_eq, ← add_div, div_eq_one_iff_eq hpos.ne']
      linarith
    show prob P A ≤ _
    rw [key]; linarith

/-- **Corollary 2 (IDM), coherence.** With at least two outcomes, every singleton bound of the
evidential triple is attained. -/
theorem idm_coherent (hn : ∀ x, 0 ≤ n x) (hs : 0 < s) (h2 : 2 ≤ Fintype.card α) (x : α) :
    (∃ P ∈ idmCredal n s, P x = n x / (∑ y, n y + s)) ∧
      (∃ P ∈ idmCredal n s, P x = (n x + s) / (∑ y, n y + s)) := by
  have hne : ({x} : Finset α) ≠ univ := by
    intro h
    have := congrArg Finset.card h
    rw [Finset.card_singleton, Finset.card_univ] at this
    omega
  obtain ⟨⟨P, hP, h1⟩, -⟩ := idm_isLeast hn hs hne
  obtain ⟨⟨Q, hQ, h2⟩, -⟩ := idm_isGreatest hn hs (Finset.singleton_nonempty x)
  simp only [prob, Finset.sum_singleton] at h1 h2
  exact ⟨⟨P, hP, h1⟩, ⟨Q, hQ, h2⟩⟩

end IDM

end NeutroEvidence
