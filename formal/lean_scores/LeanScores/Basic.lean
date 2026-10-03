import Mathlib

/-!
# Funciones de score neutrosóficas como tasas base ocultas

Formalización de los resultados algebraicos del paper. Una tripleta es
`(T, I, F)` con componentes reales; las hipótesis de normalización o de
rango se ponen explícitamente en cada enunciado.
-/

/-- Familia con tasa base `a`: `S_a = (1 - a) T + a (1 - F)`. -/
def S (a T F : ℝ) : ℝ := (1 - a) * T + a * (1 - F)

/-- Familia con tasa base `a` y aversión a la indeterminación `l`. -/
def Sl (a l T I F : ℝ) : ℝ := S a T F - l * I

/-- Score clásica `(2 + T - I - F) / 3`. -/
noncomputable def classic (T I F : ℝ) : ℝ := (2 + T - I - F) / 3

/-- **Descomposición.** La score clásica es `S_{1/2}` más un término en `I`. -/
theorem classic_decomp (T I F : ℝ) :
    classic T I F = (2 / 3) * S (1 / 2) T F + (1 - I) / 3 := by
  unfold classic S; ring

/-- La score clásica es una transformación afín positiva de `S_{1/2, 1/2}`. -/
theorem classic_eq_Sl_half_half (T I F : ℝ) :
    classic T I F = (2 / 3) * Sl (1 / 2) (1 / 2) T I F + 1 / 3 := by
  unfold classic Sl S; ring

/-- **Caso normalizado.** Si `T + I + F = 1`, la score clásica es `(1 + 2T) / 3`. -/
theorem classic_normalized (T I F : ℝ) (h : T + I + F = 1) :
    classic T I F = (1 + 2 * T) / 3 := by
  have hI : I = 1 - T - F := by linarith
  subst hI
  unfold classic; ring

/-- **Colapso.** En el caso normalizado, la score clásica ordena exactamente como `T`
(la probabilidad inferior: tasa base `a = 0`). -/
theorem classic_ranks_by_T (T₁ I₁ F₁ T₂ I₂ F₂ : ℝ)
    (h₁ : T₁ + I₁ + F₁ = 1) (h₂ : T₂ + I₂ + F₂ = 1) :
    classic T₁ I₁ F₁ < classic T₂ I₂ F₂ ↔ T₁ < T₂ := by
  rw [classic_normalized T₁ I₁ F₁ h₁, classic_normalized T₂ I₂ F₂ h₂]
  constructor <;> intro h <;> linarith

/-- La función de exactitud `T - F` ordena igual que `S_{1/2}`, sin hipótesis. -/
theorem accuracy_ranks_as_S_half (T₁ F₁ T₂ F₂ : ℝ) :
    T₁ - F₁ < T₂ - F₂ ↔ S (1 / 2) T₁ F₁ < S (1 / 2) T₂ F₂ := by
  unfold S
  constructor <;> intro h <;> linarith

/-- **Fórmula única, caso glut.** `S_a = T - a (T + F - 1)`. -/
theorem S_glut_form (a T F : ℝ) : S a T F = T - a * (T + F - 1) := by
  unfold S; ring

/-- **Caso hueco.** Si `T + F ≤ 1` y `a ∈ [0,1]`, entonces `T ≤ S_a ≤ 1 - F`. -/
theorem S_gap_bounds (a T F : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1) (h : T + F ≤ 1) :
    T ≤ S a T F ∧ S a T F ≤ 1 - F := by
  unfold S
  have hg : 0 ≤ 1 - F - T := by linarith
  constructor
  · nlinarith [mul_nonneg ha hg]
  · nlinarith [mul_nonneg (sub_nonneg.mpr ha1) hg]

/-- **Caso glut.** Si `T + F ≥ 1` y `a ∈ [0,1]`, entonces `1 - F ≤ S_a ≤ T`. -/
theorem S_glut_bounds (a T F : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1) (h : 1 ≤ T + F) :
    1 - F ≤ S a T F ∧ S a T F ≤ T := by
  unfold S
  have hc : 0 ≤ T + F - 1 := by linarith
  constructor
  · nlinarith [mul_nonneg (sub_nonneg.mpr ha1) hc]
  · nlinarith [mul_nonneg ha hc]

/-- En el caso normalizado, `S_{a,l} = T + (a - l) I`: colapsa a `T` cuando `a = l`. -/
theorem Sl_normalized (a l T I F : ℝ) (h : T + I + F = 1) :
    Sl a l T I F = T + (a - l) * I := by
  have hF : F = 1 - T - I := by linarith
  subst hF
  unfold Sl S; ring

/-- **Tasa base imprecisa.** Si `A` domina a `B` en los extremos `aL` y `aU`,
domina para toda tasa base intermedia. -/
theorem dominance_interval (aL aU a T₁ F₁ T₂ F₂ : ℝ)
    (hL : aL ≤ a) (hU : a ≤ aU)
    (h₁ : S aL T₂ F₂ ≤ S aL T₁ F₁) (h₂ : S aU T₂ F₂ ≤ S aU T₁ F₁) :
    S a T₂ F₂ ≤ S a T₁ F₁ := by
  unfold S at *
  rcases eq_or_lt_of_le (le_trans hL hU) with heq | hlt
  · have : a = aL := le_antisymm (heq ▸ hU) hL
    subst this; linarith
  · -- (aU - aL)·d(a) = (aU - a)·d(aL) + (a - aL)·d(aU) ≥ 0, con d lineal en a
    have key : (aU - aL) * (((1 - a) * T₁ + a * (1 - F₁)) - ((1 - a) * T₂ + a * (1 - F₂)))
        = (aU - a) * (((1 - aL) * T₁ + aL * (1 - F₁)) - ((1 - aL) * T₂ + aL * (1 - F₂)))
          + (a - aL) * (((1 - aU) * T₁ + aU * (1 - F₁)) - ((1 - aU) * T₂ + aU * (1 - F₂))) := by
      ring
    have hpos : 0 < aU - aL := by linarith
    have hrhs : 0 ≤ (aU - a) * (((1 - aL) * T₁ + aL * (1 - F₁)) - ((1 - aL) * T₂ + aL * (1 - F₂)))
          + (a - aL) * (((1 - aU) * T₁ + aU * (1 - F₁)) - ((1 - aU) * T₂ + aU * (1 - F₂))) := by
      apply add_nonneg
      · exact mul_nonneg (by linarith) (by linarith)
      · exact mul_nonneg (by linarith) (by linarith)
    rw [← key] at hrhs
    have := (mul_nonneg_iff_of_pos_left hpos).mp hrhs
    linarith
