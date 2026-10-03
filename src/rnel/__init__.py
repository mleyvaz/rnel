"""rnel: Refined Neutrosophic Evidential Logic (Smarandache and Leyva-Vázquez).

Core (no dependencies beyond the standard library):
  rnel.sl         Subjective Logic opinions and operators
  rnel.tuple      the RNEL evidence tuple (Definition 8.1) and fused contradiction (Definition 8.4)
  rnel.operators  priority-product operators for (T,I,N,F), (T,I,N,U1..Un,F) and RNEL
  rnel.ranking    total-order ranking cascades
  rnel.decide     from a tuple to an action
  rnel.datasets   loaders for AVeriTeC, CLIMATE-FEVER and ChaosNLI
  rnel.off        (experimental) signed evidence, retraction and over/under/off values
  rnel.conflict   (experimental) between-source conflict from axioms (K_b, C*), order-free state, provenance-aware fusion
  rnel.dsmt      (experimental) DSmT on small frames (free, Shafer and hybrid models): conjunctive/DSmC, Dempster,
                 TBM, Yager, Dubois-Prade, PCR5, PCR6, PCR6+, BetP/DSmP, and the exact dictionary between DSm masses
                 on {x, not x} and RNEL tuples
  rnel.neutro_stats (experimental) neutrosophic estimates a + sum b_k I_k with one labelled I per
                 indeterminacy type and source; credal, glut and gap readings
Optional (numpy + scikit-learn):
  rnel.credal    (experimental) credal ensembles (lower/upper probabilities), their neutrosophic reading
                 (representation only), per-view evidence, between-view conflict C* and actions
Optional (numpy + scipy):
  rnel.neutro_credal (experimental) Walley's tools on neutrosophic triples: sure loss, coherence, natural
                 extension, IDM, decisions by sets, glut lifting, and a one-call credal + RNEL diagnosis
Optional (PyTorch):
  rnel.nn         typed evidential head and per-source evidential network
"""
from .sl import Opinion
from .tuple import Reports, RNELTuple, fused_contradiction, rnel_tuple, sl_opinion

__version__ = "0.3.0"
__all__ = ["Opinion", "Reports", "RNELTuple", "rnel_tuple", "sl_opinion", "fused_contradiction"]
