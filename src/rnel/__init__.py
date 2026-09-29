"""rnel: Refined Neutrosophic Evidential Logic (Smarandache and Leyva-Vázquez).

Core (no dependencies beyond the standard library):
  rnel.sl         Subjective Logic opinions and operators
  rnel.tuple      the RNEL evidence tuple (Definition 8.1) and fused contradiction (Definition 8.4)
  rnel.operators  priority-product operators for (T,I,N,F), (T,I,N,U1..Un,F) and RNEL
  rnel.ranking    total-order ranking cascades
  rnel.decide     from a tuple to an action
  rnel.datasets   loaders for AVeriTeC, CLIMATE-FEVER and ChaosNLI
Optional (PyTorch):
  rnel.nn         typed evidential head and per-source evidential network
"""
from .sl import Opinion
from .tuple import Reports, RNELTuple, fused_contradiction, rnel_tuple, sl_opinion

__version__ = "0.2.0"
__all__ = ["Opinion", "Reports", "RNELTuple", "rnel_tuple", "sl_opinion", "fused_contradiction"]
