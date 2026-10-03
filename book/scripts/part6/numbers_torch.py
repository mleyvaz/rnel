"""WE7 (Chapter VI.2): the typed head of rnel.nn reduces to evidential deep learning. Runs torch in its own
process (importing torch after numpy/scipy crashed on this machine). Output: numbers_torch.json in the current directory (book edition)."""
import json, os, sys
import torch  # first import
from rnel.nn import rnel_from_evidence, typed_evidential_loss, fused_dissonance
OUT = os.getcwd()  # book edition
N = {}
e = torch.tensor([[3.0, 1.0, 0.0, 0.0, 0.0]])
tup = rnel_from_evidence(e)[0].tolist()  # T, F, C, U, N, G (W = 2)
for k, v in zip(("T", "F", "C", "U", "N", "G"), tup):
    N["we7." + k] = v
S = 3 + 1 + 2   # EDL with K = 2: alpha = e + 1, S = sum alpha; b_k = e_k / S, u = K / S
N["we7.edl_b"], N["we7.edl_u"] = 3 / S, 2 / S
assert abs(tup[0] - 3 / S) < 1e-7 and abs(tup[5] - 2 / S) < 1e-7
N["we7.dissonance"] = fused_dissonance(torch.tensor([[3.0, 1.0]]))[0].item()
N["we7.loss_T"] = typed_evidential_loss(e, torch.tensor([0])).item()
N["we7.loss_C"] = typed_evidential_loss(e, torch.tensor([2])).item()
json.dump(N, open(os.path.join(OUT, "numbers_torch.json"), "w"), indent=1)
print(N)
