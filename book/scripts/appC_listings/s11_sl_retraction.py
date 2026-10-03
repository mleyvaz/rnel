from rnel import neutro_credal as nc
from rnel.sl import Opinion

# Theorem 5(a)-(b): evidence (r, q) with weight W gives the same triple as the binomial IDM
r, q, W = 4.0, 1.0, 2.0
op = Opinion.from_evidence(r, q, W)
print("SL opinion (b, d, u):", (round(op.b, 4), round(op.d, 4), round(op.u, 4)))
print("IDM triple (T, I, F):", tuple(round(float(v), 4) for v in nc.idm_triple([r, q], s=W)[0]))

# Theorem 5(c): the saturating chart and its retraction
for r, q in [(3.0, 1.0), (6.0, 3.0)]:
    Ts, Fs = r / (r + W), q / (q + W)
    b, d, u = nc.sl_retraction(Ts, Fs)
    o = Opinion.from_evidence(r, q, W)
    print(f"r={r}, q={q}: Ts+Fs = {Ts + Fs:.3f} (rq > W^2: {r * q > W * W}), "
          f"betting sure loss = {nc.sure_loss_degree((Ts, 0, Fs)):.3f}, "
          f"chart interval [{b:.4f}, {b + u:.4f}], match SL = {abs(b - o.b) + abs(u - o.u) < 1e-12}")

# Theorem 5(d): on the normalised plane the two readings differ
b, d, u = nc.sl_retraction(0.2, 0.3)
print("betting [0.2, 0.7] vs chart", [round(b, 4), round(b + u, 4)])
