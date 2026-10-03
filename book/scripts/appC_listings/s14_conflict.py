from rnel.conflict import ConflictState, c_star, credal_gap, k_between, k_within
from rnel.tuple import Reports, rnel_tuple
from rnel.decide import Policy

# Three sources with evidence (for, against) a claim x
profile = [(8, 1), (1, 7), (2, 2)]
st = ConflictState.of_profile(profile)
print("R, S, K_w, K_b =", st.R, st.S, k_within(profile), k_between(profile))
print("C* =", round(c_star(profile), 4), "| reversed order:", round(c_star(profile[::-1]), 4))

# The credal gap of any interval [l, u] equals (u - l) K_b
for lo, up in [(0.0, 1.0), (0.2, 0.7)]:
    print(f"[{lo}, {up}]: credal gap = {credal_gap(profile, lo, up):.4f}, "
          f"(u - l) K_b = {(up - lo) * k_between(profile):.4f}")

# The RNEL tuple and a decision policy
t = rnel_tuple(Reports(t=11, f=10, c=3))
print("RNEL tuple:", {k: round(v, 3) for k, v in t.as_dict().items()})
print("action:", Policy().decide(t))
