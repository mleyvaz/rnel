"""Does naming the indeterminacy (several I) change which action reduces uncertainty?

Question. Two analysts see the same evidence. One sees only the interval for p(x) (the credal / IDM
output, optionally with the number of reports N); the other sees the neutrosophic number with one I
per type (C contradictory, U undetermined, N ill-posed, G missing data). With a fixed budget each must
choose ONE action: collect more reports, review contradictory reports, clarify undetermined ones, or
reformulate ill-posed questions. Does the decomposition lead to better choices, and by how much?

Simulation (all under the RNEL report model, so this tests decision relevance *if* the report types are
real; it is not an empirical validation):
  * a case: true chance theta ~ U(0, 1); N0 reports; each report is contradictory / undetermined /
    ill-posed with probabilities (q_c, q_v, q_n) ~ scaled Dirichlet, otherwise t with prob. theta, f else.
  * actions, budget B cost units:  collect (1 per new report, same mix) | review (COST_C per c report,
    resolved to t/f) | clarify (COST_V per v report, resolved) | reformulate (COST_N per n report,
    resolved with prob. P_REFORM, else becomes v).
  * outcome: width of p(x) after the action (and coverage of theta as a sanity check).
Policies: oracle (best realised action); typed (predicts each action's expected width from the counts);
interval-only and interval+N (best action learned per bin of the observed interval on a training half);
always-collect; random.

Run: python scripts/sim_typed_actions.py  [--cases 20000 --seed 0]
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from rnel.neutro_stats import decompose, estimate
from rnel.tuple import Reports

W = 2.0
BUDGET = 12
COST_C, COST_V, COST_N = 2, 3, 2
P_REFORM = 0.8
ACTIONS = ("collect", "review", "clarify", "reformulate")


# ----------------------------------------------------------------------------- world
def draw_case(rng):
    theta = rng.uniform()
    n0 = int(rng.integers(2, 61))
    q_ind = rng.uniform(0, 0.8)                       # share of indeterminate reports
    q = rng.dirichlet([1, 1, 1]) * q_ind              # split into c, v, n
    return theta, n0, q


def sample_reports(rng, n, theta, q):
    kinds = rng.choice(4, size=n, p=[q[0], q[1], q[2], 1 - q.sum()])
    c, v, nn = (kinds == 0).sum(), (kinds == 1).sum(), (kinds == 2).sum()
    det = (kinds == 3).sum()
    t = rng.binomial(det, theta)
    return Reports(t=float(t), f=float(det - t), c=float(c), v=float(v), n=float(nn))


def apply(rng, r: Reports, action, theta, q):
    if action == "collect":
        return r + sample_reports(rng, BUDGET, theta, q)
    if action == "review":
        k = min(int(r.c), BUDGET // COST_C)
        t = rng.binomial(k, theta)
        return Reports(r.t + t, r.f + k - t, r.c - k, r.v, r.n)
    if action == "clarify":
        k = min(int(r.v), BUDGET // COST_V)
        t = rng.binomial(k, theta)
        return Reports(r.t + t, r.f + k - t, r.c, r.v - k, r.n)
    if action == "reformulate":
        k = min(int(r.n), BUDGET // COST_N)
        ok = rng.binomial(k, P_REFORM)
        t = rng.binomial(ok, theta)
        return Reports(r.t + t, r.f + ok - t, r.c, r.v + k - ok, r.n - k)
    raise ValueError(action)


def width(r: Reports) -> float:
    return estimate(r, W=W).x.width


# ----------------------------------------------------------------------------- typed policy
def predicted_width(r: Reports, action) -> float:
    """Expected width after the action, computed only from the observed counts (the decomposition)."""
    N = r.t + r.f + r.c + r.v + r.n
    ind = r.c + r.v + r.n
    S = N + W
    if action == "collect":
        q_hat = ind / N if N else 0.0
        return (ind + BUDGET * q_hat + W) / (S + BUDGET)
    if action == "review":
        return (ind - min(r.c, BUDGET // COST_C) + W) / S
    if action == "clarify":
        return (ind - min(r.v, BUDGET // COST_V) + W) / S
    if action == "reformulate":
        return (ind - P_REFORM * min(r.n, BUDGET // COST_N) + W) / S
    raise ValueError(action)


def typed_choice(r: Reports) -> str:
    return min(ACTIONS, key=lambda a: predicted_width(r, a))


# ----------------------------------------------------------------------------- interval baselines
def interval_key(r: Reports, with_n: bool, bins: int = 10):
    e = estimate(r, W=W).x
    k = (min(int(e.lo * bins), bins - 1), min(int(e.width * bins), bins - 1))
    if with_n:
        N = r.t + r.f + r.c + r.v + r.n
        k += (min(int(np.log2(N + 1)), 6),)
    return k


def learn_interval_policy(train, with_n):
    """Best action (lowest mean realised width) per interval bin, from training cases."""
    acc = defaultdict(lambda: np.zeros(len(ACTIONS)))
    cnt = defaultdict(int)
    for row in train:
        k = interval_key(row["r"], with_n)
        acc[k] += row["w_after"]
        cnt[k] += 1
    glob = sum(acc.values()) / max(1, sum(cnt.values()))
    table = {k: ACTIONS[int(np.argmin(v / cnt[k]))] for k, v in acc.items()}
    default = ACTIONS[int(np.argmin(glob))]
    return lambda r: table.get(interval_key(r, with_n), default)


# ----------------------------------------------------------------------------- experiment
def mislabel(rng, r: Reports, p: float) -> Reports:
    """What the analyst observes: each indeterminate report keeps its type with prob. 1 - p, otherwise it
    is recorded as one of the other two types (the total indeterminate count is unchanged)."""
    if p <= 0:
        return r
    obs = np.zeros(3)
    for i, k in enumerate((int(r.c), int(r.v), int(r.n))):
        moved = rng.binomial(k, p)
        obs[i] += k - moved
        others = [j for j in range(3) if j != i]
        to = rng.binomial(moved, 0.5)
        obs[others[0]] += to
        obs[others[1]] += moved - to
    return Reports(r.t, r.f, obs[0], obs[1], obs[2])


def run(cases: int, seed: int, p_mislabel: float = 0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(cases):
        theta, n0, q = draw_case(rng)
        true = sample_reports(rng, n0, theta, q)
        w_after = np.array([width(apply(rng, true, a, theta, q)) for a in ACTIONS])
        r = mislabel(rng, true, p_mislabel)           # policies see r; actions act on the true reports
        rows.append({"r": r, "theta": theta, "w0": width(true), "w_after": w_after})
    half = cases // 2
    train, test = rows[:half], rows[half:]
    pol = {
        "oracle": lambda row: ACTIONS[int(np.argmin(row["w_after"]))],
        "typed (several I)": lambda row: typed_choice(row["r"]),
        "interval+N (learned)": (lambda f: lambda row: f(row["r"]))(learn_interval_policy(train, True)),
        "interval only (learned)": (lambda f: lambda row: f(row["r"]))(learn_interval_policy(train, False)),
        "always collect": lambda row: "collect",
        "random": lambda row: ACTIONS[int(rng.integers(len(ACTIONS)))],
    }
    w0 = np.mean([row["w0"] for row in test])
    oracle = np.array([row["w_after"].min() for row in test])
    print(f"type mislabelling rate: {p_mislabel}")
    print(f"cases: {cases} (test {len(test)}), W={W}, budget={BUDGET}, costs c/v/n={COST_C}/{COST_V}/{COST_N}")
    print(f"mean width before any action: {w0:.4f}\n")
    print(f"{'policy':26s} {'mean width':>10s} {'reduction':>10s} {'regret':>8s} {'= oracle':>9s}")
    res = {}
    for name, p in pol.items():
        ch = [p(row) for row in test]
        w = np.array([row["w_after"][ACTIONS.index(a)] for row, a in zip(test, ch)])
        res[name] = (w, ch)
        print(f"{name:26s} {w.mean():10.4f} {w0 - w.mean():10.4f} {np.mean(w - oracle):8.4f} "
              f"{np.mean(np.isclose(w, oracle)):9.1%}")

    # where do typed and interval+N disagree, and who wins there?
    wt, ct = res["typed (several I)"]
    wi, ci = res["interval+N (learned)"]
    dis = np.array([a != b for a, b in zip(ct, ci)])
    print(f"\ntyped vs interval+N: disagree on {dis.mean():.1%} of cases; on those, mean width "
          f"typed {wt[dis].mean():.4f} vs interval+N {wi[dis].mean():.4f}; typed better in "
          f"{np.mean(wt[dis] < wi[dis] - 1e-12):.1%}, worse in {np.mean(wt[dis] > wi[dis] + 1e-12):.1%}")

    # by dominant symbol
    print("\nreduction by dominant symbol of the width (typed vs always collect):")
    dom = []
    for row in test:
        d = decompose(estimate(row["r"], W=W).x, by="type")
        dom.append(max(d, key=d.get))
    dom = np.array(dom)
    wc, _ = res["always collect"]
    w0v = np.array([row["w0"] for row in test])
    for s in ("G", "C", "U", "N"):
        m = dom == s
        if m.any():
            print(f"  dominant I_{s}: {m.sum():5d} cases | collect reduces {np.mean(w0v[m] - wc[m]):.4f} | "
                  f"typed reduces {np.mean(w0v[m] - wt[m]):.4f}")

    # sanity: coverage of theta by the final interval of the typed policy is not checked per draw here;
    # the paired example below shows the mechanism with exact numbers.
    print("\npaired example, SAME interval [0.25, 0.75]:")
    for label, r in (("A: little data ", Reports(t=1, f=1)), ("B: contradiction", Reports(t=4, f=4, c=6))):
        e = estimate(r, W=W).x
        pw = {a: round(predicted_width(r, a), 3) for a in ACTIONS}
        print(f"  {label} [{e.lo:.3f}, {e.hi:.3f}] decomposition "
              f"{ {k: round(v, 3) for k, v in decompose(e, 'type').items()} } -> expected width {pw}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mislabel", type=float, default=0.0)
    a = ap.parse_args()
    run(a.cases, a.seed, a.mislabel)
