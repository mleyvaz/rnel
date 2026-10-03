"""offlib: linear programmes over unit-mass charges (quasi-probabilities) on finite frames.

A charge q on n atoms is written q = p - r with p, r >= 0 (componentwise).  The set
    Delta_eps = { q : sum q = 1, nu(q) <= eps },  nu(q) = sum_x max(0, -q(x))  (negative variation)
is described exactly by  { p - r : sum(p) - sum(r) = 1, sum(r) <= eps, p, r >= 0 }, because the least
sum(r) over all decompositions of q is nu(q).
Sign restrictions per atom: 'free', 'pos' (r = 0), 'neg' (p = 0).
All LPs are solved with scipy's HiGHS.
"""
import itertools
import numpy as np
from scipy.optimize import linprog

TOL = 1e-9


def patterns_all(m):
    return [tuple(b) for b in itertools.product((0, 1), repeat=m)]


def minimal_lifting(m):
    one = tuple([1] * m)
    return [tuple(1 - (j == k) for j in range(m)) for k in range(m)] + [one]


def signed_lifting(m):
    """L^{+-}: minimal lifting L plus its mirror image pi(L) = {1 - p : p in L}; deduplicated (m <= 2)."""
    L = minimal_lifting(m)
    R = [tuple(1 - b for b in p) for p in L]
    out, signs = [], []
    for p in L:
        if p not in out:
            out.append(p); signs.append("pos")
    for p in R:
        if p not in out:
            out.append(p); signs.append("neg")
        else:  # m <= 2: a mirror pattern coincides with a support pattern; it must carry both signs
            signs[out.index(p)] = "free"
    return out, signs


def event_vec(patterns, k):
    return np.array([p[k] for p in patterns], dtype=float)


class ChargeLP:
    """min / max of a linear functional c.q over {q in Delta_eps(frame), sign restrictions, A_ub q <= b_ub, A_eq q = b_eq}."""

    def __init__(self, n, eps, signs=None):
        self.n, self.eps = n, eps
        self.signs = signs or ["free"] * n
        self.Aub, self.bub, self.Aeq, self.beq = [], [], [], []

    def ge(self, a, b):  # a.q >= b
        self.Aub.append(-np.asarray(a, float)); self.bub.append(-b)

    def le(self, a, b):
        self.Aub.append(np.asarray(a, float)); self.bub.append(b)

    def eq(self, a, b):
        self.Aeq.append(np.asarray(a, float)); self.beq.append(b)

    def _lift(self, a):  # q-coefficients -> (p, r) coefficients
        a = np.asarray(a, float)
        return np.concatenate([a, -a])

    def solve(self, c, sense="min"):
        n = self.n
        Aub = [self._lift(a) for a in self.Aub]
        bub = list(self.bub)
        # sum r <= eps
        Aub.append(np.concatenate([np.zeros(n), np.ones(n)])); bub.append(self.eps)
        Aeq = [self._lift(a) for a in self.Aeq]; beq = list(self.beq)
        Aeq.append(np.concatenate([np.ones(n), -np.ones(n)])); beq.append(1.0)
        bounds = []
        for s in self.signs:
            bounds.append((0, 0) if s == "neg" else (0, None))
        for s in self.signs:
            bounds.append((0, 0) if s == "pos" else (0, None))
        cc = self._lift(c) * (1 if sense == "min" else -1)
        res = linprog(cc, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.array(Aeq), b_eq=np.array(beq),
                      bounds=bounds, method="highs")
        if res.status == 2:
            return None, None
        if res.status != 0:
            raise RuntimeError(res.message)
        q = res.x[:n] - res.x[n:]
        val = float(np.dot(c, q))
        return val, q


def nu(q):
    return float(np.sum(np.maximum(0.0, -np.asarray(q))))


def lower_envelope(patterns, x, eps, signs=None):
    """(L(E_1),...,L(E_m)) of K_eps(x) = {q in Delta_eps : Q(E_k) >= x_k}; None if empty."""
    m = len(x); n = len(patterns)
    out = []
    for k in range(m):
        lp = ChargeLP(n, eps, signs)
        for j in range(m):
            lp.ge(event_vec(patterns, j), x[j])
        v, _ = lp.solve(event_vec(patterns, k), "min")
        if v is None:
            return None
        out.append(v)
    return out


def upper_envelope(patterns, x, eps, signs=None):
    m = len(x); n = len(patterns)
    out = []
    for k in range(m):
        lp = ChargeLP(n, eps, signs)
        for j in range(m):
            lp.ge(event_vec(patterns, j), x[j])
        v, _ = lp.solve(event_vec(patterns, k), "max")
        out.append(v)
    return out


def faithful(patterns, x, eps, signs=None, tol=1e-7):
    le = lower_envelope(patterns, x, eps, signs)
    if le is None:
        return False
    return max(abs(a - b) for a, b in zip(le, x)) < tol


def eps_star(x):
    """least budget: max_k max((x_k - 1)^+, (-x_k)^+)"""
    return max(0.0, max(max(v - 1.0, -v) for v in x))
