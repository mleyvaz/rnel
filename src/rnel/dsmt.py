"""(Experimental) Dezert-Smarandache theory (DSmT): combination rules on small finite frames, decision
transforms, and the exact dictionary between DSm masses on a binary frame and RNEL tuples.

Standard library only.  Reference implementation for teaching and checking published worked examples, not a
large-frame optimiser: every rule enumerates the focal tuples of the sources.

Frames and models
-----------------
A frame Theta = {theta_1, ..., theta_n} is encoded by its Venn regions (Smarandache's codification): a region is a
non-empty set S of atom indices, read "in exactly the atoms of S".  An element of the fusion space is the set of
regions it covers; union and intersection are set union and intersection, so the hyper-power set D^Theta (closed
under union and intersection) is represented exactly.  A *model* fixes which regions exist:

* ``"free"``   free DSm model: every region exists (theta_i intersect theta_j may carry mass);
* ``"shafer"`` Shafer's model: only the singleton regions exist (exclusive hypotheses; D^Theta = 2^Theta);
* ``"hybrid"`` a hybrid DSm model: the integrity constraints ``empty=[("A", "B"), ...]`` declare the listed
  intersections empty, which removes every region containing the listed atoms.

An element is empty *under the model* when none of its regions exists; this is the conflict of the
combination rules.

Rules (all for s >= 2 sources, simultaneous)
--------------------------------------------
conjunctive  the conjunctive consensus.  On the free model it is the classic DSm rule (DSmC) and never produces
             conflict; on other models the mass of empty intersections is kept on the empty set, which is Smets'
             unnormalised rule of the transferable belief model (``tbm`` is an alias).
dempster     conjunctive consensus normalised by 1 - K (undefined at total conflict K = 1).
yager        the conflict goes to the total ignorance Theta (Yager 1987).
dubois_prade each conflicting product goes to the union of the focal elements that produced it (Dubois and Prade
             1988; for non-degenerate inputs this is the hybrid DSm rule DSmH).
pcr5         proportional conflict redistribution no. 5 (Smarandache and Dezert 2006, general s-source form):
             a conflicting product goes back to the focal elements that appear in the canonical form of the
             conflict, in proportion to the *product* of the masses committed to each of them.
pcr6         PCR6 (Martin and Osswald 2006, Eq. 18): a conflicting product goes back to the focal elements of
             the tuple in proportion to the *sum* of the masses committed to each.  PCR5 = PCR6 for two sources.
pcr6_plus    PCR6+ (Dezert, Dezert and Smarandache 2021): PCR6 restricted to the elements whose binary keeping
             index is one, which restores the neutrality of the vacuous belief assignment.
sequential   pairwise fusion in a given order (PCR5/PCR6 are not associative; Dempster's rule is).

Decision: ``betp`` (generalised pignistic transform, DSm cardinality), ``dsmp`` (DSmP_epsilon of Dezert and
Smarandache), ``belief`` and ``plausibility``.

The binary frame and RNEL (book, Chapter "Neutrosophic evidence and DSmT")
--------------------------------------------------------------------------
On Theta = {x, not_x} under the free model D^Theta = {empty, x & not_x, x, not_x, x | not_x}.  The dictionary is

    m(x) = T,  m(not_x) = F,  m(x & not_x) = C (contradiction, a glut),  m(x | not_x) = U + G (ignorance),
    m(empty) = N (ill-posed: "none of the hypotheses", the open-world reading of the TBM),

for the tuple of one body of reports (total 1).  Shafer's model is the case C = 0, where the mass is the
Subjective Logic opinion (b, d, u) = (T, F, G) (Josang's correspondence).  The split of the ignorance mass into
U and G is not identifiable from a mass: ``bba_to_tuple`` returns it as G unless ``u_share`` is given.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

from .sl import DEFAULT_W, Opinion
from .tuple import Reports, RNELTuple, rnel_tuple

Region = tuple  # sorted tuple of atom indices
Element = frozenset  # frozenset of regions
BBA = dict

TOL = 1e-12

__all__ = [
    "Frame", "binary_frame", "TotalConflictError",
    "conjunctive", "tbm", "dsmc", "dempster", "yager", "dubois_prade", "pcr5", "pcr6", "pcr6_plus",
    "sequential", "conflict", "pcr5_shares", "pcr6_shares", "RULES", "combine_all",
    "belief", "plausibility", "betp", "dsmp",
    "tuple_to_bba", "bba_to_tuple", "reports_to_bba", "opinion_to_bba", "bba_to_triple", "triple_to_bba",
    "belnap_probability",
]


class TotalConflictError(ValueError):
    """Dempster's rule is undefined when the conflict K equals 1."""


# ============================================================================ frames
class Frame:
    """A finite frame with a DSm model.

    >>> fr = Frame(["A", "B", "C"], model="shafer")
    >>> m1 = fr.bba({"A": 0.9, "C": 0.1}); m2 = fr.bba({"B": 0.9, "C": 0.1})
    >>> fr.pretty(dempster(fr, [m1, m2]))
    {'C': 1.0}
    """

    def __init__(self, atoms: Sequence[str], model: str = "free", empty: Iterable[Iterable[str]] = ()):
        atoms = tuple(str(a) for a in atoms)
        if len(atoms) < 1 or len(set(atoms)) != len(atoms):
            raise ValueError("atoms must be distinct and non-empty")
        if model not in ("free", "shafer", "hybrid"):
            raise ValueError("model must be 'free', 'shafer' or 'hybrid'")
        self.atoms = atoms
        self.model = model
        n = len(atoms)
        all_regions = [tuple(c) for k in range(1, n + 1) for c in itertools.combinations(range(n), k)]
        constraints = [frozenset(self.index(a) for a in group) for group in empty]
        if model == "shafer":
            if constraints:
                raise ValueError("Shafer's model already declares every intersection empty")
            allowed = [r for r in all_regions if len(r) == 1]
        elif model == "free":
            if constraints:
                raise ValueError("the free model has no integrity constraints; use model='hybrid'")
            allowed = all_regions
        else:
            if not constraints or any(len(c) < 2 for c in constraints):
                raise ValueError("a hybrid model needs constraints, each naming at least two atoms")
            allowed = [r for r in all_regions if not any(c <= set(r) for c in constraints)]
        self.constraints = tuple(constraints)
        self.regions = tuple(allowed)
        self.theta: Element = frozenset(self.regions)
        self.empty: Element = frozenset()

    # ------------------------------------------------------------------ elements
    def index(self, atom: str) -> int:
        try:
            return self.atoms.index(atom)
        except ValueError:
            raise KeyError(f"unknown atom {atom!r}; atoms are {self.atoms}") from None

    def atom(self, name: str) -> Element:
        i = self.index(name)
        return frozenset(r for r in self.regions if i in r)

    def element(self, spec) -> Element:
        """Build an element from a string such as ``"A"``, ``"A|B"``, ``"A&B"``, ``"(A|B)&C"`` (``&``/``∩``
        bind tighter than ``|``/``∪``; ``"Theta"``/``"Θ"`` is the total ignorance, ``"empty"``/``"∅"`` the empty
        set), from a set of atom names (read as their union, the usual power-set notation), or from an element."""
        if isinstance(spec, frozenset) and all(isinstance(r, tuple) for r in spec):
            bad = [r for r in spec if r not in self.regions]
            if bad:
                raise ValueError(f"regions {bad} do not exist under the {self.model} model")
            return spec
        if isinstance(spec, (set, frozenset, list, tuple)) and not isinstance(spec, str):
            out = frozenset()
            for a in spec:
                out = out | self.atom(a)
            return out
        if not isinstance(spec, str):
            raise TypeError("an element is a string, a set of atom names or a frozenset of regions")
        return _Parser(self, spec).parse()

    def label(self, X: Element, ascii: bool = False) -> str:
        """Canonical label: the union of the minimal intersections that generate X (disjunctive normal form)."""
        if not X:
            return "empty" if ascii else "∅"
        if X == self.theta:
            return "Theta" if ascii else "Θ"
        mins = [r for r in X if not any(set(s) < set(r) for s in X)]
        mins.sort(key=lambda r: (len(r), r))
        cap, cup = ("&", "|") if ascii else ("∩", "∪")
        parts = [cap.join(self.atoms[i] for i in r) for r in mins]
        if len(parts) > 1:
            parts = [f"({p})" if cap in p else p for p in parts]
        return cup.join(parts)

    def hyper_power_set(self) -> list:
        """All elements of the fusion space (closure of the atoms under union and intersection), with the empty
        set.  Under the free model its size is 5, 19 and 167 for n = 2, 3 and 4 atoms."""
        elems = {self.atom(a) for a in self.atoms}
        while True:
            new = set(elems)
            for X, Y in itertools.product(list(elems), repeat=2):
                new.add(X | Y)
                new.add(X & Y)
            if new == elems:
                break
            elems = new
        elems.add(self.empty)
        return sorted(elems, key=lambda X: (len(X), sorted(X)))

    # ------------------------------------------------------------------ masses
    def bba(self, masses: Mapping, *, open_world: bool = False, tol: float = 1e-9) -> BBA:
        """A basic belief assignment from {element spec: mass}.  Masses must be non-negative and sum to one.
        Mass on the empty set is accepted only with ``open_world=True`` (Smets' open world)."""
        out: defaultdict = defaultdict(float)
        for k, v in masses.items():
            v = float(v)
            if v < -tol:
                raise ValueError("a basic belief assignment cannot have negative mass")
            if v <= tol:
                continue
            X = self.element(k)
            if not X and not open_world:
                raise ValueError(f"{k!r} is empty under the {self.model} model; pass open_world=True to keep it")
            out[X] += v
        s = math.fsum(out.values())
        if abs(s - 1.0) > tol:
            raise ValueError(f"masses sum to {s}, not 1")
        return dict(out)

    def pretty(self, m: Mapping, digits: Optional[int] = None, ascii: bool = False) -> dict:
        """{label: mass}, largest first; masses rounded if ``digits`` is given."""
        items = sorted(m.items(), key=lambda kv: (-kv[1], self.label(kv[0], ascii)))
        return {self.label(k, ascii): (round(v, digits) if digits is not None else v) for k, v in items if v > TOL}

    def get(self, m: Mapping, spec) -> float:
        """Mass of one element (0 if it is not focal)."""
        return float(m.get(self.element(spec), 0.0))

    def __repr__(self):
        cons = f", empty={[tuple(self.atoms[i] for i in sorted(c)) for c in self.constraints]}" if self.constraints else ""
        return f"Frame({list(self.atoms)}, model={self.model!r}{cons})"


class _Parser:
    def __init__(self, frame: Frame, text: str):
        self.f = frame
        self.toks = self._tokenise(text)
        self.i = 0

    @staticmethod
    def _tokenise(text):
        toks, buf = [], ""
        for ch in text:
            if ch in "|&()∪∩":
                if buf.strip():
                    toks.append(buf.strip())
                buf = ""
                toks.append({"∪": "|", "∩": "&"}.get(ch, ch))
            elif ch.isspace():
                if buf.strip():
                    toks.append(buf.strip())
                buf = ""
            else:
                buf += ch
        if buf.strip():
            toks.append(buf.strip())
        return toks

    def parse(self) -> Element:
        X = self._union()
        if self.i != len(self.toks):
            raise ValueError(f"cannot parse element: unexpected {self.toks[self.i]!r}")
        return X

    def _peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _union(self):
        X = self._inter()
        while self._peek() == "|":
            self.i += 1
            X = X | self._inter()
        return X

    def _inter(self):
        X = self._atom()
        while self._peek() == "&":
            self.i += 1
            X = X & self._atom()
        return X

    def _atom(self):
        t = self._peek()
        if t is None:
            raise ValueError("cannot parse element: unexpected end")
        self.i += 1
        if t == "(":
            X = self._union()
            if self._peek() != ")":
                raise ValueError("cannot parse element: missing ')'")
            self.i += 1
            return X
        if t in ("Theta", "Θ", "Omega", "Ω"):
            return self.f.theta
        if t in ("empty", "∅"):
            return self.f.empty
        return self.f.atom(t)


def binary_frame(model: str = "free", names: tuple = ("x", "not_x")) -> Frame:
    """The frame {x, not x}.  Under the free model D^Theta = {∅, x∩not_x, x, not_x, x∪not_x}."""
    return Frame(list(names), model=model)


# ============================================================================ combination
def _check(frame: Frame, bbas: Sequence[Mapping], allow_empty: bool) -> list:
    if len(bbas) < 1:
        raise ValueError("at least one basic belief assignment is required")
    out = []
    for m in bbas:
        mm = {}
        for X, v in m.items():
            X = frame.element(X)
            if v < -TOL:
                raise ValueError("negative mass")
            if v <= TOL:
                continue
            if not X and not allow_empty:
                raise ValueError("this rule needs closed-world inputs (no mass on the empty set)")
            mm[X] = mm.get(X, 0.0) + float(v)
        if abs(math.fsum(mm.values()) - 1.0) > 1e-9:
            raise ValueError("each basic belief assignment must sum to 1")
        out.append(mm)
    return out


def _tuples(bbas):
    """Yield (focal elements, masses, product, intersection) for every focal tuple."""
    for combo in itertools.product(*(list(m.items()) for m in bbas)):
        focals = tuple(c[0] for c in combo)
        vals = tuple(c[1] for c in combo)
        inter = focals[0]
        for X in focals[1:]:
            inter = inter & X
        yield focals, vals, math.prod(vals), inter


def _clean(out) -> BBA:
    return {k: v for k, v in out.items() if v > TOL}


def conjunctive(frame: Frame, bbas: Sequence[Mapping]) -> BBA:
    """Conjunctive consensus m(X) = sum over X_1 ∩ ... ∩ X_s = X of the products of masses.  The mass of
    tuples whose intersection is empty under the model stays on the empty set (Smets' rule).  On the free model
    nothing is empty and this is the classic DSm rule."""
    bb = _check(frame, bbas, allow_empty=True)
    out: defaultdict = defaultdict(float)
    for _, _, p, inter in _tuples(bb):
        out[inter] += p
    return _clean(out)


tbm = conjunctive
dsmc = conjunctive


def conflict(frame: Frame, bbas: Sequence[Mapping]) -> float:
    """K: the conjunctive mass of the empty set under the model."""
    return conjunctive(frame, bbas).get(frame.empty, 0.0)


def dempster(frame: Frame, bbas: Sequence[Mapping]) -> BBA:
    """Dempster's rule: conjunctive consensus normalised by 1 - K."""
    c = conjunctive(frame, bbas)
    K = c.pop(frame.empty, 0.0)
    if K >= 1.0 - 1e-12:
        raise TotalConflictError("Dempster's rule is undefined at total conflict (K = 1)")
    return _clean({X: v / (1.0 - K) for X, v in c.items()})


def yager(frame: Frame, bbas: Sequence[Mapping]) -> BBA:
    """Yager's rule: the conflict K is added to the total ignorance Theta."""
    c = conjunctive(frame, _check(frame, bbas, allow_empty=False))
    K = c.pop(frame.empty, 0.0)
    c[frame.theta] = c.get(frame.theta, 0.0) + K
    return _clean(c)


def dubois_prade(frame: Frame, bbas: Sequence[Mapping]) -> BBA:
    """Dubois-Prade rule: each conflicting product goes to the union of its focal elements."""
    bb = _check(frame, bbas, allow_empty=False)
    out: defaultdict = defaultdict(float)
    for focals, _, p, inter in _tuples(bb):
        if inter:
            out[inter] += p
        else:
            u = frozenset().union(*focals)
            out[u if u else frame.theta] += p
    return _clean(out)


def _canonical_members(focals) -> list:
    """Distinct focal elements of a conflicting tuple that appear in the canonical (conjunctive normal) form of
    their intersection: an element that contains another element of the tuple is absorbed (X ∩ Y = X)."""
    distinct = list(dict.fromkeys(focals))
    return [X for X in distinct if not any(Y < X for Y in distinct)]


def pcr5_shares(focals: Sequence, vals: Sequence[float]) -> dict:
    """PCR5 redistribution of one conflicting product m_1(X_1)...m_s(X_s): element Y of the canonical form
    receives the product times w_Y / sum_Z w_Z, with w_Y the product of the masses committed to Y."""
    p = math.prod(vals)
    members = _canonical_members(focals)
    w = {Y: math.prod(v for X, v in zip(focals, vals) if X == Y) for Y in members}
    s = math.fsum(w.values())
    return {Y: p * wy / s for Y, wy in w.items()}


def pcr6_shares(focals: Sequence, vals: Sequence[float]) -> dict:
    """PCR6 redistribution of one conflicting product: element Y receives the product times
    (sum of the masses committed to Y) / (sum of all the masses of the tuple)."""
    p = math.prod(vals)
    s = math.fsum(vals)
    out: defaultdict = defaultdict(float)
    for X, v in zip(focals, vals):
        out[X] += p * v / s
    return dict(out)


def pcr5(frame: Frame, bbas: Sequence[Mapping]) -> BBA:
    """PCR5 (Smarandache and Dezert 2006), simultaneous form for s >= 2 sources.

    For a conflicting tuple (X_1, ..., X_s) with product p, each element Y of the canonical form of the conflict
    receives p * w_Y / sum_Z w_Z with w_Y = product of the masses m_i(X_i) of the sources with X_i = Y.  For two
    sources this is Eq. (32) of Smarandache and Dezert, m1(X)^2 m2(Y)/(m1(X)+m2(Y)) + m2(X)^2 m1(Y)/(m2(X)+m1(Y))."""
    bb = _check(frame, bbas, allow_empty=False)
    out: defaultdict = defaultdict(float)
    for focals, vals, p, inter in _tuples(bb):
        if inter:
            out[inter] += p
            continue
        for Y, share in pcr5_shares(focals, vals).items():
            out[Y] += share
    return _clean(out)


def pcr6(frame: Frame, bbas: Sequence[Mapping]) -> BBA:
    """PCR6 (Martin and Osswald 2006, Eq. 18), simultaneous form for s >= 2 sources: for a conflicting tuple
    with product p, the focal element X_i of source i receives p * m_i(X_i) / sum_j m_j(X_j)."""
    bb = _check(frame, bbas, allow_empty=False)
    out: defaultdict = defaultdict(float)
    for focals, vals, p, inter in _tuples(bb):
        if inter:
            out[inter] += p
            continue
        for X, share in pcr6_shares(focals, vals).items():
            out[X] += share
    return _clean(out)


def _keeping_indices(focals) -> dict:
    """Binary keeping indices of PCR6+ (Dezert, Dezert and Smarandache 2021, Eq. 23), with |X| the DSm
    cardinality.  Same logic as the reference implementation of the PCR6-RAG study."""
    distinct = tuple(dict.fromkeys(focals))
    res = {}
    for xi in distinct:
        all_contained, seen = True, False
        for small in distinct:
            for large in distinct:
                if small == large:
                    continue
                if len(xi) <= len(large) and len(small) <= len(large):
                    seen = True
                    if not small <= large:
                        all_contained = False
                        break
            if not all_contained:
                break
        res[xi] = 0 if (seen and all_contained) else 1
    return res


def pcr6_plus(frame: Frame, bbas: Sequence[Mapping]) -> BBA:
    """PCR6+ (Dezert, Dezert and Smarandache 2021, Eq. 26): PCR6 restricted to elements with keeping index 1."""
    bb = _check(frame, bbas, allow_empty=False)
    out: defaultdict = defaultdict(float)
    for focals, vals, p, inter in _tuples(bb):
        if inter:
            out[inter] += p
            continue
        keep = _keeping_indices(focals)
        w: defaultdict = defaultdict(float)
        for X, v in zip(focals, vals):
            if keep[X]:
                w[X] += v
        s = math.fsum(w.values())
        for X, wx in w.items():
            out[X] += p * wx / s
    return _clean(out)


RULES: dict = {
    "conjunctive": conjunctive, "dempster": dempster, "yager": yager, "dubois_prade": dubois_prade,
    "pcr5": pcr5, "pcr6": pcr6, "pcr6_plus": pcr6_plus,
}


def sequential(rule: Callable, frame: Frame, bbas: Sequence[Mapping], order: Optional[Sequence[int]] = None) -> BBA:
    """Fuse pairwise in the given order: rule(rule(m_a, m_b), m_c) ...  For Dempster's and the conjunctive rule
    the result does not depend on the order; for PCR5, PCR6, Yager and Dubois-Prade it can."""
    order = list(range(len(bbas))) if order is None else list(order)
    acc = bbas[order[0]]
    for k in order[1:]:
        acc = rule(frame, [acc, bbas[k]])
    return acc


def combine_all(frame: Frame, bbas: Sequence[Mapping], rules: Optional[Iterable[str]] = None) -> dict:
    """{rule name: fused BBA} for the named rules (default: all of ``RULES``); a rule that is undefined on the
    input (Dempster at K = 1) maps to None."""
    out = {}
    for name in (rules or RULES):
        try:
            out[name] = RULES[name](frame, bbas)
        except TotalConflictError:
            out[name] = None
    return out


# ============================================================================ decision
def belief(frame: Frame, m: Mapping, spec) -> float:
    """Generalised belief Bel(X) = sum of m(Y) over non-empty Y ⊆ X (Y ∩ X need not be empty on the free model)."""
    X = frame.element(spec)
    return math.fsum(v for Y, v in m.items() if Y and Y <= X)


def plausibility(frame: Frame, m: Mapping, spec) -> float:
    """Generalised plausibility Pl(X) = sum of m(Y) over Y with Y ∩ X non-empty under the model."""
    X = frame.element(spec)
    return math.fsum(v for Y, v in m.items() if Y & X)


def betp(frame: Frame, m: Mapping) -> dict:
    """Generalised pignistic transform on the atoms, GPT(theta) = sum_Y C(theta ∩ Y)/C(Y) m(Y), with C the DSm
    cardinality (number of existing regions) and mass on the empty set removed by normalisation (Smets).  On
    Shafer's model it is BetP and sums to one; on the free model the atoms overlap and the values need not."""
    m0 = float(m.get(frame.empty, 0.0))
    if m0 >= 1.0 - TOL:
        raise ValueError("BetP is undefined when all the mass is on the empty set")
    out = {}
    for a in frame.atoms:
        A = frame.atom(a)
        out[a] = math.fsum(len(A & Y) / len(Y) * v for Y, v in m.items() if Y) / (1.0 - m0)
    return out


def dsmp(frame: Frame, m: Mapping, epsilon: float = 0.001) -> dict:
    """DSmP_epsilon of Dezert and Smarandache on the atoms:
    DSmP(theta) = sum_Y [ sum_{Z ⊆ theta ∩ Y, C(Z)=1} m(Z) + eps C(theta ∩ Y) ] / [ sum_{Z ⊆ Y, C(Z)=1} m(Z) + eps C(Y) ] m(Y).
    On Shafer's model the mass of a compound element is shared among its singletons in proportion to their own
    masses plus epsilon."""
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    unit = {X: v for X, v in m.items() if len(X) == 1}

    def unit_mass(S):
        return math.fsum(v for Z, v in unit.items() if Z <= S)

    out = {}
    for a in frame.atoms:
        A = frame.atom(a)
        tot = 0.0
        for Y, v in m.items():
            if not Y:
                continue
            num = unit_mass(A & Y) + epsilon * len(A & Y)
            den = unit_mass(Y) + epsilon * len(Y)
            if den <= 0:  # epsilon = 0 and no unit mass inside Y: share by cardinality
                num, den = len(A & Y), len(Y)
            tot += num / den * v
        out[a] = tot
    m0 = float(m.get(frame.empty, 0.0))
    if m0:
        out = {a: v / (1.0 - m0) for a, v in out.items()}
    return out


# ============================================================================ RNEL dictionary (binary frame)
def _bin(frame: Optional[Frame], model: str = "free") -> Frame:
    fr = frame or binary_frame(model)
    if len(fr.atoms) != 2:
        raise ValueError("the RNEL dictionary is defined on a binary frame {x, not_x}")
    return fr


def tuple_to_bba(tp: RNELTuple, *, frame: Optional[Frame] = None, extra: str = "ignorance", tol: float = 1e-9) -> BBA:
    """RNEL tuple of one body of reports -> DSm masses on the free binary frame:
    m(x)=T, m(not_x)=F, m(x∩not_x)=C, m(x∪not_x)=U+G (+ the extra indeterminacies I_k if ``extra='ignorance'``),
    m(∅)=N (open world).  Requires the tuple to sum to one; a fused tuple of Definition 8.4 whose contradiction
    is read across sources (total > 1) is not a mass assignment and is refused."""
    fr = _bin(frame)
    if fr.model != "free" and tp.C > tol:
        raise ValueError("a tuple with C > 0 needs the free model (x ∩ not_x must exist)")
    if abs(tp.total - 1.0) > tol:
        raise ValueError(f"the tuple sums to {tp.total:.6g}; only a tuple of one body of reports (total 1) is a BBA")
    if extra not in ("ignorance", "refuse"):
        raise ValueError("extra must be 'ignorance' or 'refuse'")
    if extra == "refuse" and sum(tp.I) > tol:
        raise ValueError("the tuple has extra indeterminacy components I_k")
    x, nx = fr.atoms
    spec = {x: tp.T, nx: tp.F, "Theta": tp.U + tp.G + sum(tp.I), "empty": tp.N}
    if tp.C > tol:
        spec[f"{x}&{nx}"] = tp.C
    return fr.bba(spec, open_world=True)


def reports_to_bba(r: Reports, W: float = DEFAULT_W, **kw) -> BBA:
    """Typed report counts -> RNEL tuple (Definition 8.1) -> DSm masses (``tuple_to_bba``)."""
    return tuple_to_bba(rnel_tuple(r, W), **kw)


def opinion_to_bba(op: Opinion, frame: Optional[Frame] = None) -> BBA:
    """Subjective Logic binomial opinion (b, d, u) -> Shafer mass m(x)=b, m(not_x)=d, m(Theta)=u (Josang)."""
    fr = frame or binary_frame("shafer")
    x, nx = fr.atoms
    return fr.bba({x: op.b, nx: op.d, "Theta": op.u})


def bba_to_tuple(m: Mapping, *, frame: Optional[Frame] = None, u_share: float = 0.0) -> RNELTuple:
    """DSm masses on a binary frame -> RNEL tuple (T, C, U, N, G, F) with T=m(x), F=m(not_x), C=m(x∩not_x),
    N=m(∅) and the ignorance m(x∪not_x) split as U = u_share*m, G = (1-u_share)*m.  The split is a choice: a mass
    assignment does not distinguish undetermined reports (U) from the absence of reports (G)."""
    fr = _bin(frame)
    if not 0.0 <= u_share <= 1.0:
        raise ValueError("u_share must lie in [0, 1]")
    x, nx = fr.atoms
    X, NX = fr.atom(x), fr.atom(nx)
    known = {X, NX, fr.theta, fr.empty, X & NX}
    if any(k not in known for k in m):
        raise ValueError("unexpected focal element for a binary frame")
    ign = float(m.get(fr.theta, 0.0))
    C = float(m.get(X & NX, 0.0)) if (X & NX) else 0.0
    return RNELTuple(T=float(m.get(X, 0.0)), C=C, U=u_share * ign, N=float(m.get(fr.empty, 0.0)),
                     G=(1.0 - u_share) * ign, F=float(m.get(NX, 0.0)))


def belnap_probability(m: Mapping, frame: Optional[Frame] = None) -> dict:
    """Free binary mass -> probability on the Belnap frame {t, f, b, n}: t=m(x), f=m(not_x), b=m(x∩not_x) (told
    both, the glut), n=m(x∪not_x).  Then (P(E_T), P(E_I), P(E_F)) with E_T={t,b}, E_I={n}, E_F={f,b} is
    (Bel(x), m(x∪not_x), Bel(not_x)) (Proposition on the credal reading)."""
    fr = _bin(frame)
    if fr.empty in m and m[fr.empty] > TOL:
        raise ValueError("closed-world masses only")
    x, nx = fr.atoms
    X, NX = fr.atom(x), fr.atom(nx)
    return {"t": float(m.get(X, 0.0)), "f": float(m.get(NX, 0.0)),
            "b": float(m.get(X & NX, 0.0)) if (X & NX) else 0.0, "n": float(m.get(fr.theta, 0.0))}


def bba_to_triple(m: Mapping, frame: Optional[Frame] = None) -> tuple:
    """Credal reading tau(m) = (Bel(x), m(x∪not_x), Bel(not_x)) of a closed-world binary mass."""
    fr = _bin(frame)
    p = belnap_probability(m, fr)
    return (p["t"] + p["b"], p["n"], p["f"] + p["b"])


def triple_to_bba(T: float, I: float, F: float, frame: Optional[Frame] = None, tol: float = 1e-12) -> BBA:
    """Inverse of ``bba_to_triple`` on its image R = {T+I <= 1, F+I <= 1, T+I+F >= 1}:
    m(x∪not_x)=I, m(x∩not_x)=T+I+F-1, m(x)=1-F-I, m(not_x)=1-T-I."""
    fr = _bin(frame)
    if T + I > 1 + tol or F + I > 1 + tol or T + I + F < 1 - tol:
        raise ValueError("the triple is outside the image R = {T+I<=1, F+I<=1, T+I+F>=1} of the free binary masses")
    g = max(0.0, T + I + F - 1.0)
    if g > tol and fr.model != "free":
        raise ValueError("T+I+F > 1 needs the free model (glut x∩not_x)")
    x, nx = fr.atoms
    spec = {x: max(0.0, 1 - F - I), nx: max(0.0, 1 - T - I), "Theta": I}
    if g > tol:
        spec[f"{x}&{nx}"] = g
    return fr.bba(spec)
