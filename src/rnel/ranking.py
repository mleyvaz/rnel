"""Total-order ranking cascades (Smarandache 2017, Critical Review XIV, [4] and [5])."""
from __future__ import annotations

from typing import Mapping


def tinf_key(x: Mapping[str, float]) -> tuple[float, ...]:
    """(T, I, N, F): score, accuracy, extended certainty, certainty (larger is better)."""
    T, I, N, F = x["T"], x["I"], x["N"], x["F"]
    return ((2 + T + N - I - F) / 4, T + N - F, T + N, T)


def mu_key(x: Mapping[str, float]) -> tuple[float, ...]:
    """(T, I, N, U1..Un, F): s, a, ec(n), ..., ec(0), c, as restated in Definition 3.3.

    The score constant n + 2 is inferred from the n = 0 case (2 + T + N - I - F)/4; confirm against [5].
    """
    U = [x[f"U{k}"] for k in range(1, 1 + sum(1 for c in x if c.startswith("U")))]
    n = len(U)
    T, I, N, F = x["T"], x["I"], x["N"], x["F"]
    s = (n + 2 + T + N - I - sum(U) - F) / (n + 4)
    a = T + N - F - sum(U)
    ecs = [T + N - sum(U[:j]) for j in range(n, -1, -1)]
    return (s, a, *ecs, T)


def rank(tuples: list[Mapping[str, float]], key=tinf_key) -> list[int]:
    """Indices of the tuples from best to worst (lexicographic cascade)."""
    return sorted(range(len(tuples)), key=lambda i: key(tuples[i]), reverse=True)
