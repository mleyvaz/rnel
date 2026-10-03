"""Common style and helpers for the CRC figures of the book (Smarandache & Leyva-Vazquez).

House style (one module for every figure, so the book reads as one system):
  - serif text matching Times (Times New Roman; STIX for mathematics), 9 pt labels, 8 pt ticks;
  - 7 x 10 in trim: full width 5.5 in, half width 2.65 in;
  - grayscale-safe: every series differs by line style, marker or hatching, never by colour alone;
    the four colour slots (ink, blue, orange, aqua) are those of the dataviz reference palette, whose first three
    slots pass the all-pairs CVD check (validate_palette.js), and ink is black;
  - no titles inside the image (the caption carries it); hairline solid grid; left/bottom spines only;
  - output: vector PDF (TrueType fonts embedded), 600-dpi TIFF (LZW), PNG preview, and a grayscale PNG check copy.
Every figure script writes the numbers it plots to figures/data/<name>.json (book edition: under the output dir).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.getcwd(), "figures")  # book edition: figures go to <output dir>/figures
DATA = os.path.join(OUT, "data")
GRAY = os.path.join(OUT, "grayscale")
for d in (OUT, DATA, GRAY):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- read-only sources
BOOKDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # the book/ folder


def book_result(name, expected_rel):
    """Book edition: a result file of another book script. The fresh copy in the current (output) directory is used
    when present (same regenerate run); otherwise the stored expected copy under book/expected/."""
    p = os.path.join(os.getcwd(), name)
    return p if os.path.exists(p) else os.path.join(BOOKDIR, expected_rel)


CASE = os.path.join(BOOKDIR, "data", "cached", "paracetamol_case")  # CACHED summaries (see data/cached/README.md)
EXP = os.path.join(BOOKDIR, "data", "cached", "experiments")        # CACHED summaries (see data/cached/README.md)
# rnel is imported from PYTHONPATH (set by `python -m rnel.book regenerate`)

# ---------------------------------------------------------------- palette and encodings
INK = "#0b0b0b"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK2 = "#52514e"      # secondary ink
MUTED = "#898781"     # axis labels / muted
GRID = "#e1e0d9"      # hairline grid
FILL1 = "#c3c2b7"     # neutral fills (light gray)
FILL2 = "#e1e0d9"
SERIES = [INK, BLUE, ORANGE, AQUA]
LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D"]
HATCHES = ["", "///", "...", "xxx", "\\\\\\", "++"]

FULL = 5.5
HALF = 2.65

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "lines.linewidth": 1.2,
    "lines.markersize": 4.5,
    "legend.frameon": False,
    "hatch.linewidth": 0.5,
    "hatch.color": INK2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "figure.dpi": 100,
})


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def save(fig, name, data=None):
    """Write PDF + 600-dpi TIFF + PNG preview + grayscale PNG, and the plotted numbers as JSON."""
    base = os.path.join(OUT, name)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    from PIL import Image
    Image.open(base + ".png").convert("L").save(os.path.join(GRAY, name + "_gray.png"))
    if data is not None:
        json.dump(data, open(os.path.join(DATA, name + ".json"), "w", encoding="utf-8"), indent=1,
                  ensure_ascii=False, default=float)
    plt.close(fig)
    print("wrote", name)


def r(x, k=4):
    """Round for the JSON record (the plot itself uses full precision)."""
    if isinstance(x, (list, tuple)):
        return [r(v, k) for v in x]
    return float(round(float(x), k))


# ---------------------------------------------------------------- small LP helpers (credal sets on finite frames)
def lp_min(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, n=None):
    from scipy.optimize import linprog
    n = n or len(c)
    if A_eq is None:
        A_eq, b_eq = [np.ones(n)], [1.0]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n, method="highs")
    return (res.fun if res.status == 0 else None), res


def lifting_envelope(patterns, x):
    """Lower envelope of K(x) = {P on atoms with pattern bits: P(E_k) >= x_k} on the events E_k; None if empty."""
    patterns = np.array(patterns, dtype=float)
    n, m = patterns.shape
    A_ub = -patterns.T  # -P(E_k) <= -x_k
    b_ub = -np.array(x, dtype=float)
    out = []
    for k in range(m):
        v, res = lp_min(patterns[:, k], A_ub, b_ub, n=n)
        if v is None:
            return None
        out.append(v)
    return out


MINIMAL = [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]   # (T, I, F) bits: Theorem 8(a), m = 3
BELNAP = [(1, 0, 0), (0, 0, 1), (1, 0, 1), (0, 1, 0)]    # t, f, b, n with E_I = {n}: Theorem 8(d)
DISJOINT = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]             # Shafer's model: Theorem 8(d)


# ---------------------------------------------------------------- copulas (as in book_v11 .../nnorm_choice.py)
def Pi(u, v): return u * v
def Mc(u, v): return min(u, v)
def Wc(u, v): return max(0.0, u + v - 1.0)


def frank(th):
    import math
    if th == 0:
        return Pi

    def Cpos(u, v, t):
        if u <= 0 or v <= 0:
            return 0.0
        m, Mx = min(u, v), max(u, v)
        arg = (1 + math.exp(-t * (Mx - m)) - math.exp(-t * Mx) - math.exp(-t * (1 - m))) / (-math.expm1(-t))
        return min(max(m - math.log(arg) / t, max(0.0, u + v - 1)), m)

    def C(u, v):
        if th > 0:
            return Cpos(u, v, th)
        return min(max(u - Cpos(u, 1 - v, -th), max(0.0, u + v - 1)), min(u, v))
    return C


def clayton(th):
    def C(u, v):
        if u <= 0 or v <= 0:
            return 0.0
        return max(u ** -th + v ** -th - 1.0, 0.0) ** (-1.0 / th)
    return C


def survival(C):
    return lambda u, v: u + v - 1.0 + C(1.0 - u, 1.0 - v)
