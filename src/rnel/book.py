"""Reproducibility runner for the book *Neutrosophic Evidence* (Smarandache & Leyva-Vazquez).

Usage::

    python -m rnel.book list [--book-dir PATH]
    python -m rnel.book regenerate [--book-dir PATH] [--only NAME[,NAME...]] [--out DIR]
                                   [--check] [--all] [--bless] [--timeout SECONDS]

The ``book/`` folder holds ``manifest.json``, the scripts that produce every number, table and figure
of the book (``book/scripts/...``), the stored expected outputs (``book/expected/...``) and cached
experiment summaries (``book/data/cached/...``). Each script is run as a subprocess whose working
directory is the output folder (default ``book/outputs/``); its standard output is written to
``<name>.out`` there. With ``--check`` the fresh outputs are compared with the expected ones
(whitespace-insensitive; numbers compared with a small tolerance) and a PASS/FAIL table is printed;
the exit code is non-zero when any check fails or any script errors.

The book folder is located by ``--book-dir``, else the environment variable ``RNEL_BOOK_DIR``, else by
walking up from the current directory and from this file looking for ``book/manifest.json``.

This module only uses the standard library and imports nothing heavy at import time.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

__all__ = ["find_book_dir", "load_manifest", "compare_text", "compare_json", "main"]

MANIFEST = "manifest.json"
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-9


# ----------------------------------------------------------------------------------------- locating
def _candidates(start: Path):
    start = start.resolve()
    for d in [start, *start.parents]:
        yield d / "book"
        if d.name == "book":
            yield d


def find_book_dir(book_dir: str | os.PathLike | None = None) -> Path:
    """Return the ``book/`` directory (the one containing ``manifest.json``)."""
    if book_dir:
        p = Path(book_dir).expanduser().resolve()
        if (p / MANIFEST).is_file():
            return p
        if (p / "book" / MANIFEST).is_file():
            return p / "book"
        raise FileNotFoundError(f"no {MANIFEST} in {p}")
    env = os.environ.get("RNEL_BOOK_DIR")
    if env:
        return find_book_dir(env)
    for start in (Path.cwd(), Path(__file__).parent):
        for c in _candidates(start):
            if (c / MANIFEST).is_file():
                return c.resolve()
    raise FileNotFoundError("book/manifest.json not found: pass --book-dir or set RNEL_BOOK_DIR")


def _auto_manifest(book: Path) -> list[dict]:
    """Discover executable book scripts and their recorded outputs using repository-relative paths."""
    helpers = {"common.py", "offlib.py", "re_common.py", "plan16.py"}
    special_json = {
        "verify_corrections": "corrections_checks.json",
        "open_problems": "v10_open_problems.json",
        "compute_numbers": "numbers.json",
    }
    entries = []
    for script in sorted((book / "scripts").rglob("*.py")):
        if script.name in helpers:
            continue
        rel = script.relative_to(book).as_posix()
        name = script.stem
        expected = sorted((book / "expected").rglob(name + ".out"))
        expected_files = {}
        produced = special_json.get(name, name + ".json")
        json_expected = sorted((book / "expected").rglob(produced))
        if json_expected:
            expected_files[produced] = json_expected[0].relative_to(book).as_posix()
        if name == "compute_numbers":
            extra = book / "expected/part6/numbers_sources.json"
            if extra.is_file():
                expected_files["numbers_sources.json"] = extra.relative_to(book).as_posix()
        requires = []
        if any(p in rel for p in ("figures_", "dossier/", "added_sections/", "ch10_plithogeny/", "exercises/")):
            requires = ["numpy", "scipy"]
        if "figures_" in rel:
            requires.append("matplotlib")
        if name == "numbers_torch":
            requires.append("torch")
        entries.append({
            "name": rel.removeprefix("scripts/").removesuffix(".py").replace("/", "__"),
            "section": script.parent.name,
            "script": rel,
            "expected": expected[0].relative_to(book).as_posix() if expected else None,
            "expected_files": expected_files,
            "uses_cache": any(p in rel for p in ("figures_", "part6/", "verify_corrections.py", "exercises/")),
            "slow": any(p in rel for p in ("p1_plithogenic.py", "open_problems.py", "t1_reduction_hierarchy.py")),
            "requires": requires,
            "ignore_keys": ["rnel.branch", "rnel.commit"] if name == "compute_numbers" else [],
        })
    return entries


def load_manifest(book: Path) -> list[dict]:
    data = json.loads((book / MANIFEST).read_text(encoding="utf-8"))
    if isinstance(data, dict) and data.get("auto_discover"):
        entries = _auto_manifest(book)
    else:
        entries = data["scripts"] if isinstance(data, dict) else data
    for e in entries:
        e.setdefault("expected", None)
        e.setdefault("expected_files", {})
        e.setdefault("uses_cache", False)
        e.setdefault("slow", False)
        e.setdefault("requires", [])
        e.setdefault("ignore_keys", [])
    return entries


# ----------------------------------------------------------------------------------------- comparing
_NUM = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
_TOKEN = re.compile(rf"{_NUM}|[A-Za-z_]+|\S")
_WRAP = re.compile(r"\bnp\.(?:float|int|bool_|str_)\w*\(([^()]*)\)")


def _tokens(text: str) -> list[str]:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"(?m)^OK\s+", "", text)  # presentation-only prefix used by older solution logs
    text = _WRAP.sub(r"\1", text)  # np.float64(0.5) -> 0.5 (numpy >= 2 repr)
    return _TOKEN.findall(text)


def _num(tok: str):
    if re.fullmatch(_NUM, tok):
        try:
            return float(tok)
        except ValueError:
            return None
    return None


def _close(a: float, b: float, rtol: float, atol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def compare_text(fresh: str, expected: str, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL):
    """Whitespace-insensitive comparison; numeric tokens compared with tolerance. Returns (ok, detail)."""
    a, b = _tokens(fresh), _tokens(expected)
    max_dev = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            continue
        fx, fy = _num(x), _num(y)
        if fx is not None and fy is not None and _close(fx, fy, rtol, atol):
            max_dev = max(max_dev, abs(fx - fy))
            continue
        ctx_a = " ".join(a[max(0, i - 6):i + 6])
        ctx_b = " ".join(b[max(0, i - 6):i + 6])
        return False, f"token {i}: got {x!r}, expected {y!r} | got: ...{ctx_a}... | expected: ...{ctx_b}..."
    if len(a) != len(b):
        return False, f"length differs: {len(a)} tokens vs {len(b)} expected"
    return True, ("identical" if max_dev == 0 else f"max numeric deviation {max_dev:.2e}")


def compare_json(fresh, expected, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL,
                 ignore_keys=(), path: str = "$"):
    """Recursive comparison with numeric tolerance. Returns (ok, detail)."""
    if isinstance(expected, dict) and isinstance(fresh, dict):
        ka = set(fresh) - set(ignore_keys)
        kb = set(expected) - set(ignore_keys)
        if ka != kb:
            return False, f"{path}: keys differ (missing {sorted(kb - ka)[:5]}, extra {sorted(ka - kb)[:5]})"
        for k in expected:
            if k in ignore_keys:
                continue
            ok, d = compare_json(fresh[k], expected[k], rtol, atol, ignore_keys, f"{path}.{k}")
            if not ok:
                return ok, d
        return True, "ok"
    if isinstance(expected, list) and isinstance(fresh, list):
        if len(expected) != len(fresh):
            return False, f"{path}: list length {len(fresh)} vs {len(expected)} expected"
        for i, (x, y) in enumerate(zip(fresh, expected)):
            ok, d = compare_json(x, y, rtol, atol, ignore_keys, f"{path}[{i}]")
            if not ok:
                return ok, d
        return True, "ok"
    num = (int, float)
    if isinstance(expected, num) and isinstance(fresh, num) and not isinstance(expected, bool) \
            and not isinstance(fresh, bool):
        if _close(float(fresh), float(expected), rtol, atol):
            return True, "ok"
        return False, f"{path}: got {fresh!r}, expected {expected!r}"
    if fresh == expected:
        return True, "ok"
    return False, f"{path}: got {fresh!r:.80}, expected {expected!r:.80}"


# ----------------------------------------------------------------------------------------- running
def _select(entries: list[dict], only: str | None, run_all: bool):
    chosen, skipped = [], []
    pats = [p.strip() for p in only.split(",")] if only else None
    for e in entries:
        if pats is not None:
            if any(fnmatch.fnmatch(e["name"], p) or fnmatch.fnmatch(e["script"], p) for p in pats):
                chosen.append(e)
            continue
        if e["slow"] and not run_all:
            skipped.append(e)
        else:
            chosen.append(e)
    return chosen, skipped


def _missing_requirements(reqs) -> list[str]:
    import importlib.util
    return [r for r in reqs if importlib.util.find_spec(r) is None]


def _env(book: Path) -> dict:
    env = dict(os.environ)
    paths = []
    repo_src = book.parent / "src"
    if (repo_src / "rnel" / "__init__.py").is_file():
        paths.append(str(repo_src))
    else:  # fall back to the directory that holds the imported rnel package
        paths.append(str(Path(__file__).resolve().parents[1]))
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONHASHSEED"] = "0"
    env.setdefault("PYTHONUTF8", "1")
    return env


def _run_one(e: dict, book: Path, out: Path, env: dict, timeout: float | None) -> dict:
    script = (book / e["script"]).resolve()
    t0 = time.perf_counter()
    try:
        cp = subprocess.run([sys.executable, str(script)], cwd=str(out), env=env, capture_output=True,
                            timeout=timeout)
        rc, so, se = cp.returncode, cp.stdout, cp.stderr
    except subprocess.TimeoutExpired as ex:
        rc, so, se = -999, ex.stdout or b"", (ex.stderr or b"") + b"\nTIMEOUT"
    dt = time.perf_counter() - t0
    stdout = so.decode("utf-8", errors="replace").replace("\r\n", "\n")
    stderr = se.decode("utf-8", errors="replace").replace("\r\n", "\n")
    (out / f"{e['name']}.out").write_text(stdout, encoding="utf-8")
    if stderr.strip():
        (out / f"{e['name']}.err").write_text(stderr, encoding="utf-8")
    return dict(returncode=rc, seconds=round(dt, 2), stderr_tail=stderr.strip().splitlines()[-3:])


def _check_one(e: dict, book: Path, out: Path, rtol: float, atol: float):
    details, ok_all, compared = [], True, 0
    if e.get("expected"):
        exp = book / e["expected"]
        fresh = (out / f"{e['name']}.out").read_text(encoding="utf-8")
        ok, d = compare_text(fresh, exp.read_text(encoding="utf-8", errors="replace"), rtol, atol)
        compared += 1
        ok_all &= ok
        details.append(("stdout: " + d) if not ok else f"stdout {d}")
    for produced, expected in e.get("expected_files", {}).items():
        fp, ep = out / produced, book / expected
        compared += 1
        if not fp.is_file():
            ok_all = False
            details.append(f"{produced}: not produced")
            continue
        if produced.endswith(".json"):
            try:
                ok, d = compare_json(json.loads(fp.read_text(encoding="utf-8")),
                                     json.loads(ep.read_text(encoding="utf-8")), rtol, atol,
                                     e.get("ignore_keys", ()))
            except json.JSONDecodeError:
                ok, d = compare_text(fp.read_text(encoding="utf-8"), ep.read_text(encoding="utf-8"), rtol, atol)
        else:
            ok, d = compare_text(fp.read_text(encoding="utf-8", errors="replace"),
                                 ep.read_text(encoding="utf-8", errors="replace"), rtol, atol)
        ok_all &= ok
        if not ok:
            details.append(f"{produced}: {d}")
    if compared == 0:
        return None, "no expected output"
    return ok_all, "; ".join(details) if not ok_all else f"{compared} output(s) match ({details[0] if details else 'files'})"


def _bless(e: dict, book: Path, out: Path):
    import shutil
    if e.get("expected"):
        dst = book / e["expected"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / f"{e['name']}.out", dst)
    for produced, expected in e.get("expected_files", {}).items():
        if (out / produced).is_file():
            dst = book / expected
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out / produced, dst)


def cmd_list(book: Path, entries: list[dict]) -> int:
    print(f"book folder: {book}")
    print(f"{'name':<34} {'section':<26} {'slow':<5} {'cache':<6} {'expected':<9} script")
    for e in entries:
        exp = "yes" if (e["expected"] or e["expected_files"]) else "-"
        print(f"{e['name']:<34} {str(e.get('section', ''))[:26]:<26} {('yes' if e['slow'] else '-'):<5} "
              f"{('yes' if e['uses_cache'] else '-'):<6} {exp:<9} {e['script']}")
    print(f"{len(entries)} entries ({sum(e['slow'] for e in entries)} slow, "
          f"{sum(e['uses_cache'] for e in entries)} use cached data)")
    return 0


def cmd_regenerate(book: Path, entries: list[dict], args) -> int:
    out = Path(args.out).resolve() if args.out else book / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    chosen, skipped = _select(entries, args.only, args.all)
    if not chosen:
        print("nothing selected", file=sys.stderr)
        return 2
    env = _env(book)
    print(f"book folder: {book}\noutput dir:  {out}\nrunning {len(chosen)} script(s)"
          + (f"; skipping {len(skipped)} slow one(s) (use --all)" if skipped else ""), flush=True)
    rows, timings, bad = [], {}, 0
    for e in chosen:
        miss = _missing_requirements(e["requires"])
        if miss:
            rows.append((e["name"], "SKIP", 0.0, "missing: " + ", ".join(miss)))
            print(f"  {e['name']:<34} SKIP (missing {', '.join(miss)})", flush=True)
            continue
        print(f"  {e['name']:<34} ...", end="", flush=True)
        r = _run_one(e, book, out, env, args.timeout)
        timings[e["name"]] = r["seconds"]
        if r["returncode"] != 0:
            status, detail = "ERROR", f"exit {r['returncode']}: " + " | ".join(r["stderr_tail"])
            bad += 1
        elif args.check:
            ok, detail = _check_one(e, book, out, args.rtol, args.atol)
            status = "NOEXP" if ok is None else ("PASS" if ok else "FAIL")
            bad += ok is False
        else:
            status, detail = "RAN", ""
        if args.bless and r["returncode"] == 0:
            _bless(e, book, out)
            detail = (detail + "; " if detail else "") + "expected outputs overwritten (--bless)"
        rows.append((e["name"], status, r["seconds"], detail))
        print(f"\r  {e['name']:<34} {status:<5} {r['seconds']:8.1f} s", flush=True)
    for e in skipped:
        rows.append((e["name"], "SKIP", 0.0, "slow (use --all)"))
    (out / "_timings.json").write_text(json.dumps(timings, indent=1), encoding="utf-8")
    print()
    print(f"{'name':<34} {'status':<6} {'seconds':>8}  detail")
    print("-" * 100)
    for name, st, sec, det in rows:
        print(f"{name:<34} {st:<6} {sec:8.1f}  {det[:160]}")
    print("-" * 100)
    counts = {s: sum(1 for r in rows if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "NOEXP", "RAN", "SKIP")}
    print("  ".join(f"{k} {v}" for k, v in counts.items() if v) + f"   total {sum(timings.values()):.1f} s")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rnel.book", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list", "regenerate"):
        p = sub.add_parser(name)
        p.add_argument("--book-dir", default=None, help="path to the book/ folder (or its parent)")
        if name == "regenerate":
            p.add_argument("--only", default=None, help="comma-separated names or glob patterns")
            p.add_argument("--out", default=None, help="output directory (default book/outputs)")
            p.add_argument("--check", action="store_true", help="compare with the stored expected outputs")
            p.add_argument("--all", action="store_true", help="include slow scripts")
            p.add_argument("--bless", action="store_true", help="overwrite expected outputs with fresh ones")
            p.add_argument("--timeout", type=float, default=None, help="per-script timeout in seconds")
            p.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
            p.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    args = ap.parse_args(argv)
    try:
        book = find_book_dir(args.book_dir)
    except FileNotFoundError as ex:
        print(ex, file=sys.stderr)
        return 2
    entries = load_manifest(book)
    if args.cmd == "list":
        return cmd_list(book, entries)
    return cmd_regenerate(book, entries, args)


if __name__ == "__main__":
    sys.exit(main())
