"""Run theorem-labelled tests and write the book result -> automated test map."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("book/THEOREM_TEST_MAP.md"))
    ap.add_argument("--results", type=Path, help="reuse an RNEL_THEOREM_JSON file instead of running pytest")
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    temporary = args.results is None
    if args.results is None:
        fd, raw_path = tempfile.mkstemp(prefix="rnel-theorems-", suffix=".json")
        os.close(fd)
        result_path = Path(raw_path)
    else:
        result_path = args.results
    try:
        if temporary:
            env = dict(os.environ)
            env["RNEL_THEOREM_JSON"] = str(result_path)
            env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
            cp = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "theorem"], cwd=root, env=env)
            if cp.returncode:
                return cp.returncode
        records = json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        if temporary:
            result_path.unlink(missing_ok=True)

    by_label = defaultdict(list)
    for nodeid, rec in sorted(records.items()):
        for label in rec["labels"]:
            by_label[label].append((nodeid, rec["outcome"]))
    passed_cases = sum(rec["outcome"] == "passed" for rec in records.values())
    failed_cases = sum(rec["outcome"] in {"failed", "error"} for rec in records.values())
    skipped_cases = sum(rec["outcome"] == "skipped" for rec in records.values())
    lines = [
        "# RNEL 0.3.0 — theorem-to-test map", "",
        "Generated from `@pytest.mark.theorem(...)` markers; statuses come from an executed pytest session.", "",
        f"- Numbered results/labels: **{len(by_label)}**",
        f"- Marked test cases: **{len(records)}** ({passed_cases} passed, {failed_cases} failed, {skipped_cases} skipped)", "",
        "| Book result | Automated test(s) | Status |", "|---|---|---|",
    ]
    for label in sorted(by_label, key=str.casefold):
        tests = by_label[label]
        statuses = {s for _, s in tests}
        status = "PASS" if statuses == {"passed"} else ", ".join(sorted(statuses)).upper()
        short = [n.replace("tests/", "", 1).replace("tests\\", "", 1) for n, _ in tests]
        safe_label = label.replace("|", "\\|")
        lines.append(f"| {safe_label} | " + "<br>".join(f"`{n}`" for n in short) + f" | {status} |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {len(by_label)} labels, {len(records)} cases, {failed_cases} failed, {skipped_cases} skipped")
    return 1 if failed_cases else 0


if __name__ == "__main__":
    raise SystemExit(main())
