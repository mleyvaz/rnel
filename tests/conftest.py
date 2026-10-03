"""Shared pytest configuration for rnel.

Book markers
------------
A test that checks a numbered result of the book *Neutrosophic Evidence* (Smarandache and Leyva-Vázquez) carries

    @pytest.mark.theorem("Theorem 10(b)")                 # one or more labels
    @pytest.mark.theorem("Corollary 5", "Theorem 10(c)")

``tools/theorem_map.py`` collects these markers, runs the tests and writes the table
book result -> test(s) -> status (THEOREM_TEST_MAP.md). When the environment variable RNEL_THEOREM_JSON is set,
this conftest records the outcome of every marked test into that JSON file at the end of the session.
"""
import json
import os

import pytest

_RESULTS = {}


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "theorem(*labels): the test checks the named numbered result(s) of the book (e.g. 'Theorem 10(b)')",
    )
    config.addinivalue_line("markers", "slow: long-running test")


def _labels(item):
    out = []
    for m in item.iter_markers(name="theorem"):
        out.extend(str(a) for a in m.args)
    return out


def pytest_collection_modifyitems(config, items):
    for item in items:
        labels = _labels(item)
        if labels:
            _RESULTS[item.nodeid] = {"labels": labels, "outcome": "not run"}


def pytest_runtest_logreport(report):
    rec = _RESULTS.get(report.nodeid)
    if rec is None:
        return
    if report.when == "call":
        rec["outcome"] = report.outcome  # passed / failed / skipped
    elif report.when == "setup" and report.outcome != "passed":
        rec["outcome"] = "skipped" if report.skipped else "error"


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get("RNEL_THEOREM_JSON")
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_RESULTS, fh, indent=1, sort_keys=True)
