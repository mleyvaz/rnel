"""The reconciled plan of the CRC edition: five Parts and 16 chapters (PROPOSAL_STRUCTURE_v13.1.md), and the maps
from the 19-chapter plan of crc_assets (CHAPTER_PLAN.md of 3 October, backed up in crc_assets/_plan19_backup/) to it.

Imported by renumber_assets.py, build_crc.py and check_crc.py, so that every file uses one table.
"""
import re

PARTS = [
    ("I", "Foundations and the correspondence problem", [1, 2]),
    ("II", "Reduction, retraction and lifted representations", [3, 4, 5, 6, 7]),
    ("III", "Dependence, plithogeny and off-values", [8, 9, 10, 11]),
    ("IV", "Statistics, evidence fusion, computation and applications", [12, 13, 14]),
    ("V", "Probability logics, questions from the literature and the research agenda", [15, 16]),
]

TITLES = {
    1: "Introduction: what a credal set can see of a neutrosophic triple",
    2: "Foundations: imprecise, neutrosophic and plithogenic probability",
    3: "The normalised subclass: reduction, hierarchy and singletons",
    4: "The classical frame: sure loss, uniqueness and collapses",
    5: "Retraction to Subjective Logic",
    6: "The glut lifting and N-norms as natural extensions",
    7: "Models of imprecise probability and what event bounds can see",
    8: "Indeterminacy profiles, conservative copulas and open problems",
    9: "Choosing an N-norm in practice",
    10: "Plithogenic probability and credal sets, with applications",
    11: "Off-values as signed credal sets",
    12: "RNEL and the fused credal interval in data",
    13: "Neutrosophic statistics with refined indeterminacy",
    14: "A running example and computing with rnel",
    15: "Complete probability logics and questions raised in the literature",
    16: "Discussion and research agenda",
}

# sections of the book (names kept) in each chapter, in reading order
SECTIONS = {
    1: ["1", "1.1"], 2: ["F.1-F.9", "2.1-2.9"], 3: ["3", "3.1", "4", "4.1"], 4: ["5.1-5.3"], 5: ["6"],
    6: ["7.1-7.3"], 7: ["12.1-12.3"], 8: ["8.1", "8.2", "8.2.1"], 9: ["8.3"], 10: ["10.1-10.5", "11.1-11.5"],
    11: ["8.4"], 12: ["12.4", "9", "9.1"], 13: ["12.7"], 14: ["11.6", "Appendix C"], 15: ["12.6", "12.5", "12.8"],
    16: ["13.1-13.7"],
}


def part_of(ch):
    for num, title, chs in PARTS:
        if ch in chs:
            return num, title
    raise KeyError(ch)


# ---------------------------------------------------------------- 19-chapter plan -> 16-chapter plan
CHMAP = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 5, 8: 6, 9: 8, 10: 9, 11: 11, 12: 10, 13: 10, 14: 7, 15: 12,
         16: 13, 17: 15, 18: 15, 19: 16}            # old chapter 13 = Sections 11.1-11.6: 11.6 goes to Chapter 14
# figures are numbered by order of first appearance within each chapter
FIGMAP = {"1.1": "1.2", "1.2": "1.1", "2.1": "2.1", "3.1": "2.2", "4.1": "3.1", "4.2": "3.2", "6.1": "4.1",
          "8.1": "6.1", "8.2": "6.2", "8.3": "6.3", "14.1": "7.1", "10.1": "9.2", "10.2": "9.1", "12.1": "10.1",
          "11.1": "11.1", "15.1": "12.1", "15.2": "12.2", "16.1": "13.2", "16.2": "13.1", "16.3": "13.3",
          "16.4": "13.4", "16.5": "13.5", "16.6": "13.6", "13.1": "14.1", "17.1": "15.1"}
_EXBLOCKS = [  # (old chapter, first k, last k, new chapter, offset)
    (1, 1, 7, 1, 0), (2, 1, 9, 2, 0), (3, 1, 7, 2, 9), (4, 1, 7, 3, 0), (5, 1, 6, 3, 7), (6, 1, 7, 4, 0),
    (7, 1, 6, 5, 0), (8, 1, 7, 6, 0), (14, 1, 6, 7, 0), (9, 1, 7, 8, 0), (10, 1, 8, 9, 0), (12, 1, 7, 10, 0),
    (13, 1, 5, 10, 7), (11, 1, 8, 11, 0), (15, 1, 7, 12, 0), (16, 1, 8, 13, 0), (13, 6, 7, 14, -5),
    (17, 1, 7, 15, 0), (18, 1, 6, 15, 7), (19, 1, 6, 16, 0)]
EXMAP = {}
for oc, a, b, nc, off in _EXBLOCKS:
    for k in range(a, b + 1):
        EXMAP["%d.%d" % (oc, k)] = "%d.%d" % (nc, k + off)
assert len(EXMAP) == 133 and len(set(EXMAP.values())) == 133
NEW_EXERCISES = ["14.3", "14.4", "14.5", "14.6"]      # written for Chapter 14 (computing with rnel)


def chmap_section_aware(ch, context=""):
    """Old chapter 13 holds Sections 11.1-11.6; 11.6 (the running example) goes to Chapter 14, the rest to 10."""
    if ch == 13 and re.search(r"\b11\.6", context):
        return 14
    return CHMAP[ch]


_REF = re.compile(r"\b(Figures?|Exercises?|Chapters?|Solutions?)(\s+)((?:\d+(?:\.\d+)?(?:\([a-z]\))?(?:(?:,\s*|\s+and\s+|\s*[–-]\s*|\s+to\s+))?)+)")
_NUM = re.compile(r"\d+(?:\.\d+)?")


def remap_refs(text):
    """Rewrite 'Figure 13.1', 'Exercises 4.1-4.3', 'Chapters 15 and 16' etc. from the 19- to the 16-chapter plan."""
    def sub(m):
        kind = m.group(1).lower()

        def one(n):
            s = n.group(0)
            if kind.startswith("fig"):
                return FIGMAP.get(s, s)
            if kind.startswith("ex") or kind.startswith("sol"):
                return EXMAP.get(s, s)
            if "." in s:
                return s
            return str(CHMAP.get(int(s), int(s)))
        return m.group(1) + m.group(2) + _NUM.sub(one, m.group(3))
    return _REF.sub(sub, text)
