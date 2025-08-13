from __future__ import annotations

import re
from typing import List

# Compile patterns once

# Circulars (modern + legacy)
PATTERN_CIRCULAR_1 = re.compile(
    r"(?:SEBI/)?(?:HO/)?[A-Z][A-Z0-9_/-]+/CIR(?:/P)?/(?:\d{4}/)?\d+(?:/\d+)?",
    re.IGNORECASE,
)
PATTERN_CIRCULAR_2 = re.compile(
    r"(?:CIR|MRD|MIRSD|IMD)[/_-][A-Z0-9/._-]*\d{1,2}/\d{4}",
    re.IGNORECASE,
)
PATTERN_GAZETTE = re.compile(
    r"(?:No\.?\s*)?(?:SEBI/)?LAD-?NRO/GN/[0-9-]+(?:/\d+)?",
    re.IGNORECASE,
)

# Regulations/sections/acts
PATTERN_REGULATION = re.compile(r"[Rr]egulation\s+\d+(?:\s*\([^)]+\))?")
PATTERN_SECTION_ACT = re.compile(
    r"[Ss]ection\s+\d+(?:\s*\([^)]+\))?\s+of\s+[^,\n]+(?:Act|Rules|Regulations),?\s+\d{4}"
)

# Internal refs
PATTERN_ANNEXURE = re.compile(r"(?:Annexure|Appendix)\s+(?:[A-Z]|\d+|[IVX]+)")
PATTERN_CHAPTER_PARA = re.compile(r"(?:Chapter|Para(?:graph)?)\s+\d+(?:\.\d+)*")

# Notifications/urls
PATTERN_NOTIFICATION = re.compile(
    r"(?:[Gg]azette\s+)?[Nn]otification\s+(?:No\.?\s*)?[\w/().-]+"
)
PATTERN_URL = re.compile(r"https?://[^\s)>,]+")
PATTERN_PRESS_RELEASE = re.compile(r"\bPR\s*(?:No\.? )?\s*:?\s*(\d+/\d{4})\b", re.IGNORECASE)

ALL_COMPILED_PATTERNS: List[re.Pattern[str]] = [
    PATTERN_CIRCULAR_1,
    PATTERN_CIRCULAR_2,
    PATTERN_GAZETTE,
    # Gazette S.O./G.S.R. numbering like "S.O. 5401 (E)"
    re.compile(r"(?:S\.?O\.?|G\.?S\.?R\.?)\s*\d+\s*\([A-Za-z]\)", re.IGNORECASE),
    PATTERN_REGULATION,
    PATTERN_SECTION_ACT,
    PATTERN_ANNEXURE,
    PATTERN_CHAPTER_PARA,
    PATTERN_NOTIFICATION,
    PATTERN_URL,
    PATTERN_PRESS_RELEASE,
]


def guess_type(text: str) -> str:
    t = text.strip()
    low = t.lower()

    # Press Release identifiers
    if PATTERN_PRESS_RELEASE.search(t) or low.startswith("pr no") or low.startswith("pr "):
        return "press_release"

    # Gazette codes like S.O. 5401(E) or G.S.R. 123(E)
    if GAZETTE_CODE.search(t):
        return "gazette"

    # URL
    if low.startswith("http://") or low.startswith("https://"):
        return "url"

    # Circular-like markers
    if "cir" in low or "mirsd" in low or "imd" in low or "mrd" in low:
        # Heuristic: many circular ids contain CIR
        if re.search(r"\bCIR\b", t):
            return "circular"
        # Gazette numbers often tagged LAD-NRO/GN
    if "lad-nro" in low or "/gn/" in low:
        return "gazette"

    # Regulation mentions
    if re.search(r"\b[Rr]egulation\b", t):
        return "regulation"

    # Sections and Acts
    if re.search(r"\b[Ss]ection\b", t) and re.search(r"\b(Act|Rules|Regulations)\b", t):
        return "section"
    if re.search(r"\bAct\b", t) and re.search(r"\b\d{4}\b", t):
        return "act"

    # Notifications and Gazette
    if re.search(r"\b[Gg]azette\b", t):
        return "gazette"
    if re.search(r"\b[Nn]otification\b", t):
        return "notification"

    # Internal refs
    if re.search(r"\bAnnexure\b|\bAppendix\b", t):
        return "annexure"
    if re.search(r"\bChapter\b|\bPara(?:graph)?\b", t):
        return "chapter"

    # Legacy circular patterns
    if re.search(r"\b(SEBI/)?(?:HO/)?[A-Z][A-Z0-9_/-]+/CIR", t):
        return "circular"

    return "other"


# Canonicalization and detection helpers
CIRC_TRAILING_NUM = re.compile(r"(?P<prefix>.*/)(?P<num>\d+)(?P<suffix>(?:/\d+)?$)", re.IGNORECASE)
GAZETTE_CODE = re.compile(r"\b(?:S\.O\.|G\.S\.R\.)\s*\d+\s*\(?[A-Za-z]?\)?", re.IGNORECASE)
DATED_PHRASE = re.compile(
    r"\bdated\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)


