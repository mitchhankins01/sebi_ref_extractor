from __future__ import annotations

import json
import re
from typing import List, Optional, Dict

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None  # type: ignore
from patterns import CIRC_TRAILING_NUM, DATED_PHRASE, GAZETTE_CODE, PATTERN_SECTION_ACT
import os
import time


def extract_pages(pdf_path: str) -> List[str]:
    if fitz is None:  # pragma: no cover
        raise ImportError("PyMuPDF (fitz) is not installed. Please install requirements.txt.")
    pages: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            try:
                text = page.get_text("text")
            except Exception:
                text = page.get_text() or ""
            cleaned = preclean_text(text or "")
            pages.append(cleaned)
    return pages


def get_context(text: str, start: int, end: int, win: int = 120) -> str:
    a = max(0, start - win)
    b = min(len(text), end + win)
    snippet = text[a:b].strip()
    if len(snippet) > 240:
        snippet = snippet[:240]
    return snippet


def safe_json_parse(raw_text: Optional[str]) -> Optional[dict]:
    if not raw_text:
        return None
    s = raw_text.strip()
    # Try to find the first JSON object in the text
    # Remove code fences if present
    s = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.IGNORECASE | re.MULTILINE)
    # Find first { ... } block
    first_brace = s.find("{")
    last_brace = s.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        try:
            return json.loads(s)
        except Exception:
            return None
    candidate = s[first_brace : last_brace + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def find_title_from_first_page(text: str) -> str:
    if not text:
        return ""
    # 1) Subject pattern
    m = re.search(r"(?im)^\s*(?:Sub(?:ject)?|Subject)\s*:\s*(.+)$", text)
    if m:
        return m.group(1).strip()

    # 2) Line after a CIRCULAR header
    # Look for a header like 'CIRCULAR' and capture the next non-empty line
    lines = [ln.strip() for ln in text.splitlines()]
    for i, ln in enumerate(lines):
        if re.match(r"(?i)^\s*CIRCULAR\s*$", ln):
            # find next non-empty line
            for j in range(i + 1, min(i + 6, len(lines))):
                nxt = lines[j].strip()
                if nxt:
                    return nxt
            break

    # 3) Heuristic: first meaningful line that looks like a subject
    for ln in lines[:15]:
        if len(ln) > 8 and ln.endswith(":"):
            # next line could be subject
            continue
        # contains common keywords
        if re.search(r"\bsubject\b|\bsub:\b", ln, flags=re.IGNORECASE):
            # may already match above but keep heuristic
            return re.sub(r"(?i)^.*subject\s*:\s*", "", ln).strip()

    return ""


def preclean_text(text: str) -> str:
    # Join hard line breaks inside words or series that likely belong together
    # Collapse multiple spaces and normalize hyphenated line breaks
    if not text:
        return ""
    s = text
    # Replace soft hyphen line breaks like: ABC-
    s = re.sub(r"-\n\s*", "", s)
    # Replace newline followed by lowercase/uppercase continuation with space
    s = re.sub(r"\n(?=[a-z0-9])", " ", s)
    s = re.sub(r"\n(?=[A-Z])", " ", s)
    # Normalize spaces
    s = re.sub(r"[ \t]+", " ", s)
    # Remove lingering spaces before punctuation
    s = re.sub(r"\s+([,.);:])", r"\1", s)
    return s


def normalize_circular_id(raw: str) -> str:
    m = CIRC_TRAILING_NUM.search(raw)
    if not m:
        return raw
    prefix = m.group("prefix")
    num = m.group("num")
    suffix = m.group("suffix") or ""
    stripped = str(int(num)) if num.lstrip("0") != "" else "0"
    return f"{prefix}{stripped}{suffix}"


_MONTH_TO_NUM = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}


def normalize_date_mdy(raw: str) -> Optional[str]:
    m = DATED_PHRASE.search(raw)
    if not m:
        return None
    try:
        # Extract the whole date portion after 'dated '
        date_str = m.group(0)
        # e.g., 'dated July 5, 2021'
        parts = re.search(r"dated\s+(\w+)\s+(\d{1,2}),\s+(\d{4})", date_str, flags=re.IGNORECASE)
        if not parts:
            return None
        month_name = parts.group(1).lower()
        day = int(parts.group(2))
        year = int(parts.group(3))
        month = _MONTH_TO_NUM.get(month_name)
        if not month:
            return None
        return f"{year:04d}-{month}-{day:02d}"
    except Exception:
        return None


def classify_scope(ref_type: str) -> str:
    internal_set = {"annexure", "chapter", "clause", "paragraph", "para"}
    return "internal" if ref_type.lower() in internal_set else "external"


def is_gazette_code(s: str) -> bool:
    return bool(GAZETTE_CODE.search(s))


# ---- Title helpers (optional, best-effort) ----

_MONTH_TO_NUM = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}

_MONTH_ABBR = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "sept": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


def normalize_date_mdy_loose(raw: str) -> Optional[str]:
    """Parse 'Month DD, YYYY' or 'Mon DD, YYYY' and return ISO date."""
    if not raw:
        return None
    m = re.search(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\b", raw)
    if not m:
        return None
    mon = m.group(1).lower()
    day = int(m.group(2))
    year = int(m.group(3))
    mm = _MONTH_TO_NUM.get(mon) or _MONTH_ABBR.get(mon[:3])
    if not mm:
        return None
    return f"{year:04d}-{mm}-{day:02d}"


def build_annexure_title_map_by_date(page_texts: List[str]) -> Dict[str, str]:
    """Map ISO date -> title from Annexure 'List of Circulars' blocks where
    lines look like 'Jan 10, 2022 <Title text>' possibly without IDs.
    """
    header_re = re.compile(r"(?im)^\s*Annexure\s+[A-Z].*List of Circulars.*$")
    next_annex_re = re.compile(r"(?im)^\s*Annexure\s+[A-Z]")
    date_title_re = re.compile(r"\b([A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4})\b\s+(.+)")
    mapping: Dict[str, str] = {}
    for text in page_texts:
        if not text or not header_re.search(text):
            continue
        start = header_re.search(text).end()
        tail = text[start:]
        mnext = next_annex_re.search(tail)
        if mnext:
            tail = tail[: mnext.start()]
        lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
        for ln in lines:
            dm = date_title_re.search(ln)
            if not dm:
                continue
            iso = normalize_date_mdy_loose(dm.group(1))
            title = dm.group(2).strip()
            if iso and title:
                mapping[iso] = title
    return mapping


def is_valid_llm_reference(ref_text: str, ref_type: str, snippet: str) -> bool:
    if not ref_text:
        return False
    text_norm = ref_text.strip().lower()
    # Drop placeholders
    if text_norm == "sebi circular no.":
        return False
    if text_norm.endswith("circular no."):
        return False
    if text_norm.startswith("sebi circular") and "cir/" not in text_norm and not re.search(r"\d{4}", text_norm):
        return False

    # Type-specific validation
    if ref_type == "circular":
        can = normalize_circular_id(ref_text)
        # Keep only if canonicalization detected numeric suffix change or matched pattern
        if can == ref_text:
            # still valid if it already matched a full id with number at end
            return bool(CIRC_TRAILING_NUM.search(ref_text))
        return True

    if ref_type == "gazette":
        return is_gazette_code(ref_text)

    if ref_type == "url":
        return ref_text.startswith("http://") or ref_text.startswith("https://")

    if ref_type in ("section", "regulation"):
        # require explicit Act/Rules/Regulations context
        return bool(PATTERN_SECTION_ACT.search(ref_text) or PATTERN_SECTION_ACT.search(snippet))

    # Other types are not accepted from LLM for now to prevent hallucinations
    return False


def build_annexure_title_map(page_texts: List[str]) -> dict:
    """Cheap heuristic: from Annexure pages listing circulars, map id -> title fragment.
    Looks for lines like: SEBI/.../CIR/... - Title or : Title
    """
    mapping: dict = {}
    id_re = re.compile(r"((?:SEBI/)?(?:HO/)?[A-Z][A-Z0-9_/-]+/CIR(?:/P)?/(?:\d{4}/)?\d+(?:/\d+)?)", re.IGNORECASE)
    for text in page_texts:
        if not text:
            continue
        if re.search(r"Annexure", text, flags=re.IGNORECASE) and re.search(
            r"List of Circulars", text, flags=re.IGNORECASE
        ):
            for line in text.splitlines():
                m = id_re.search(line)
                if not m:
                    continue
                cid = normalize_circular_id(m.group(1))
                # title after '-' or ':' on same line
                tail = re.split(r"\s[-:]\s", line, maxsplit=1)
                title_fragment = None
                if len(tail) == 2:
                    title_fragment = tail[1].strip()
                if title_fragment:
                    mapping[cid] = title_fragment
    return mapping


def guess_cited_title(canonical_id: str, contexts: list[str]) -> dict | None:
    """
    Returns {"cited_title": str, "cited_date": "YYYY-MM-DD" or None, "confidence": float} or None.
    Uses a strict JSON-only prompt. If unsure, returns None.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        return None

    prompt = (
        "Given this SEBI circular ID and surrounding text mentions, extract the official title (subject) only if it is explicitly present in the snippets.\n"
        "Do not use any external sources. Output JSON only as: {\"cited_title\": str|null, \"cited_date\": \"YYYY-MM-DD\"|null, \"confidence\": 0-1}.\n"
        "If unsure or not explicitly present, return {\"cited_title\": null, \"cited_date\": null, \"confidence\": 0}.\n"
        f"Circular ID: {canonical_id}\n"
        f"Snippets:\n- " + "\n- ".join(contexts[:5])
    )
    try:
        resp = model.generate_content(prompt)
        raw_text = getattr(resp, "text", None)
        from .utils import safe_json_parse  # type: ignore  # avoid circular; patched below if needed
    except Exception:
        raw_text = None
        safe_json_parse = None  # type: ignore
    if not raw_text:
        return None
    try:
        # local import if direct import failed
        from utils import safe_json_parse as _safe
        data = _safe(raw_text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    cited_title = data.get("cited_title")
    cited_date = data.get("cited_date")
    confidence = data.get("confidence")
    if cited_title is None:
        return None
    # Basic validation
    if cited_date is not None and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(cited_date)):
        cited_date = None
    try:
        conf_f = float(confidence)
    except Exception:
        conf_f = 0.0
    return {"cited_title": cited_title, "cited_date": cited_date, "confidence": conf_f, "_raw": raw_text}


def reflow_preserve_paragraphs(text: str) -> str:
    if not text:
        return ""
    s = text.replace("\r\n", "\n")
    # Protect double newlines
    s = s.replace("\n\n", "<PARA_BREAK>")
    # Collapse single newlines to spaces
    s = re.sub(r"\n+", " ", s)
    # Restore para breaks
    s = s.replace("<PARA_BREAK>", "\n\n")
    # Normalize spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_annexure_title_map_v2(page_texts: List[str]) -> dict:
    """Scan annexure 'List of Circulars' blocks and extract id -> title/date.
    We detect headers like 'Annexure X ... List of Circulars' and collect until next blank line or next Annexure header.
    """
    header_re = re.compile(r"(?im)^\s*Annexure\s+[A-Z].*List of Circulars.*$")
    next_annex_re = re.compile(r"(?im)^\s*Annexure\s+[A-Z]")
    line_re = re.compile(r"(SEBI/[^\s,;]+)\s*[-–:]\s*(.+?)(?:\s*[,;]\s*dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4}))?$", re.IGNORECASE)
    mapping: dict = {}
    for text in page_texts:
        if not text:
            continue
        if not header_re.search(text):
            continue
        # From first header, take text until next Annexure header or blank line sequence
        start = header_re.search(text).end()
        tail = text[start:]
        # Stop at next Annexure header if present
        mnext = next_annex_re.search(tail)
        if mnext:
            tail = tail[: mnext.start()]
        # Split into lines and reflow wrapped lines by joining
        # Remove excessive blank lines
        lines = [ln.strip() for ln in tail.splitlines()]
        # Build buffered line groups separated by empty lines
        buf = []
        for ln in lines:
            if ln == "":
                buf.append("\n")
            else:
                buf.append(ln)
        text_block = " ".join([x for x in buf if x != "\n"]).strip()
        # Now try to extract multiple entries
        for m in line_re.finditer(text_block):
            raw_id = m.group(1)
            title = (m.group(2) or "").strip()
            date_raw = m.group(3)
            cid = normalize_circular_id(raw_id)
            mapping[cid] = title
    return mapping


