#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from models import ExtractionResult, ReferenceHit, SourceDoc
from patterns import (
    ALL_COMPILED_PATTERNS,
    DATED_PHRASE,
    GAZETTE_CODE,
    guess_type,
)
from utils import (
    extract_pages,
    find_title_from_first_page,
    get_context,
    safe_json_parse,
    normalize_circular_id,
    normalize_date_mdy,
    classify_scope,
    build_annexure_title_map_v2,
    guess_cited_title,
    reflow_preserve_paragraphs,
    is_valid_llm_reference,
    build_annexure_title_map_by_date,
)

__version__ = "0.1.0"
_LLM_NOTICE_EMITTED = False


def _init_gemini_if_available() -> Optional[Any]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        # Use a fast, inexpensive model for extraction
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model
    except Exception:
        return None


def _split_sentences(text: str) -> List[str]:
    # Simple sentence splitter on ., ?, ! while keeping basic context
    # Avoid splitting on common abbreviations crudely by limiting split to space after punctuation
    parts = re.split(r"(?<=[.?!])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _has_ambiguous_reference_phrase(sentence: str) -> bool:
    candidates = [
        "vide circular",
        "refer circular",
        "in partial modification",
        "partial modification",
        "as per regulation",
        "pursuant to regulation",
        "as per sebi circular",
        "in terms of regulation",
        "as per the provisions of",
        "as provided under regulation",
        "as provided under the regulation",
    ]
    s = sentence.lower()
    return any(kw in s for kw in candidates)


def _run_regex_on_page(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    results: List[Tuple[str, Tuple[int, int]]] = []
    for pattern in ALL_COMPILED_PATTERNS:
        for m in pattern.finditer(text):
            results.append((m.group(0), m.span()))
    return results


def _maybe_llm_fallback_for_page(
    model: Any, page_text: str
) -> List[Dict[str, Any]]:
    if model is None:
        return []

    sentences = _split_sentences(page_text)
    llm_hits: List[Dict[str, Any]] = []
    for sent in sentences:
        if not _has_ambiguous_reference_phrase(sent):
            continue
        # If regex already finds something inside this sentence, skip llm for it.
        regex_found_here = False
        for pattern in ALL_COMPILED_PATTERNS:
            if pattern.search(sent):
                regex_found_here = True
                break
        if regex_found_here:
            continue

        prompt = (
            "Find regulatory references (circulars, regulations, acts, notifications, gazette, URLs) "
            "explicitly mentioned in this passage. Return JSON: {\"references\":[{\"reference_text\":\"\",\"reference_type\":\"\",\"confidence\":0-1}]}. "
            "Only include references literally present in the passage; do not infer or hallucinate.\n"
            f"Passage:\n{sent}"
        )
        try:
            resp = model.generate_content(prompt)
            raw_text = getattr(resp, "text", None)
            data = safe_json_parse(raw_text) if raw_text else None
            if not data:
                continue
            refs = data.get("references") if isinstance(data, dict) else None
            if isinstance(refs, list):
                for r in refs:
                    if not isinstance(r, dict):
                        continue
                    ref_text = (r.get("reference_text") or "").strip()
                    ref_type = (r.get("reference_type") or "").strip().lower()
                    confidence = r.get("confidence")
                    try:
                        conf_val = float(confidence)
                    except Exception:
                        conf_val = 0.0
                    if not ref_text:
                        continue
                    # Basic sanity for type; if missing try to guess
                    if ref_type not in {
                        "circular",
                        "regulation",
                        "section",
                        "act",
                        "notification",
                        "gazette",
                        "url",
                        "annexure",
                        "chapter",
                        "other",
                    }:
                        ref_type = guess_type(ref_text)
                    llm_hits.append(
                        {
                            "reference_text": ref_text,
                            "reference_type": ref_type,
                            "confidence": conf_val,
                            "context_sentence": sent,
                        }
                    )
        except Exception:
            # Never crash on Gemini issues
            continue

    return llm_hits


def _dedup_key(reference_type: str, text: str, page: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower().rstrip(".,;:")
    return f"{page}|{reference_type}|{normalized}"


def _dedup_key_with_canonical(reference_type: str, text: str, page: int, canonical_id: Optional[str]) -> str:
    if canonical_id:
        return f"{page}|{reference_type}|{canonical_id.strip().lower()}"
    return _dedup_key(reference_type, text, page)


def _extract_nearby_date_iso(text: str, start: int, end: int) -> Optional[str]:
    # Prefer date immediately after the match; fallback to just before
    after = text[end : min(len(text), end + 180)]
    date_iso = normalize_date_mdy(after)
    if date_iso:
        return date_iso
    before = text[max(0, start - 180) : start]
    return normalize_date_mdy(before)


def _normalize_url(text: str) -> str:
    if text.startswith("http://") or text.startswith("https://"):
        return text.rstrip(".,);>]")
    return text


def _drop_substring_duplicates(hits: List[ReferenceHit]) -> List[ReferenceHit]:
    # Remove substring duplicates on the same page for same type, preferring longer text
    grouped: Dict[Tuple[int, str], List[ReferenceHit]] = {}
    for h in hits:
        grouped.setdefault((h.page_number, h.reference_type), []).append(h)
    deduped: List[ReferenceHit] = []
    for (page, rtype), group in grouped.items():
        group_sorted = sorted(group, key=lambda x: len(x.reference_text), reverse=True)
        kept: List[ReferenceHit] = []
        for cand in group_sorted:
            if any(cand.reference_text in k.reference_text for k in kept):
                continue
            kept.append(cand)
        deduped.extend(kept)
    return deduped


def _find_span_center_in_page(page_text: str, hit: ReferenceHit) -> Optional[int]:
    # Try canonical id first
    if hit.canonical_id:
        pos = page_text.find(hit.canonical_id)
        if pos != -1:
            return pos + len(hit.canonical_id) // 2
    # Try full reference text
    if hit.reference_text:
        pos = page_text.find(hit.reference_text)
        if pos != -1:
            return pos + len(hit.reference_text) // 2
    # Try a chunk from the snippet
    snippet = hit.context_snippet or ""
    probe = snippet[:80]
    if probe:
        pos = page_text.find(probe)
        if pos != -1:
            return pos + len(probe) // 2
    return None


def _filter_dated_only_gazettes(hits: List[ReferenceHit], pages: List[str]) -> List[ReferenceHit]:
    by_page: Dict[int, List[ReferenceHit]] = {}
    for h in hits:
        by_page.setdefault(h.page_number, []).append(h)
    filtered: List[ReferenceHit] = []
    for page_num, page_hits in by_page.items():
        page_text = pages[page_num - 1] if 0 <= page_num - 1 < len(pages) else ""
        merged_codes = [
            h for h in page_hits if h.reference_type == "gazette" and (h.canonical_id or "").strip() != ""
        ]
        dated_only = [
            h
            for h in page_hits
            if h.reference_type == "gazette"
            and not h.canonical_id
            and ("Gazette notification" in h.reference_text or True)
        ]
        merged_centers = [
            _find_span_center_in_page(page_text, h) for h in merged_codes
        ]
        merged_centers = [c for c in merged_centers if c is not None]
        for h in page_hits:
            if h in dated_only and merged_centers:
                c = _find_span_center_in_page(page_text, h)
                if c is not None and any(abs(c - mc) <= 250 for mc in merged_centers):
                    continue
            filtered.append(h)
    return filtered


def _emit_validation_warnings(hits: List[ReferenceHit]) -> None:
    circ_suffix_re = re.compile(r"/([0-9]+)(?:/\d+)?$")
    iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for h in hits:
        if h.reference_type == "circular" and h.canonical_id:
            m = circ_suffix_re.search(h.canonical_id)
            if not m:
                print(f"Warning: circular canonical_id missing numeric suffix: {h.canonical_id}")
            else:
                if m.group(1) == "0":
                    print(f"Warning: circular canonical_id ends with /0: {h.canonical_id}")
        if h.date and not iso_re.match(h.date):
            print(f"Warning: non-ISO date detected: {h.date} in '{h.reference_text[:80]}…'")


def run_extraction(
    pdf_path: str,
    out_json: str,
    out_csv: str,
    *,
    external_only: bool = False,
    summary: bool = False,
    md_path: Optional[str] = None,
    guess_titles: bool = False,
) -> ExtractionResult:
    pages = extract_pages(pdf_path)
    total_pages = len(pages)
    file_name = os.path.basename(pdf_path)

    # Determine title (regex, then LLM fallback once)
    first_page_text = pages[0] if pages else ""
    title = find_title_from_first_page(first_page_text)

    model = _init_gemini_if_available()
    global _LLM_NOTICE_EMITTED
    if model is None and (os.getenv("GEMINI_API_KEY", "").strip() == "") and not _LLM_NOTICE_EMITTED:
        print("LLM fallback: disabled (no GEMINI_API_KEY)")
        _LLM_NOTICE_EMITTED = True
    if (not title or title.strip().lower() in {"", "unknown"}) and model is not None and first_page_text:
        try:
            prompt = (
                "Extract the subject/title of this SEBI circular from this first-page text. "
                "Return JSON: {\"title\": \"\"}. If unsure, return {\"title\": \"Unknown\"}. Text:\n"
                f"{first_page_text}"
            )
            resp = model.generate_content(prompt)
            raw_text = getattr(resp, "text", None)
            data = safe_json_parse(raw_text) if raw_text else None
            if isinstance(data, dict):
                maybe = (data.get("title") or "").strip()
                if maybe:
                    title = maybe
        except Exception:
            pass

    if not title:
        title = "Unknown"

    source_doc = SourceDoc(title=title, pages=total_pages, file_name=file_name)

    seen: Dict[str, bool] = {}
    hits: List[ReferenceHit] = []

    for page_index, text in enumerate(pages):
        page_number = page_index + 1

        # Regex extraction
        regex_matches = _run_regex_on_page(text)
        # Gazette merge preparation per page
        gazette_candidates: List[Tuple[str, Tuple[int, int]]] = []
        other_matches: List[Tuple[str, Tuple[int, int]]] = []
        for mt, span in regex_matches:
            if GAZETTE_CODE.search(mt) or "Gazette notification" in mt:
                gazette_candidates.append((mt, span))
            else:
                other_matches.append((mt, span))

        # Process non-gazette matches first
        for match_text, (start, end) in other_matches:
            match_text = _normalize_url(match_text)
            rtype = guess_type(match_text)
            snippet = get_context(text, start, end, win=120)
            canonical_id = None
            date_iso = None
            issuing_body = "SEBI"
            # Circular canonicalization
            if rtype == "circular":
                canonical_id = normalize_circular_id(match_text)
            # Date normalization if present
            if DATED_PHRASE.search(snippet) or DATED_PHRASE.search(match_text):
                date_iso = _extract_nearby_date_iso(text, start, end) or normalize_date_mdy(match_text)
            scope = classify_scope(rtype)
            # Dedupe on canonical form when available
            dkey = _dedup_key_with_canonical(rtype, match_text, page_number, canonical_id)
            if dkey in seen:
                continue
            seen[dkey] = True
            hits.append(
                ReferenceHit(
                    reference_text=match_text,
                    reference_type=rtype,
                    page_number=page_number,
                    context_snippet=snippet,
                    method="regex",
                    canonical_id=canonical_id,
                    issuing_body=issuing_body,
                    date=date_iso,
                    scope=scope,
                )
            )

        # Merge gazette hits: look for code + dated phrase near each other
        # Build a simplified list from gazette_candidates
        for g_text, (g_start, g_end) in gazette_candidates:
            # Try to find a code in this or nearby text region
            context = get_context(text, g_start, g_end, win=150)
            code_match = GAZETTE_CODE.search(g_text) or GAZETTE_CODE.search(context)
            if not code_match:
                # If no code, treat as generic gazette notification
                rtype = "gazette"
                key = _dedup_key(rtype, g_text, page_number)
                if key in seen:
                    continue
                seen[key] = True
                date_iso = normalize_date_mdy(context) or normalize_date_mdy(g_text)
                scope = classify_scope(rtype)
                hits.append(
                    ReferenceHit(
                        reference_text=g_text,
                        reference_type=rtype,
                        page_number=page_number,
                        context_snippet=context,
                        method="regex",
                        canonical_id=None,
                        issuing_body="Government of India",
                        date=date_iso,
                        scope=scope,
                    )
                )
                continue

            code = code_match.group(0).replace(" ", "")  # compact S.O.5401(E)
            # Build a merged raw string
            date_iso = normalize_date_mdy(context)
            merged_raw = code
            if date_iso:
                merged_raw = f"{code} dated {date_iso}"
            rtype = "gazette"
            scope = classify_scope(rtype)
            # Dedupe by canonical_id per page
            canonical_id = code
            key_for_dedup = f"{page_number}|{rtype}|{canonical_id.lower()}"
            if key_for_dedup in seen:
                continue
            seen[key_for_dedup] = True
            hits.append(
                ReferenceHit(
                    reference_text=merged_raw,
                    reference_type=rtype,
                    page_number=page_number,
                    context_snippet=context,
                    method="regex",
                    canonical_id=canonical_id,
                    issuing_body="Government of India",
                    date=date_iso,
                    scope=scope,
                )
            )

        # LLM fallback for ambiguous sentences only if key exists
        if model is not None:
            try:
                llm_candidates = _maybe_llm_fallback_for_page(model, text)
                for cand in llm_candidates:
                    ref_text = cand.get("reference_text", "").strip()
                    if not ref_text:
                        continue
                    rtype = cand.get("reference_type", "other").strip().lower() or "other"
                    # filter to decent confidence; keep light
                    conf_val = float(cand.get("confidence", 0.0) or 0.0)
                    if conf_val < 0.5:
                        continue
                    sentence = cand.get("context_sentence", "")
                    snippet = sentence[:500]
                    canonical_id = None
                    issuing_body = "SEBI"
                    date_iso = None
                    if rtype == "circular":
                        canonical_id = normalize_circular_id(ref_text)
                    if DATED_PHRASE.search(snippet) or DATED_PHRASE.search(ref_text):
                        # We don't have exact offsets; try from the provided sentence first
                        date_iso = normalize_date_mdy(snippet) or normalize_date_mdy(ref_text)
                    # Validate LLM hit
                    if not is_valid_llm_reference(ref_text, rtype, snippet):
                        continue
                    scope = classify_scope(rtype)
                    dkey = _dedup_key_with_canonical(rtype, ref_text, page_number, canonical_id)
                    if dkey in seen:
                        continue
                    # If regex already matched similar text/type on this page, prefer regex (skip LLM)
                    if any(
                        hh.page_number == page_number and hh.reference_type == rtype and (
                            (canonical_id and hh.canonical_id == canonical_id)
                            or (not canonical_id and hh.reference_text.strip().lower() == ref_text.strip().lower())
                        ) and hh.method == "regex"
                        for hh in hits
                    ):
                        continue
                    seen[dkey] = True
                    hits.append(
                        ReferenceHit(
                            reference_text=ref_text,
                            reference_type=rtype,
                            page_number=page_number,
                            context_snippet=snippet,
                            method="llm",
                            canonical_id=canonical_id,
                            issuing_body=issuing_body,
                            date=date_iso,
                            scope=scope,
                            confidence=conf_val,
                        )
                    )
            except Exception:
                # Always continue even if LLM fails
                pass

    # Gazette de-dup: remove dated-only entries that are near merged code hits
    hits = _filter_dated_only_gazettes(hits, pages)
    # Remove substring duplicates per page/type after gathering
    hits = _drop_substring_duplicates(hits)
    # External-only filter if requested
    if external_only:
        hits = [h for h in hits if h.scope == "external"]
    result = ExtractionResult(source_document=source_doc, references=hits)

    # Optional cited-title resolution
    debug_lines: List[str] = []
    try:
        # 1) Annexure map heuristic (v2)
        annex_map = build_annexure_title_map_v2(pages)
        # Build id -> contexts and hits
        circ_map: Dict[str, Dict[str, Any]] = {}
        for h in hits:
            if h.reference_type == "circular" and h.canonical_id:
                entry = circ_map.setdefault(h.canonical_id, {"contexts": [], "hits": []})
                # collect up to 5 contexts per id from FULL document pages
                page_text = pages[h.page_number - 1] if 0 <= h.page_number - 1 < len(pages) else ""
                ref = h.reference_text
                pos = page_text.find(ref) if ref else -1
                if pos != -1:
                    a = max(0, pos - 500)
                    b = min(len(page_text), pos + len(ref) + 500)
                    ctx = reflow_preserve_paragraphs(page_text[a:b].strip())
                    if ctx:
                        entry["contexts"].append(ctx)
                elif h.context_snippet:
                    entry["contexts"].append(reflow_preserve_paragraphs(h.context_snippet))
                entry["hits"].append(h)

        # Apply annexure titles (by ID)
        for cid, bag in circ_map.items():
            if cid in annex_map:
                title = annex_map[cid]
                for h in bag["hits"]:
                    h.cited_title = title

        # Secondary mapping: by date if ID not present in annexure list
        date_title_map = build_annexure_title_map_by_date(pages)
        if date_title_map:
            for cid, bag in circ_map.items():
                if any(h.cited_title for h in bag["hits"]):
                    continue
                # If all hits for this ID share the same normalized ISO date, map title by date
                dates = {h.date for h in bag["hits"] if h.date}
                if len(dates) == 1:
                    iso = next(iter(dates))
                    if iso and iso in date_title_map:
                        for h in bag["hits"]:
                            h.cited_title = date_title_map[iso]

        # 2) LLM guessing if enabled and not already titled
        if guess_titles:
            for cid, bag in circ_map.items():
                if any(h.cited_title for h in bag["hits"]):
                    continue
                contexts = list(dict.fromkeys(bag["contexts"]))[:5]
                if not contexts:
                    continue
                guessed = guess_cited_title(cid, contexts)
                if guessed and guessed.get("cited_title"):
                    for h in bag["hits"]:
                        h.cited_title = guessed.get("cited_title")
                        h.cited_date = guessed.get("cited_date")
                        try:
                            h.cited_title_confidence = float(guessed.get("confidence") or 0.0)
                        except Exception:
                            h.cited_title_confidence = 0.0
                # record debug lines unconditionally (printed later if flag set)
                debug_lines.append(f"ID: {cid}")
                for i, s in enumerate(contexts[:2], 1):
                    s_short = (s[:300] + '…') if len(s) > 300 else s
                    debug_lines.append(f"  snippet{i}: {s_short}")
                if guessed:
                    debug_lines.append(f"  llm: {guessed}")
                else:
                    debug_lines.append("  llm: null")
                debug_lines.append(f"  annexure_map: {'hit' if cid in annex_map else 'miss'}\n")
    except Exception:
        # Never fail the run on title guessing
        pass

    # Light validation warnings to stdout
    _emit_validation_warnings(hits)

    # Export JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

    # Export CSV (flattened)
    csv_fields = [
        "reference_text",
        "reference_type",
        "page_number",
        "context_snippet",
        "method",
        "canonical_id",
        "issuing_body",
        "date",
        "scope",
        "confidence",
        "cited_title",
        "cited_date",
        "cited_title_confidence",
        "source_title",
        "source_pages",
        "source_file_name",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for h in result.references:
            writer.writerow(
                {
                    "reference_text": h.reference_text,
                    "reference_type": h.reference_type,
                    "page_number": h.page_number,
                    "context_snippet": h.context_snippet,
                    "method": h.method,
                    "canonical_id": h.canonical_id,
                    "issuing_body": h.issuing_body,
                    "date": h.date,
                    "scope": h.scope,
                    "confidence": h.confidence,
                    "source_title": result.source_document.title,
                    "source_pages": result.source_document.pages,
                    "source_file_name": result.source_document.file_name,
                    "cited_title": h.cited_title,
                    "cited_date": h.cited_date,
                    "cited_title_confidence": h.cited_title_confidence,
                }
            )

    # Print counts by type and note gazette merging
    counts: Dict[str, int] = {}
    for h in result.references:
        counts[h.reference_type] = counts.get(h.reference_type, 0) + 1
    print("Reference counts by type:")
    for k in sorted(counts.keys()):
        print(f"  {k}: {counts[k]}")
    if summary:
        total = sum(counts.values())
        print(f"\nSummary (external-only={external_only})")
        for k in sorted(counts.keys()):
            print(f"{k}: {counts[k]}")
        print(f"total: {total}")

    # Markdown report
    if md_path:
        _write_markdown_report(result, md_path, external_only)
    # Debug title logs if requested via CLI
    if 'debug_titles' in locals() and debug_titles and debug_lines:
        ts_dir = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(ts_dir, exist_ok=True)
        dbg_path = os.path.join(ts_dir, "title_debug.log")
        with open(dbg_path, "w", encoding="utf-8") as f:
            f.write("\n".join(debug_lines))
        print(f"Title debug log: {dbg_path}")

    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract references from a SEBI PDF using regex, with optional Gemini fallback."
    )
    parser.add_argument("--pdf", required=True, help="Path to input PDF file")
    parser.add_argument("--out", required=True, help="Path to output JSON file")
    parser.add_argument("--csv", required=True, help="Path to output CSV file")
    parser.add_argument("--md", help="Optional path to write a markdown report")
    parser.add_argument(
        "--external-only",
        action="store_true",
        help="Only include references with scope=external",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a compact count table by reference type",
    )
    parser.add_argument(
        "--guess-titles",
        action="store_true",
        help="Attempt to resolve cited circular titles from nearby text only; uses LLM if configured",
    )
    parser.add_argument(
        "--debug-titles",
        action="store_true",
        help="Write per-circular title resolution debug info to runs/<ts>/title_debug.log",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if zero external references found",
    )

    args = parser.parse_args(argv)

    load_dotenv(override=False)

    pdf_path = args.pdf
    out_json = args.out
    out_csv = args.csv
    md_path = args.md
    debug_titles = bool(args.debug_titles)

    if not os.path.isfile(pdf_path):
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    try:
        result = run_extraction(
            pdf_path,
            out_json,
            out_csv,
            external_only=bool(args.external_only),
            summary=bool(args.summary),
            md_path=md_path,
            guess_titles=bool(args.guess_titles),
        )
        if bool(args.strict):
            # Consider external refs only regardless of external_only flag
            refs = result.references
            if len(refs) == 0:
                return 3
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def _write_markdown_report(result: ExtractionResult, md_path: str, external_only_flag: bool) -> None:
    # Header
    lines: List[str] = []
    title = result.source_document.title
    fname = result.source_document.file_name
    pages = result.source_document.pages
    lines.append(f"# {title} — {fname} (pages: {pages})")
    lines.append("")
    # Counts block (same as summary)
    counts: Dict[str, int] = {}
    for h in result.references:
        counts[h.reference_type] = counts.get(h.reference_type, 0) + 1
    total = sum(counts.values())
    # Title coverage for external circulars
    externals = [h for h in result.references if h.scope == "external"]
    ext_circs = [h for h in externals if h.reference_type == "circular"]
    titled = [h for h in ext_circs if (h.cited_title or "").strip()]
    denom = len(ext_circs)
    num = len(titled)
    pct = f"{(num/denom*100):.1f}%" if denom else "n/a"
    lines.append(f"Summary (external-only={external_only_flag})")
    for k in sorted(counts.keys()):
        lines.append(f"{k}: {counts[k]}")
    lines.append(f"total: {total}")
    lines.append(f"Title coverage: {num}/{denom} circulars ({pct})")
    lines.append("")
    # External-only table
    lines.append("| page | type | canonical_id | date | raw | method |")
    lines.append("|---:|---|---|---|---|---|")
    externals = [h for h in result.references if h.scope == "external"]
    externals_sorted = sorted(
        externals,
        key=lambda h: (h.page_number, h.reference_type, h.canonical_id or ""),
    )
    for h in externals_sorted:
        raw = h.reference_text
        if len(raw) > 120:
            raw = raw[:120] + "…"
        # Replace pipe to avoid breaking table
        safe_raw = raw.replace("|", "/")
        lines.append(
            f"| {h.page_number} | {h.reference_type} | {h.canonical_id or ''} | {h.date or ''} | {safe_raw} | {h.method} |"
        )
    content = "\n".join(lines)
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    raise SystemExit(main())

