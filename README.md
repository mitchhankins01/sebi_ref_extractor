## sebi_ref_extractor

Minimal Python 3.11 tool that extracts references (to other circulars, regulations, acts, gazette notifications, URLs) from a SEBI PDF. It uses robust regexes first and a constrained Gemini fallback only when needed. Every reference is page-anchored and exported to JSON/CSV; an optional markdown report is available.

### Features
- Page-anchored references with per-hit context snippet
- Types: circular, regulation, section, gazette, url (internal items like Annexure/Chapter are also detected but can be filtered out). Optional: press_release.
- Source doc metadata (title, total pages, file name)
- Optional Gemini fallback (strict JSON, no hallucinations; low-confidence or placeholder hits are discarded)
- Exports JSON, CSV, and optional markdown

### Requirements
- Python 3.11+

### Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Environment variables (optional for Gemini fallback):
- Copy `env.example` to `.env` and set `GEMINI_API_KEY`. If unset, the extractor runs in regex-only mode.

```bash
cp env.example .env
echo 'GEMINI_API_KEY=your_key_here' >> .env
```

### Project structure
```
sebi_ref_extractor/
├─ extract_references.py      # CLI + end-to-end pipeline
├─ patterns.py                # compiled regexes + guess_type + helpers
├─ utils.py                   # PDF text, context, normalization, LLM/title helpers
├─ models.py                  # Pydantic models (ReferenceHit, SourceDoc, ExtractionResult)
├─ scripts/
│  └─ run_all.sh              # batch all PDFs in samples/, writes runs/<timestamp>/
├─ tests/
│  └─ test_utils.py           # normalizer + LLM-filter unit tests
├─ samples/
│  └─ README.txt              # how to drop PDFs and run a quick command
├─ requirements.txt
├─ env.example                # copy to .env for GEMINI_API_KEY (optional)
└─ README.md
```

### Usage (single PDF)
```bash
# default (all refs) + markdown report (write to runs/dev)
python extract_references.py --pdf "./samples/1. Recent, citation-dense Master Circular (2024–2025).pdf" --out runs/dev/master_recent_2024_2025.json --csv runs/dev/master_recent_2024_2025.csv --summary --md runs/dev/master_recent_2024_2025.md

# external-only (filters out Annexure/Chapter/etc.) + markdown
python extract_references.py --pdf "./samples/1. Recent, citation-dense Master Circular (2024–2025).pdf" --out runs/dev/master_recent_2024_2025.external.json --csv runs/dev/master_recent_2024_2025.external.csv --external-only --summary --md runs/dev/master_recent_2024_2025.external.md

# show version
python extract_references.py --version

# try to resolve cited circular titles from nearby text only (no web requests)
python extract_references.py --pdf "./samples/1. Recent, citation-dense Master Circular (2024–2025).pdf" \
  --out runs/dev/master_recent_2024_2025.json --csv runs/dev/master_recent_2024_2025.csv --md runs/dev/master_recent_2024_2025.md --external-only --guess-titles --summary
```

Outputs include for each hit: `reference_text`, `reference_type`, `page_number`, `context_snippet`, `method`, `canonical_id`, `issuing_body`, `date` (ISO), `scope`, `confidence`, and (when resolved) `cited_title`, `cited_date`, `cited_title_confidence`.

### Output schema (JSON)
```json
{
  "source_document": {
    "title": "Master Circular for Electronic Gold Receipts (EGRs)",
    "pages": 52,
    "file_name": "egr_master_2023.pdf"
  },
  "references": [
    {
      "reference_text": "SEBI/HO/MRD/MRD-PoD-1/P/CIR/2023/82",
      "reference_type": "circular",
      "page_number": 1,
      "context_snippet": "... page text around the match ...",
      "method": "regex",
      "canonical_id": "SEBI/HO/MRD/MRD-PoD-1/P/CIR/2023/82",
      "issuing_body": "SEBI",
      "date": null,
      "scope": "external",
      "confidence": null,
      "cited_title": null,
      "cited_date": null,
      "cited_title_confidence": null
    }
  ]
}
```

CSV contains the same rows, flattened, with additional source document columns (`source_title`, `source_pages`, `source_file_name`).

### Title extraction
- Regex on page 1 using common patterns like `Subject:` or line after a `CIRCULAR` header.
- If not found and `GEMINI_API_KEY` is set, a single Gemini call is made with a strict JSON schema: `{ "title": "" }`.

### LLM fallback for ambiguous mentions
Only triggered for sentences with phrases like “vide circular”, “in partial modification”, “as per regulation”, etc., that didn’t match regex patterns. Prompt enforces literal extraction and returns:

```json
{"references":[{"reference_text":"","reference_type":"","confidence":0.0}]}
```

LLM results are deduped per page and only kept with reasonable confidence. LLM hits without a concrete identifier (e.g., “SEBI Circular No.”) are discarded to prevent hallucinations.

### Notes and limitations
- Works best on digitally generated PDFs; OCR’d scans may have broken text or hyphenation
- Regex-first captures most canonical IDs; narrative-only mentions may rely on LLM and strict filters
- Gemini usage is intentionally minimal: it is only invoked for clearly ambiguous phrases and optional cited-title guessing. On many samples, snippets don’t contain explicit titles, so LLM adds little and is frequently skipped by filters. The extractor remains primarily regex-driven.
- If Gemini is unavailable, the tool runs regex-only and does not fail
- Every reference always has a page anchor

### V2 improvements (roadmap)
- Dockerize the tool for easy deployment and team collaboration
- OCR and layout: add OCR and layout-aware extraction (word coordinates, table parsing) to handle scanned PDFs and wrapped lines better
- Smarter Annexure parsing: parse List-of-Circulars tables with structure-aware heuristics to link IDs ↔ titles even when split across columns/lines
- Title resolution: augment with a small local index of known SEBI IDs ↔ titles, plus cached lookups; relax filters with confidence thresholds when context is strong
- Model routing: allow configurable model choice (e.g., Gemini Pro/Flash) and retries; batch LLM calls per document to reduce latency
- Knowledge graph export: emit normalized nodes/edges for direct ingestion into a graph DB (IDs, dates, types, page anchors, contexts)

### Development
- All regex patterns are compiled once in `patterns.py`
- Utilities in `utils.py` handle page extraction (PyMuPDF), context trimming, title heuristics, and safe JSON parsing
- Models in `models.py` (Pydantic v2) define the output schema

### Batch
Process all PDFs under `samples/` into a timestamped run folder, with an index report:
```bash
bash scripts/run_all.sh
```
The index is written to `runs/<timestamp>/index.md` and shows counts by type and how many rows resolved `cited_title`. Markdown reports also show a “Title coverage” line (e.g., `Title coverage: 6/11 circulars (54.5%)`).

### Deliverable 1 note
We capture page numbers for every reference and attempt to resolve cited circular titles via Annexure heuristics and a constrained LLM pass; the markdown report shows per-file title coverage.

### Tests
Install deps and run:
```bash
pip install -r requirements.txt
pytest -q
```
