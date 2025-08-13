Drop your SEBI PDFs in this folder. Example:

  egr_master_2023.pdf

Quick run from project root (all refs):

  python extract_references.py --pdf ./samples/egr_master_2023.pdf --out out.json --csv out.csv --summary --md out.md

External-only (filter Annexure/Chapter) and markdown:

  python extract_references.py --pdf ./samples/egr_master_2023.pdf --out out.external.json --csv out.external.csv --external-only --summary --md out.external.md


