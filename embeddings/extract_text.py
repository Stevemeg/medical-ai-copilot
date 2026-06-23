import json
import pdfplumber
from pathlib import Path
from tqdm import tqdm

# Project root is the parent of this file's parent directory (embeddings/ -> project root)
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw_docs"
OUT_DIR = BASE_DIR / "data" / "processed"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# pdfplumber.PDF.pages is a cached property: the first access builds and
# retains a Page object for every page in the document for the life of the
# `with pdfplumber.open(...)` block. On a large PDF (the 1358-page anatomy
# textbook here) this grows to several GB of resident memory as you iterate
# and the process gets OOM-killed partway through extraction (verified:
# unbounded RSS hits ~3.7GB by page 1000). Re-opening the file in fixed-size
# page batches keeps only one batch's worth of Page objects alive at a time,
# bounding peak RSS to ~650MB for this file while keeping the per-batch
# reparsing overhead small.
PAGE_BATCH_SIZE = 100


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def extract_pdf_text(pdf_path: Path) -> list[dict]:
    """
    Returns a list of {"page": page_number, "text": cleaned_text} dicts,
    one per non-empty page. Page numbers are preserved here (instead of
    being discarded by flattening all pages into one string) so that
    downstream chunking can tag each chunk with the page(s) it came from,
    enabling real page-level citations instead of filename-only attribution.
    """
    with pdfplumber.open(pdf_path) as probe:
        n_pages = len(probe.pages)

    pages_out = []
    for start in range(0, n_pages, PAGE_BATCH_SIZE):
        page_numbers = list(range(start + 1, min(start + PAGE_BATCH_SIZE, n_pages) + 1))
        with pdfplumber.open(pdf_path, pages=page_numbers) as pdf:
            for page in pdf.pages:
                text = clean_text(page.extract_text())
                if len(text) > 50:  # skip empty/noisy pages
                    pages_out.append({"page": page.page_number, "text": text})
    return pages_out


for pdf_path in tqdm(list(RAW_DIR.glob("*.pdf"))):
    pages_out = extract_pdf_text(pdf_path)

    output_file = OUT_DIR / f"{pdf_path.stem}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pages_out, f, indent=2)

    print(f"Saved: {output_file.name} ({len(pages_out)} pages)")