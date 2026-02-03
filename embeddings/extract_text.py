import pdfplumber
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path(r"C:\medical-ai-copilot\data\raw_docs")
OUT_DIR = Path(r"C:\medical-ai-copilot\data\processed")

OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text

for pdf_path in tqdm(list(RAW_DIR.glob("*.pdf"))):
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            text = clean_text(text)
            if len(text) > 50:  # skip empty/noisy pages
                all_text.append(text)

    output_file = OUT_DIR / f"{pdf_path.stem}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_text))

    print(f"Saved: {output_file.name}")
