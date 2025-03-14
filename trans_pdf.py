import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm 

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    return text

directory = Path("../academic_db")

pdf_files = list(directory.glob("*.pdf"))
print(pdf_files)

for pdf in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
    text = extract_text_from_pdf(pdf)

    txt_file = pdf.with_suffix(".txt")

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(text)
