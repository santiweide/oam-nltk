import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm
import argparse

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    # Extract text from all pages
    text = "\n".join(page.get_text("text") for page in doc)
    return text

def main(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {input_path.resolve()}.")

    for pdf in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        text = extract_text_from_pdf(pdf)
        
        txt_filename = pdf.stem + ".txt"
        txt_file_path = output_path / txt_filename

        # Write in UTF-8 to preserve multilingual content (English, French, Arabic)
        with open(txt_file_path, "w", encoding="utf-8") as f:
            f.write(text)

    print(f"Text files saved to {output_path.resolve()}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from PDFs with multilingual support.")
    parser.add_argument("input_dir", help="Path to the directory containing PDF files.")
    parser.add_argument("output_dir", help="Path to the directory where text files will be saved.")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
