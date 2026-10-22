
from pathlib import Path

import pymupdf  # PyMuPDF


def extract_pages(pdf_path: str) -> list[dict]:
    """Extract text while preserving PDF page numbers."""

    pdf_file = Path(pdf_path)
    pages = []

    with pymupdf.open(pdf_file) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    {
                        "source": pdf_file.name,
                        "page": page_number,
                        "text": text,
                    }
                )

    return pages


if __name__ == "__main__":
    papers_folder = Path("data/papers")
    pdf_files = list(papers_folder.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in data/papers/")
    else:
        for pdf_file in pdf_files:
            pages = extract_pages(str(pdf_file))

            print(f"\nPDF: {pdf_file.name}")
            print(f"Pages with extracted text: {len(pages)}")

            if pages:
                print("First page preview:")
                print(pages[0]["text"][:500])
