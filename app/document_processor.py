
from pathlib import Path

import fitz  # PyMuPDF


def extract_pages(pdf_path: str | Path) -> list[dict]:
    """
    Extract readable text from a PDF.

    Returns:
        A list of dictionaries containing:
        - source: PDF filename
        - page: Original, 1-based page number
        - text: Extracted page text
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"The provided path is not a file: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, received: {path.name}"
        )

    pages = []

    try:
        with fitz.open(path) as pdf:
            if pdf.page_count == 0:
                raise ValueError(
                    f"The PDF contains no pages: {path.name}"
                )

            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                # Skip pages with no extractable text.
                if not text:
                    continue

                pages.append(
                    {
                        "source": path.name,
                        "page": page_number,
                        "text": text,
                    }
                )

    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Could not read PDF '{path.name}': {exc}"
        ) from exc

    return pages


def extract_pdf_info(pdf_path: str | Path) -> dict:
    """
    Return basic PDF information without extracting all page text.
    """
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {path}"
        )

    try:
        with fitz.open(path) as pdf:
            return {
                "source": path.name,
                "total_pages": pdf.page_count,
                "metadata": pdf.metadata or {},
            }
    except Exception as exc:
        raise RuntimeError(
            f"Could not inspect PDF '{path.name}': {exc}"
        ) from exc
