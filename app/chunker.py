
from pathlib import Path

from document_processor import extract_pages


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[dict]:
    """Split page text into overlapping character-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": page["source"],
                        "page": page["page"],
                    }
                )

            if end == len(text):
                break

            start = end - overlap

    return chunks


if __name__ == "__main__":
    papers_folder = Path("data/papers")
    all_chunks = []

    for pdf_file in sorted(papers_folder.glob("*.pdf")):
        pages = extract_pages(str(pdf_file))
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)

        print(f"\nPDF: {pdf_file.name}")
        print(f"Pages extracted: {len(pages)}")
        print(f"Chunks created: {len(chunks)}")

    print(f"\nTOTAL CHUNKS: {len(all_chunks)}")

    if all_chunks:
        print("\nFIRST CHUNK")
        print("Source:", all_chunks[0]["source"])
        print("Page:", all_chunks[0]["page"])
        print("Text:", all_chunks[0]["text"][:300])
