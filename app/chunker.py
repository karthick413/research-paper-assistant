
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split extracted PDF pages into overlapping text chunks.

    Each chunk preserves its PDF filename and original page number.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        source = page.get("source")
        page_number = page.get("page")
        text = " ".join(str(page.get("text", "")).split())

        if not text:
            continue

        if not source or page_number is None:
            raise ValueError(
                "Every page must contain 'source', 'page', and 'text'."
            )

        for start in range(0, len(text), step):
            chunk_text = text[start:start + chunk_size].strip()

            if chunk_text:
                chunks.append(
                    {
                        "source": str(source),
                        "page": int(page_number),
                        "text": chunk_text,
                    }
                )

            if start + chunk_size >= len(text):
                break

    return chunks
