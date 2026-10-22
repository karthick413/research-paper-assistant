
from pathlib import Path

import faiss
import numpy as np
import ollama

from chunker import chunk_pages
from document_processor import extract_pages


PAPERS_DIR = Path("data/papers")
EMBED_MODEL = "nomic-embed-text"


def load_chunks() -> list[dict]:
    """Extract and chunk every PDF in the papers directory."""
    chunks = []

    for pdf_path in sorted(PAPERS_DIR.glob("*.pdf")):
        pages = extract_pages(str(pdf_path))
        pdf_chunks = chunk_pages(pages)
        chunks.extend(pdf_chunks)

        print(f"Loaded {pdf_path.name}: {len(pdf_chunks)} chunks")

    return chunks


def embed_texts(texts: list[str]) -> np.ndarray:
    """Convert texts to normalized float32 embedding vectors."""
    response = ollama.embed(
        model=EMBED_MODEL,
        input=texts,
    )

    vectors = np.asarray(
        response["embeddings"],
        dtype=np.float32,
    )

    faiss.normalize_L2(vectors)
    return vectors


def build_index(chunks: list[dict]):
    """Build an exact FAISS cosine-similarity search index."""
    if not chunks:
        raise ValueError(
            "No chunks found. Check that PDFs exist in data/papers."
        )

    # Embed in batches to keep memory usage manageable.
    batch_size = 32
    vector_batches = []

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embed_texts([item["text"] for item in batch])
        vector_batches.append(vectors)

        print(
            f"Embedded {min(start + batch_size, len(chunks))}"
            f"/{len(chunks)} chunks"
        )

    all_vectors = np.vstack(vector_batches)

    index = faiss.IndexFlatIP(all_vectors.shape[1])
    index.add(all_vectors)

    return index


def search(
    index,
    chunks: list[dict],
    question: str,
    k: int = 4,
    source_filter: str | None = None,
) -> list[dict]:
    """Retrieve relevant passages, optionally from one PDF."""

    if index.ntotal == 0:
        return []

    query_vector = embed_texts([question])

    # If filtering by PDF, inspect the full index so that
    # relevant passages from that PDF are not missed.
    search_k = index.ntotal if source_filter else min(k, index.ntotal)

    scores, indices = index.search(query_vector, search_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        chunk = chunks[int(idx)]

        if source_filter and chunk["source"] != source_filter:
            continue

        results.append(
            {
                "score": float(score),
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
            }
        )

        if len(results) >= k:
            break

    return results


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"\nTotal chunks: {len(chunks)}")

    index = build_index(chunks)
    print(f"FAISS vectors stored: {index.ntotal}")

    question = "How does the Transformer architecture use attention?"

    print(f"\nQuestion: {question}")
    print("\nTOP MATCHES")

    for rank, result in enumerate(
        search(index, chunks, question, k=4),
        start=1,
    ):
        print(f"\nRank: {rank}")
        print(f"Similarity: {result['score']:.4f}")
        print(
            f"Source: {result['source']}, "
            f"page {result['page']}"
        )
        print(f"Text: {result['text'][:350]}")
