
from pathlib import Path
import json
import re
from collections import Counter

import faiss
import numpy as np

from app.config import (
    PAPERS_DIR,
    STORAGE_DIR,
    EMBED_MODEL,
    EMBED_BATCH_SIZE,
    TOP_K,
)
from app.document_processor import extract_pages
from app.chunker import chunk_pages
from app.ollama_compat import embed_texts


INDEX_PATH = STORAGE_DIR / "papers.faiss"
METADATA_PATH = STORAGE_DIR / "papers_metadata.json"


def _embed_in_batches(texts: list[str]) -> np.ndarray:
    """Embed text in batches and return normalized float32 vectors."""
    if not texts:
        raise ValueError("Cannot create embeddings from an empty text list.")

    if EMBED_BATCH_SIZE <= 0:
        raise ValueError("EMBED_BATCH_SIZE must be greater than zero.")

    batches = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        vectors = embed_texts(
            texts[start:start + EMBED_BATCH_SIZE],
            EMBED_MODEL,
        )
        batches.append(np.asarray(vectors, dtype=np.float32))

    matrix = np.vstack(batches)

    if matrix.ndim != 2 or matrix.shape[0] != len(texts):
        raise RuntimeError("The embedding model returned an invalid matrix.")

    if not np.isfinite(matrix).all():
        raise RuntimeError("Embeddings contain invalid numeric values.")

    if matrix.shape[1] == 0:
        raise RuntimeError("The embedding model returned empty vectors.")

    faiss.normalize_L2(matrix)
    return matrix


def build_index(pdf_dir: str | Path = PAPERS_DIR) -> dict:
    """Extract PDF text, chunk it, embed it, and save a FAISS index."""
    folder = Path(pdf_dir)

    if not folder.is_dir():
        raise NotADirectoryError(f"PDF directory not found: {folder}")

    pdf_files = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        raise ValueError(f"No PDF files found in {folder}")

    all_pages = []

    for pdf_path in pdf_files:
        all_pages.extend(extract_pages(pdf_path))

    chunks = chunk_pages(all_pages)

    if not chunks:
        raise ValueError(
            "No readable text was extracted. Scanned PDFs may need OCR."
        )

    vectors = _embed_in_batches([chunk["text"] for chunk in chunks])

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "embedding_model": EMBED_MODEL,
        "vector_dimension": int(vectors.shape[1]),
        "pdf_files": [path.name for path in pdf_files],
        "chunks": chunks,
    }

    temporary_index = INDEX_PATH.with_suffix(".faiss.tmp")
    temporary_metadata = METADATA_PATH.with_suffix(".json.tmp")

    try:
        faiss.write_index(index, str(temporary_index))
        temporary_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_index.replace(INDEX_PATH)
        temporary_metadata.replace(METADATA_PATH)
    finally:
        if temporary_index.exists():
            temporary_index.unlink()
        if temporary_metadata.exists():
            temporary_metadata.unlink()

    return {
        "pdf_count": len(pdf_files),
        "chunk_count": len(chunks),
        "vector_count": int(index.ntotal),
        "vector_dimension": int(vectors.shape[1]),
        "pdf_files": [path.name for path in pdf_files],
    }


def load_index() -> tuple[faiss.Index, list[dict], dict]:
    """Load and validate the saved index and chunk metadata."""
    if not INDEX_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError(
            "No saved index found. Build the paper index first."
        )

    index = faiss.read_index(str(INDEX_PATH))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    if metadata.get("embedding_model") != EMBED_MODEL:
        raise ValueError(
            "The saved index uses a different embedding model. "
            "Rebuild the index."
        )

    chunks = metadata.get("chunks", [])

    if index.ntotal != len(chunks):
        raise ValueError("Index and metadata sizes do not match.")

    if index.d != metadata.get("vector_dimension"):
        raise ValueError("Index and metadata vector dimensions do not match.")

    return index, chunks, metadata


def _tokenize(text: str) -> list[str]:
    """Extract searchable words, including technical terms and acronyms."""
    return re.findall(r"[a-zA-Z0-9]+(?:[._/-][a-zA-Z0-9]+)*", text.lower())


def _keyword_scores(query: str, chunks: list[dict]) -> list[float]:
    """Score chunks by query-term coverage with simple inverse frequency."""
    query_terms = set(_tokenize(query))

    if not query_terms:
        return [0.0] * len(chunks)

    document_terms = [
        set(_tokenize(chunk.get("text", "")))
        for chunk in chunks
    ]

    document_frequency = Counter(
        term
        for terms in document_terms
        for term in terms
        if term in query_terms
    )

    weights = {
        term: np.log(1 + (len(chunks) + 1) /
                     (document_frequency.get(term, 0) + 1))
        for term in query_terms
    }

    scores = []

    for terms in document_terms:
        matched = query_terms.intersection(terms)

        if not matched:
            scores.append(0.0)
            continue

        score = sum(weights[term] for term in matched)
        total = sum(weights.values()) or 1.0
        scores.append(float(score / total))

    return scores


def search_index(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    top_k: int = TOP_K,
    source_filter: str | None = None,
) -> list[dict]:
    """
    Hybrid search using semantic similarity and keyword matching.

    Reciprocal-rank fusion combines both rankings without assuming that
    semantic and keyword scores use the same scale.
    """
    query = query.strip()

    if not query:
        return []

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if index.ntotal != len(chunks):
        raise ValueError("Index and chunk metadata sizes do not match.")

    eligible = [
        i for i, chunk in enumerate(chunks)
        if not source_filter or chunk.get("source") == source_filter
    ]

    if not eligible:
        return []

    query_vector = _embed_in_batches([query])

    # Search the full index, then restrict results to the selected source.
    semantic_count = min(index.ntotal, max(top_k * 10, 50))
    semantic_scores, semantic_positions = index.search(
        query_vector,
        semantic_count,
    )

    semantic_rank = {}
    rank = 1

    for score, position in zip(
        semantic_scores[0], semantic_positions[0]
    ):
        position = int(position)

        if position < 0 or position not in eligible:
            continue

        semantic_rank[position] = rank
        rank += 1

    eligible_chunks = [chunks[i] for i in eligible]
    keyword_values = _keyword_scores(query, eligible_chunks)

    keyword_order = sorted(
        range(len(eligible)),
        key=lambda j: keyword_values[j],
        reverse=True,
    )

    keyword_rank = {
        eligible[j]: rank
        for rank, j in enumerate(keyword_order, start=1)
        if keyword_values[j] > 0
    }

    # Reciprocal-rank fusion rewards chunks ranked highly by either method.
    rrf_constant = 60
    fused = []

    for position in eligible:
        semantic_r = semantic_rank.get(position)
        keyword_r = keyword_rank.get(position)

        fusion_score = 0.0

        if semantic_r is not None:
            fusion_score += 1.0 / (rrf_constant + semantic_r)

        if keyword_r is not None:
            fusion_score += 1.0 / (rrf_constant + keyword_r)

        if fusion_score == 0:
            continue

        result = dict(chunks[position])
        result["score"] = float(fusion_score)
        result["semantic_rank"] = semantic_r
        result["keyword_rank"] = keyword_r
        fused.append(result)

    fused.sort(key=lambda item: item["score"], reverse=True)
    return fused[:top_k]
