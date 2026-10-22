
import re
import ollama

from vector_store import load_chunks, build_index, search


LLM_MODEL = "llama3.2:3b"

PAPERS = {
    "1706.03762v7.pdf": "Attention Is All You Need",
    "2005.11401v4.pdf": (
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    ),
    "2307.03172v3.pdf": (
        "Lost in the Middle: How Language Models Use Long Contexts"
    ),
}


def identify_paper(question: str) -> str | None:
    """Identify a paper when the user names it explicitly."""
    q = question.lower()

    for filename, title in PAPERS.items():
        if title.lower() in q or filename.lower() in q:
            return filename

    # Handle common shortened titles.
    if "attention is all you need" in q:
        return "1706.03762v7.pdf"

    if (
        "retrieval-augmented generation" in q
        or "rag paper" in q
    ):
        return "2005.11401v4.pdf"

    if "lost in the middle" in q:
        return "2307.03172v3.pdf"

    return None


def is_summary_question(question: str) -> bool:
    """Detect requests for a paper summary."""
    q = question.lower()

    summary_phrases = (
        "summarize",
        "summary",
        "overview",
        "give me a summary",
    )

    return any(phrase in q for phrase in summary_phrases)


def validate_citations(
    answer: str,
    sources: list[dict],
) -> list[str]:
    """
    Flag page citations that do not match the retrieved evidence.

    This checks citation references, not whether the cited passage
    semantically supports the associated claim.
    """
    allowed = {
        (item["source"], int(item["page"]))
        for item in sources
    }

    found = re.findall(
        r"\[([^\],]+),\s*p\.?\s*(\d+)\]",
        answer,
        flags=re.IGNORECASE,
    )

    invalid = []

    for filename, page in found:
        citation = (filename.strip(), int(page))

        if citation not in allowed:
            invalid.append(
                f"[{citation[0]}, p. {citation[1]}]"
            )

    return sorted(set(invalid))


def find_uncited_paragraphs(answer: str) -> list[str]:
    """
    Find paragraphs that lack a page citation at the end.

    Standalone headings are ignored. This is a formatting check,
    not a semantic verification of factual claims.
    """
    uncited = []

    paragraphs = re.split(
        r"\n\s*\n",
        answer.strip(),
    )

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # Skip standalone Markdown headings.
        if re.match(r"^#{1,6}\s+", paragraph):
            continue

        # Skip standalone bold headings.
        if re.fullmatch(r"\*\*[^*]+\*\*", paragraph):
            continue

        # Require a page citation at the end of the paragraph.
        if not re.search(
            r"\[[^\[\]]+,\s*p\.?\s*\d+\]\s*$",
            paragraph,
            flags=re.IGNORECASE,
        ):
            uncited.append(paragraph[:180])

    return uncited


def answer_question_with_sources(
    question: str,
    index,
    chunks: list[dict],
    k: int | None = None,
):
    """
    Answer a question using retrieved PDF passages.

    Returns:
        answer: Generated answer text.
        results: Retrieved passages and their metadata.
        diagnostics: Citation and evidence checks.
    """
    summary_mode = is_summary_question(question)
    paper_filter = identify_paper(question)

    # Retrieve more evidence for summaries than for ordinary questions.
    if k is None:
        k = 6 if summary_mode else 4

    results = search(
        index=index,
        chunks=chunks,
        question=question,
        k=k,
        source_filter=paper_filter,
    )

    diagnostics = {
        "invalid_citations": [],
        "uncited_paragraphs": [],
        "paper_filter": paper_filter,
        "evidence_found": bool(results),
    }

    if not results:
        return (
            "I couldn't find relevant evidence in the loaded papers "
            "to answer that question. Try mentioning a paper title "
            "or asking about a specific concept.",
            [],
            diagnostics,
        )

    # Build the context and preserve exact source/page metadata.
    context_parts = []

    for item in results:
        context_parts.append(
            f"[{item['source']}, p. {item['page']}]\n"
            f"{item['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    if summary_mode:
        task = """
Write a concise research-paper summary using these sections:

**Research Problem**

**Main Contributions**

**Methodology**

**Findings**

**Limitations or Caveats**

STRICT EVIDENCE AND CITATION RULES:

1. Use only claims directly supported by the retrieved evidence.

2. Every factual paragraph, including the Research Problem section,
   must end with at least one citation in this exact format:
   [filename.pdf, p. 3]

3. Copy filenames and page numbers exactly from the evidence labels.

4. Each citation must support the specific claim in its paragraph.
   Do not cite a page merely because it discusses a related topic.

5. Never invent experimental results, performance numbers,
   limitations, author claims, or conclusions.

6. Do not use general knowledge to fill gaps in the retrieved evidence.

7. If evidence is insufficient for a section, state this clearly.
   Cite the retrieved evidence when it directly supports that statement.

8. For Limitations or Caveats, report only limitations explicitly
   supported by the retrieved passages.

9. If the retrieved passages do not establish a specific limitation,
   say exactly:
   "The retrieved passages do not establish a specific limitation."
   Do not add a contradictory limitation afterward.

10. Do not claim state-of-the-art results unless the retrieved
    passages explicitly support that claim.

11. Distinguish the paper's reported findings from your interpretation.

12. Keep the summary concise and avoid repeating the same point.

13. Put each section heading on its own line, followed by a blank
    line and then the section's paragraph.

14. Put citations at the end of the factual paragraph, not on a
    separate line. Do not output uncited factual paragraphs.
"""
    else:
        task = """
Answer the user's question directly using only the retrieved evidence.

STRICT EVIDENCE AND CITATION RULES:

1. Every factual paragraph must end with a citation in this format:
   [filename.pdf, p. 3]

2. Copy filenames and page numbers exactly from the evidence labels.

3. Citations must support the specific claims in their paragraphs.

4. Do not invent details or fill evidence gaps using general knowledge.

5. If the evidence is insufficient, say so clearly.

6. Do not cite pages that were not included in the retrieved evidence.

7. Put each paragraph on its own block, separated by a blank line.

8. Put the citation at the end of the paragraph, not on a separate line.
"""

    prompt = f"""
You are a careful research-paper assistant.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}

INSTRUCTIONS:
{task}

Write the answer now. Do not add a bibliography of sources that
you did not use. Citations must refer only to the retrieved evidence
above.
"""

    # Generate the answer with one Ollama chat call.
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions about research papers accurately. "
                    "Never fabricate evidence or citations. "
                    "If the evidence does not support a claim, do not "
                    "present that claim as a fact."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={"temperature": 0.1},
    )

    answer = response["message"]["content"].strip()

    # Run citation diagnostics.
    diagnostics["invalid_citations"] = validate_citations(
        answer,
        results,
    )

    diagnostics["uncited_paragraphs"] = find_uncited_paragraphs(
        answer
    )

    return answer, results, diagnostics


def answer_question(
    question: str,
    index,
    chunks: list[dict],
) -> str:
    """Compatibility wrapper for callers that only need the answer."""
    answer, _, _ = answer_question_with_sources(
        question,
        index,
        chunks,
    )

    return answer


if __name__ == "__main__":
    print("Loading paper chunks and building the FAISS index...")

    chunks = load_chunks()
    index = build_index(chunks)

    print(f"\nReady! Indexed {len(chunks)} chunks.")

    while True:
        question = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        answer, sources, diagnostics = answer_question_with_sources(
            question,
            index,
            chunks,
        )

        print("\nANSWER\n")
        print(answer)

        print("\nRETRIEVED SOURCES")

        for item in sources:
            print(
                f"- {item['source']}, "
                f"page {item['page']}, "
                f"similarity {item['score']:.3f}"
            )

        if diagnostics["invalid_citations"]:
            print("\nWARNING: Invalid citations:")

            for citation in diagnostics["invalid_citations"]:
                print(f"- {citation}")

        if diagnostics["uncited_paragraphs"]:
            print("\nWARNING: Paragraphs without ending citations:")

            for paragraph in diagnostics["uncited_paragraphs"]:
                print(f"- {paragraph}")
