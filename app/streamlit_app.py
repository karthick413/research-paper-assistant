
from pathlib import Path
import re

import streamlit as st

from app.config import (
    LLM_MODEL,
    EMBED_MODEL,
    PAPERS_DIR,
    TOP_K,
)
from app.ollama_compat import check_required_models
from app.vector_store import build_index, load_index
from app.rag_pipeline import answer_question


st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Research Paper Assistant")
st.write(
    "Upload research papers, ask questions, and explore answers "
    "with supporting passages and page references."
)

st.session_state.setdefault("index", None)
st.session_state.setdefault("chunks", [])
st.session_state.setdefault("metadata", {})
st.session_state.setdefault("messages", [])


def load_saved_index():
    """Load the saved FAISS index into the current session."""
    try:
        index, chunks, metadata = load_index()
        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.metadata = metadata
        return True, f"Loaded {index.ntotal} searchable chunks."

    except FileNotFoundError:
        st.session_state.index = None
        st.session_state.chunks = []
        st.session_state.metadata = {}
        return False, "No saved index found. Build an index first."

    except Exception as exc:
        return False, f"Could not load the index: {exc}"


def safe_filename(filename):
    """Remove directory components and unsafe filename characters."""
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._() -]", "_", name)
    name = name.strip(" .")

    if not name or name.lower() in {".", ".."}:
        raise ValueError("The uploaded file has an invalid filename.")

    if not name.lower().endswith(".pdf"):
        raise ValueError(f"'{name}' is not a PDF file.")

    return name


def save_uploaded_pdfs(uploaded_files):
    """Save uploaded PDFs without overwriting existing papers."""
    saved = []
    errors = []

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        try:
            filename = safe_filename(uploaded_file.name)
            content = uploaded_file.getvalue()

            if not content.startswith(b"%PDF-"):
                raise ValueError(
                    f"'{filename}' does not appear to be a valid PDF."
                )

            destination = PAPERS_DIR / filename

            # Never overwrite an existing paper.
            if destination.exists():
                stem = destination.stem
                suffix = destination.suffix
                number = 2

                while destination.exists():
                    destination = PAPERS_DIR / (
                        f"{stem}_{number}{suffix}"
                    )
                    number += 1

            destination.write_bytes(content)
            saved.append(destination.name)

        except Exception as exc:
            errors.append(f"{uploaded_file.name}: {exc}")

    return saved, errors


def rebuild_index():
    """Build a fresh index and load it into this session."""
    stats = build_index()
    index, chunks, metadata = load_index()

    st.session_state.index = index
    st.session_state.chunks = chunks
    st.session_state.metadata = metadata

    # Clear old conversation because the searchable library has changed.
    st.session_state.messages = []

    return stats


def display_assistant_details(message):
    """Display citation status, source metadata, and retrieved passages safely."""
    citation_status = message.get("citation_status", "unknown")

    if citation_status == "needs_review":
        st.warning(
            "Citations need review. Compare the answer with the retrieved "
            "passages before relying on it."
        )
    elif citation_status == "no_evidence":
        st.info("No relevant evidence was retrieved for this question.")
    elif citation_status == "verified_ids":
        st.caption(
            "Citation IDs matched retrieved metadata. This does not prove "
            "that every claim is supported by its cited passage."
        )

    sources = message.get("sources") or []
    if sources:
        with st.expander("Cited sources", expanded=False):
            for source in sources:
                if not isinstance(source, dict):
                    st.write(str(source))
                    continue

                filename = source.get("source") or source.get("filename") or "Unknown source"
                page = source.get("page", "?")
                score = source.get("score")
                line = f"- {filename} — page {page}"
                if isinstance(score, (int, float)):
                    line += f" (retrieval score: {score:.3f})"
                st.markdown(line)

    evidence = message.get("evidence") or []
    if evidence:
        # Show unique source/page pairs in a compact list.
        seen = set()
        retrieved_sources = []
        for item in evidence:
            if not isinstance(item, dict):
                continue
            filename = item.get("source") or item.get("filename") or "Unknown source"
            page = item.get("page", "?")
            key = (filename, page)
            if key not in seen:
                seen.add(key)
                retrieved_sources.append((filename, page))

        with st.expander("Retrieved sources (search results)", expanded=False):
            for filename, page in retrieved_sources:
                st.write(f"- {filename} — page {page}")

        with st.expander("Retrieved evidence passages", expanded=False):
            for number, item in enumerate(evidence, start=1):
                if not isinstance(item, dict):
                    st.write(str(item))
                    continue
                filename = item.get("source") or item.get("filename") or "Unknown source"
                page = item.get("page", "?")
                text = item.get("text") or item.get("content") or "No passage text was provided."
                st.markdown(f"**{number}. {filename} — page {page}**")
                st.write(text)


# Load an existing index when the app first starts.
if st.session_state.index is None:
    load_saved_index()


# --------------------------------------------------
# Sidebar: settings, PDF upload, and index controls
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    st.caption(f"Chat model: `{LLM_MODEL}`")
    st.caption(f"Embedding model: `{EMBED_MODEL}`")

    if st.button("Check Ollama", use_container_width=True):
        with st.spinner("Checking Ollama models..."):
            status = check_required_models(LLM_MODEL, EMBED_MODEL)

        if status["ok"]:
            st.success("Both models are available.")
            st.caption(", ".join(status["available"]))
        else:
            st.error("Ollama or a required model is unavailable.")

            if status.get("error"):
                st.code(status["error"])

            if status.get("missing"):
                st.write("Missing:", ", ".join(status["missing"]))

    st.divider()
    st.header("📤 Upload research papers")

    uploaded_files = st.file_uploader(
        "Choose PDF files from your computer",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can select one or several research papers.",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected.")

        if st.button(
            "Upload and Index PDFs",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Saving uploaded PDFs..."):
                saved, errors = save_uploaded_pdfs(uploaded_files)

            for filename in saved:
                st.success(f"Uploaded: {filename}")

            for error in errors:
                st.error(error)

            if saved:
                with st.spinner(
                    "Extracting papers and building the searchable index..."
                ):
                    try:
                        stats = rebuild_index()
                        st.success("Upload complete and index updated!")
                        st.json(stats)

                    except Exception as exc:
                        st.error(
                            "The PDFs were saved, but indexing failed. "
                            "Your saved PDFs have not been deleted. "
                            f"Error: {exc}"
                        )

    st.divider()
    st.header("🗂️ Index controls")

    if st.button("Build / Rebuild Index", use_container_width=True):
        with st.spinner("Building the index..."):
            try:
                stats = rebuild_index()
                st.success("Index built successfully.")
                st.json(stats)

            except Exception as exc:
                st.error(f"Index build failed: {exc}")

    if st.button("Reload Saved Index", use_container_width=True):
        ok, message = load_saved_index()

        if ok:
            st.success(message)
        else:
            st.warning(message)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# Research library
# --------------------------------------------------

pdf_files = sorted(
    path.name
    for path in PAPERS_DIR.glob("*.pdf")
    if path.is_file()
)

st.subheader("📁 Your research library")

if pdf_files:
    st.caption(f"{len(pdf_files)} PDF file(s) stored locally.")

    with st.expander("View uploaded and existing PDFs", expanded=True):
        for filename in pdf_files:
            st.write(f"📄 {filename}")
else:
    st.info(
        "Your paper library is empty. Upload a PDF using the sidebar."
    )


# --------------------------------------------------
# Index metrics
# --------------------------------------------------

index = st.session_state.index
chunks = st.session_state.chunks
metadata = st.session_state.metadata

metric1, metric2, metric3 = st.columns(3)

metric1.metric("PDF files", len(pdf_files))
metric2.metric("Text chunks", len(chunks))
metric3.metric(
    "Indexed vectors",
    index.ntotal if index is not None else 0,
)


# --------------------------------------------------
# Research chat
# --------------------------------------------------

if index is None:
    st.info(
        "Upload a PDF and choose 'Upload and Index PDFs', "
        "or build an index from your existing papers."
    )

else:
    indexed_files = metadata.get("pdf_files", []) if isinstance(metadata, dict) else []
    choices = ["All papers"] + sorted(indexed_files)

    selected_source = st.selectbox(
        "Which papers should the assistant search?",
        choices,
    )

    source_filter = (
        None if selected_source == "All papers" else selected_source
    )

    st.subheader("💬 Ask your research assistant")

    # Render previous messages, including their saved citation status.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                display_assistant_details(message)

    question = st.chat_input(
        "Ask about a paper's methods, results, or conclusions..."
    )

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(
                "Searching papers and generating an answer..."
            ):
                try:
                    result = answer_question(
                        question=question,
                        index=st.session_state.index,
                        chunks=st.session_state.chunks,
                        top_k=TOP_K,
                        source_filter=source_filter,
                    )

                    assistant_message = {
                        "role": "assistant",
                        "content": result.get("answer", "The answer pipeline returned no answer."),
                        "sources": result.get("sources", []),
                        "evidence": result.get("retrieved_chunks", []),
                        "citation_status": result.get(
                            "citation_status", "unknown"
                        ),
                    }

                    st.markdown(assistant_message["content"])
                    display_assistant_details(assistant_message)

                    # Save the answer and citation status for future reruns.
                    st.session_state.messages.append(assistant_message)

                except Exception as exc:
                    error_message = (
                        "I couldn't answer that question. "
                        f"Details: {exc}"
                    )

                    st.error(error_message)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "evidence": [],
                        "citation_status": "unknown",
                    })
