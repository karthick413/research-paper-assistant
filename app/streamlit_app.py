
import streamlit as st

from rag_pipeline import (
    PAPERS,
    answer_question_with_sources,
)
from vector_store import load_chunks, build_index


st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Research Paper Assistant")
st.caption(
    "Ask questions about your research papers. Answers are grounded "
    "in retrieved PDF passages."
)


@st.cache_resource(show_spinner="Loading PDFs and building the search index...")
def initialize_rag():
    chunks = load_chunks()
    index = build_index(chunks)
    return chunks, index


def display_sources(sources):
    if not sources:
        st.info("No relevant passages were retrieved.")
        return

    with st.expander("Retrieved evidence", expanded=False):
        for number, item in enumerate(sources, start=1):
            st.markdown(
                f"**{number}. {item['source']} — page {item['page']}**"
            )
            st.caption(f"Similarity score: {item['score']:.3f}")
            st.write(item["text"])
            st.divider()


def display_diagnostics(diagnostics):
    invalid = diagnostics.get("invalid_citations", [])
    uncited = diagnostics.get("uncited_paragraphs", [])

    if invalid:
        st.warning(
            "Citations not matching the retrieved evidence: "
            + ", ".join(invalid)
        )

    if uncited:
        st.warning(
            f"{len(uncited)} paragraph(s) appear to lack page citations."
        )
        with st.expander("View uncited paragraphs"):
            for paragraph in uncited:
                st.write(f"- {paragraph}")

    if not invalid and not uncited:
        st.success(
            "Citation format checks passed. This does not guarantee "
            "that every claim is fully supported by its cited passage."
        )


def display_assistant_message(message, show_debug):
    st.markdown(message["content"])

    if show_debug:
        display_sources(message.get("sources", []))
        display_diagnostics(message.get("diagnostics", {}))


with st.sidebar:
    st.header("Settings")

    st.write("**Language model**")
    st.code("llama3.2:3b")

    st.write("**Embedding model**")
    st.code("nomic-embed-text")

    show_debug = st.checkbox(
        "Show retrieved evidence and diagnostics",
        value=False,
    )

    st.divider()
    st.subheader("Loaded papers")

    for filename, title in PAPERS.items():
        st.markdown(f"- **{title}**")
        st.caption(filename)

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    chunks, index = initialize_rag()
except Exception as exc:
    st.error(f"Could not initialize the RAG system: {exc}")
    st.info(
        "Check that Ollama is running, both models are installed, "
        "and the PDFs are inside data/papers."
    )
    st.stop()

st.caption(
    f"Ready to search {len(chunks)} text chunks from "
    f"{len(PAPERS)} configured papers."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            display_assistant_message(message, show_debug)
        else:
            st.markdown(message["content"])

question = st.chat_input("Ask about a research paper...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers and preparing an answer..."):
            try:
                answer, sources, diagnostics = answer_question_with_sources(
                    question=question,
                    index=index,
                    chunks=chunks,
                )

                message = {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "diagnostics": diagnostics,
                }

                st.session_state.messages.append(message)
                display_assistant_message(message, show_debug)

            except Exception as exc:
                st.error(f"Something went wrong while answering: {exc}")
                st.caption(
                    "Check the terminal for details and confirm that "
                    "Ollama is running."
                )
