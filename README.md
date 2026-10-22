# Research Paper Assistant

A Retrieval-Augmented Generation (RAG) application that lets users ask questions about research papers and receive answers grounded in retrieved PDF passages.

## Features

- Extracts text from research papers in PDF format.
- Splits documents into overlapping text chunks.
- Generates embeddings using Ollama.
- Stores and searches embeddings using FAISS.
- Generates answers with a local LLM.
- Provides page-level source citations.
- Includes a Streamlit chat interface.
- Displays retrieved evidence and citation diagnostics.

## Tech Stack

- Python
- Streamlit
- Ollama
- Llama 3.2 (3B)
- `nomic-embed-text`
- FAISS
- PyMuPDF
- NumPy

## Project Structure

```text
research-paper-assistant/
├── app/
│   ├── test_ollama.py
│   ├── test_faiss.py
│   ├── document_processor.py
│   ├── chunker.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   └── streamlit_app.py
├── data/
│   └── papers/
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.12
- Git
- Ollama

## Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the required Python packages.
4. Pull the required Ollama models.
5. Place your permitted research PDFs in `data/papers/`.
6. Run the Streamlit application.

### Install dependencies

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the packages used by the project:

```bash
pip install streamlit ollama faiss-cpu pymupdf numpy
```

### Download Ollama models

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Make sure Ollama is running before starting the application.

### Run the application

From the project root:

```bash
python -m streamlit run app/streamlit_app.py
```

## How It Works

1. **Document processing:** Extracts text and page metadata from PDFs.
2. **Chunking:** Splits page text into overlapping chunks.
3. **Embedding:** Converts chunks into vectors using `nomic-embed-text`.
4. **Indexing:** Stores normalized vectors in a FAISS inner-product index.
5. **Retrieval:** Finds passages most similar to the user's question.
6. **Generation:** Uses Llama 3.2 to answer based on the retrieved passages.
7. **Citations:** Checks answer citations against retrieved source pages.

## Current Scope

The application currently processes local PDF files and builds its FAISS index when initialized. Citation diagnostics check citation formatting and whether cited pages were retrieved; they do not guarantee that every claim is semantically supported.

## Future Improvements

- Persist the vector index between runs.
- Improve retrieval quality and paper filtering.
- Add automated tests.
- Strengthen citation verification.
- Add configurable chunking and retrieval settings.

## License

A license has not yet been selected for this project.