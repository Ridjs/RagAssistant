# Personal Document RAG Engine

An end-to-end Retrieval-Augmented Generation (RAG) web app that lets you upload a PDF and ask natural-language questions about its content. Built with Streamlit, sentence-transformer embeddings, and Groq's LLM API.

## How it works

1. **Upload** a PDF document through the web interface
2. **Parse & chunk** — text is extracted page-by-page and split into overlapping character-based chunks
3. **Embed & index** — each chunk is embedded using `sentence-transformers` (`all-MiniLM-L6-v2`) and normalized for cosine similarity search
4. **Cache** — embeddings and chunks are saved to disk as compressed `.npz` files so re-indexing the same document is instant on future runs
5. **Retrieve** — when you ask a question, the top-k most similar chunks are retrieved via cosine similarity
6. **Generate** — retrieved chunks are passed as context to an LLM (via Groq's API), which answers strictly from that context

## Tech stack

| Layer | Tool |
|---|---|
| UI / app framework | [Streamlit](https://streamlit.io/) |
| PDF parsing | [PyMuPDF](https://pymupdf.readthedocs.io/) |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| Vector search | NumPy (cosine similarity via normalized dot product) |
| LLM inference | [Groq API](https://console.groq.com/) (`openai/gpt-oss-20b`) |

## Project structure

```
.
├── app.py                # Streamlit UI and app logic
├── document_parser.py    # PDF text extraction and chunking
├── vector_store.py        # Embedding, indexing, search, and disk caching
├── requirements.txt
└── .streamlit/
    └── secrets.toml       # API keys (not committed)
```

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

### 2. Add your Groq API key

Get a free API key from [console.groq.com](https://console.groq.com/), then create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-key-here"
```

This file is gitignored.

### 3. Run the app

```bash
streamlit run app.py
```

## Configuration

The sidebar exposes a few tunable parameters:

- **Chunk size** — number of characters per chunk (200–1000)
- **Chunk overlap** — overlap between consecutive chunks, to preserve context across boundaries
- **Top-K** — number of retrieved chunks passed to the LLM as context

## Deployment

The app is designed to run on [Streamlit Community Cloud](https://streamlit.io/cloud). Add `GROQ_API_KEY` under **App settings → Secrets** in the same TOML format as above — no code changes needed between local and deployed environments.

## Notes & limitations

- Answers are generated strictly from retrieved context — the model is instructed not to use outside knowledge
- Uses a shared free-tier API key, so response availability may be limited under heavy or concurrent use
