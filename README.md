# Educational RAG Assistant

A retrieval-augmented learning assistant that answers questions grounded in your
own uploaded course materials, with source citations for every answer.

## How it works

1. **Upload** PDF/TXT/MD course materials in the "Upload Materials" tab.
2. Each file is split into overlapping text chunks and embedded locally with
   `sentence-transformers` (`all-MiniLM-L6-v2`), then stored in a persistent
   Chroma vector database (`data/chroma/`).
3. **Ask** a question in the "Ask" tab. The top matching chunks are retrieved
   and passed to Claude, which is instructed to answer only from that context
   and cite which excerpt each claim came from.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then add your ANTHROPIC_API_KEY
```

## Run

```bash
python main.py
```

Then open the local URL Gradio prints (usually http://127.0.0.1:7860).

## Notes

- Uploaded files and the vector index are stored under `data/` and are
  git-ignored — they're local to your machine.
- To start fresh, use "Clear all materials" in the Upload tab, or delete
  `data/chroma/` and `data/uploads/`.
