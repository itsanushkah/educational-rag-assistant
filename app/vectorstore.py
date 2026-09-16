"""Chroma-backed vector store using a local sentence-transformers embedding model."""
from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from app.ingest import Chunk

PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
COLLECTION_NAME = "course_materials"

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


class VectorStore:
    def __init__(self) -> None:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(PERSIST_DIR))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=_embedding_fn
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
        )

    def query(self, question: str, n_results: int = 5) -> list[dict]:
        if self._collection.count() == 0:
            return []
        n_results = min(n_results, self._collection.count())
        result = self._collection.query(query_texts=[question], n_results=n_results)
        hits = []
        for text, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append({"text": text, "source": meta["source"], "chunk_index": meta["chunk_index"], "distance": dist})
        return hits

    def list_sources(self) -> list[str]:
        data = self._collection.get()
        return sorted({m["source"] for m in data["metadatas"]}) if data["metadatas"] else []

    def clear(self) -> None:
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=_embedding_fn
        )
