"""Retrieval-augmented answer generation, grounded in retrieved chunks with citations."""
from __future__ import annotations

import os

from anthropic import Anthropic

from app.vectorstore import VectorStore

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = """You are a study assistant that answers questions using ONLY the \
provided course material excerpts. Follow these rules:

1. Base your answer strictly on the excerpts below. Do not use outside knowledge.
2. If the excerpts don't contain enough information to answer, say so plainly \
instead of guessing.
3. After each claim, cite the excerpt it came from using its bracketed tag, e.g. [1].
4. Keep the answer focused and student-friendly."""


def _format_context(hits: list[dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] (source: {hit['source']}, chunk {hit['chunk_index']})\n{hit['text']}")
    return "\n\n".join(blocks)


def answer_question(store: VectorStore, question: str, n_results: int = 5) -> dict:
    hits = store.query(question, n_results=n_results)
    if not hits:
        return {
            "answer": "No course materials have been uploaded yet, so I can't answer this from source content.",
            "sources": [],
        }

    context = _format_context(hits)
    client = Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Course material excerpts:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    answer_text = "".join(block.text for block in response.content if block.type == "text")
    return {"answer": answer_text, "sources": hits}
