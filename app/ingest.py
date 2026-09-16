"""Load course material files and split them into overlapping text chunks."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(errors="ignore")


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_break = max(chunk.rfind("\n\n"), chunk.rfind(". "))
            if last_break > chunk_size * 0.5:
                end = start + last_break + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if c]


def ingest_file(path: Path) -> list[Chunk]:
    text = load_text(path)
    pieces = split_into_chunks(text)
    return [
        Chunk(id=str(uuid.uuid4()), text=piece, source=path.name, chunk_index=i)
        for i, piece in enumerate(pieces)
    ]
