"""Gradio UI for the Educational RAG Assistant."""
from __future__ import annotations

import shutil
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from app.ingest import ingest_file
from app.rag import answer_question
from app.vectorstore import VectorStore

UPLOAD_DIR = Path(__file__).resolve().parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

store = VectorStore()


def upload_materials(files: list[str]) -> str:
    if not files:
        return "No files selected."

    added_chunks = 0
    names = []
    for file_path in files:
        src = Path(file_path)
        dest = UPLOAD_DIR / src.name
        shutil.copy(src, dest)
        chunks = ingest_file(dest)
        store.add_chunks(chunks)
        added_chunks += len(chunks)
        names.append(src.name)

    sources = ", ".join(store.list_sources())
    return (
        f"Indexed {added_chunks} chunks from: {', '.join(names)}\n\n"
        f"All materials currently loaded: {sources}"
    )


def clear_materials() -> str:
    store.clear()
    return "Cleared all uploaded materials."


def ask(question: str, history: list) -> tuple[list, str]:
    if not question.strip():
        return history, ""

    result = answer_question(store, question)
    answer = result["answer"]

    if result["sources"]:
        citation_lines = [
            f"[{i}] {hit['source']} (chunk {hit['chunk_index']})"
            for i, hit in enumerate(result["sources"], start=1)
        ]
        answer += "\n\n**Sources**\n" + "\n".join(citation_lines)

    history = history + [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return history, ""


with gr.Blocks(title="Educational RAG Assistant") as demo:
    gr.Markdown("# Educational RAG Assistant")
    gr.Markdown(
        "Upload your course materials, then ask questions. Answers are grounded "
        "in your uploaded content and cite the source chunk they came from."
    )

    with gr.Tab("Upload Materials"):
        file_input = gr.Files(label="Course materials (PDF, TXT, MD)", file_types=[".pdf", ".txt", ".md"])
        upload_button = gr.Button("Upload & Index", variant="primary")
        clear_button = gr.Button("Clear all materials")
        upload_status = gr.Textbox(label="Status", interactive=False)

        upload_button.click(upload_materials, inputs=file_input, outputs=upload_status)
        clear_button.click(clear_materials, outputs=upload_status)

    with gr.Tab("Ask"):
        chatbot = gr.Chatbot(label="Conversation", height=450)
        question_input = gr.Textbox(label="Your question", placeholder="Ask about your course materials...")
        ask_button = gr.Button("Ask", variant="primary")

        ask_button.click(ask, inputs=[question_input, chatbot], outputs=[chatbot, question_input])
        question_input.submit(ask, inputs=[question_input, chatbot], outputs=[chatbot, question_input])

if __name__ == "__main__":
    demo.launch()
