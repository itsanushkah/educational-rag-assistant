"""Gradio UI for the Educational RAG Assistant."""
from __future__ import annotations

import html
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

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.stone,
    font=(gr.themes.GoogleFont("Inter"), "sans-serif"),
    font_mono=(gr.themes.GoogleFont("JetBrains Mono"), "monospace"),
).set(
    body_background_fill="#f6f1e7",
    background_fill_primary="#fffdf8",
    background_fill_secondary="#f1ead9",
    border_color_primary="#ded2b3",
    block_background_fill="#fffdf8",
    block_border_color="#e6dcc2",
    block_radius="14px",
    block_shadow="0 1px 2px rgba(60, 45, 20, 0.06)",
    button_primary_background_fill="#a8432a",
    button_primary_background_fill_hover="#8f3721",
    button_primary_text_color="#fff8ee",
    button_secondary_background_fill="#fffdf8",
    button_secondary_border_color="#ded2b3",
    body_text_color="#332b20",
    body_text_color_subdued="#7a6d54",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&display=swap');

footer { display: none !important; }
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; }

#hero { padding: 28px 4px 8px 4px; }
#hero h1 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.1rem;
    letter-spacing: -0.01em;
    color: #2c2416;
    margin: 0;
}
#hero p {
    color: #7a6d54;
    font-size: 0.95rem;
    margin-top: 6px;
}

.panel-label {
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #2c2416;
    margin: 4px 0 10px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-label .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #a8432a; display: inline-block;
}

.source-chips { display: flex; flex-wrap: wrap; gap: 6px; min-height: 20px; }
.source-chip {
    background: #f1ead9;
    border: 1px solid #ded2b3;
    color: #4a3f2c;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
}
.empty-hint {
    color: #9c9078;
    font-size: 0.85rem;
    font-style: italic;
}

.citation-grid { display: flex; flex-direction: column; gap: 8px; }
.citation-card {
    border: 1px solid #e6dcc2;
    background: #fbf8f1;
    border-radius: 10px;
    padding: 10px 14px;
    display: flex;
    gap: 12px;
    align-items: baseline;
}
.citation-tag {
    font-family: 'JetBrains Mono', monospace;
    color: #a8432a;
    font-weight: 700;
    font-size: 0.85rem;
    flex-shrink: 0;
}
.citation-body { display: flex; flex-direction: column; gap: 2px; }
.citation-meta {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #8a7c60;
}
.citation-snippet {
    font-size: 0.85rem;
    color: #4a3f2c;
    line-height: 1.4;
}

#ask-row textarea { font-size: 0.95rem; }
"""

FORCE_LIGHT_JS = """
<script>
(function () {
    const forceLight = () => document.body.classList.remove('dark');
    forceLight();
    new MutationObserver(forceLight).observe(document.body, { attributes: true, attributeFilter: ['class'] });
})();
</script>
"""

HERO_HTML = """
<div id="hero">
  <h1>Educational RAG Assistant</h1>
  <p>Upload course materials on the left, then ask questions on the right —
  every answer is grounded in your materials and cites exactly where it came from.</p>
</div>
"""


def render_source_chips(sources: list[str]) -> str:
    if not sources:
        return '<div class="empty-hint">Nothing uploaded yet — add a file to get started.</div>'
    chips = "".join(f'<span class="source-chip">{html.escape(name)}</span>' for name in sources)
    return f'<div class="source-chips">{chips}</div>'


def render_citations(hits: list[dict]) -> str:
    if not hits:
        return ""
    cards = []
    for i, hit in enumerate(hits, start=1):
        snippet = html.escape(hit["text"][:160].strip())
        cards.append(
            f'<div class="citation-card">'
            f'<div class="citation-tag">[{i}]</div>'
            f'<div class="citation-body">'
            f'<div class="citation-meta">{html.escape(hit["source"])} · chunk {hit["chunk_index"]}</div>'
            f'<div class="citation-snippet">{snippet}…</div>'
            f"</div></div>"
        )
    return f'<div class="citation-grid">{"".join(cards)}</div>'


def upload_materials(files: list[str]) -> tuple[str, str]:
    if not files:
        return "No files selected.", render_source_chips(store.list_sources())

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

    status = f"Indexed {added_chunks} chunks from {', '.join(names)}."
    return status, render_source_chips(store.list_sources())


def clear_materials() -> tuple[str, str]:
    store.clear()
    return "Cleared all uploaded materials.", render_source_chips(store.list_sources())


def ask(question: str, history: list) -> tuple[list, str, str]:
    if not question.strip():
        return history, "", ""

    result = answer_question(store, question)
    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": result["answer"]},
    ]
    return history, "", render_citations(result["sources"])


with gr.Blocks(title="Educational RAG Assistant") as demo:
    gr.HTML(HERO_HTML)

    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=260):
            gr.HTML('<div class="panel-label"><span class="dot"></span>Library</div>')
            file_input = gr.Files(label="Add materials (PDF, TXT, MD)", file_types=[".pdf", ".txt", ".md"])
            with gr.Row():
                upload_button = gr.Button("Index", variant="primary", size="sm")
                clear_button = gr.Button("Clear", size="sm")
            upload_status = gr.Markdown()
            source_chips = gr.HTML(render_source_chips(store.list_sources()))

        with gr.Column(scale=3):
            gr.HTML('<div class="panel-label"><span class="dot"></span>Ask</div>')
            chatbot = gr.Chatbot(label=None, show_label=False, height=380)
            with gr.Row(elem_id="ask-row"):
                question_input = gr.Textbox(
                    placeholder="Ask about your course materials...", show_label=False, scale=5
                )
                ask_button = gr.Button("Ask", variant="primary", scale=1)
            gr.HTML('<div class="panel-label"><span class="dot"></span>Sources</div>')
            citation_panel = gr.HTML()

    upload_button.click(upload_materials, inputs=file_input, outputs=[upload_status, source_chips])
    clear_button.click(clear_materials, outputs=[upload_status, source_chips])

    ask_button.click(ask, inputs=[question_input, chatbot], outputs=[chatbot, question_input, citation_panel])
    question_input.submit(ask, inputs=[question_input, chatbot], outputs=[chatbot, question_input, citation_panel])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, head=FORCE_LIGHT_JS, footer_links=[])
