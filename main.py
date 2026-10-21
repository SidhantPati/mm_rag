import gradio as gr
from answer_generation import llm_response

# ---------------------------------------------------------------------------
# Chat handler — bridges Gradio's history format to answer_generation.py
# ---------------------------------------------------------------------------

def chat(user_message: str, history: list[dict]) -> tuple[list[dict], str]:
    """
    Parameters
    ----------
    user_message : str
        The latest user input.
    history : list[dict]
        Gradio messages list – each entry is {"role": "user"|"assistant", "content": str}.

    Returns
    -------
    history : list[dict]
        Updated conversation history (appended user + assistant turn).
    sources_text : str
        Retrieved source chunks displayed in the Sources accordion.
    """
    if not user_message.strip():
        return history, ""

    # Build chat_history expected by llm_response (exclude the current turn)
    chat_history = [{"role": m["role"], "content": m["content"]} for m in history]

    # Call the RAG pipeline
    response_text, retrieved_docs = llm_response(user_message, chat_history)

    # Append turns to history
    history = history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": response_text},
    ]

    # Format retrieved source chunks for the accordion panel
    if retrieved_docs:
        sources_parts = []
        for i, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "unknown")
            snippet = doc.page_content.strip().replace("\n", " ")
            sources_parts.append(f"**[{i}] {source}**\n> {snippet}")
        sources_text = "\n\n---\n\n".join(sources_parts)
    else:
        sources_text = "_No sources retrieved._"

    return history, sources_text


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
)

CSS = """
/* ── Global ──────────────────────────────────────────────── */
body { background: #0f0e17; }

/* ── Header ──────────────────────────────────────────────── */
#header {
    text-align: center;
    padding: 2rem 1rem 1rem;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    margin-bottom: 1.25rem;
    border: 1px solid rgba(139,92,246,0.25);
}
#header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.35rem;
}
#header p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
}

/* ── Chatbot ─────────────────────────────────────────────── */
#chatbot {
    border-radius: 14px;
    border: 1px solid rgba(139,92,246,0.2);
    background: rgba(15,14,23,0.85);
    backdrop-filter: blur(10px);
}

/* ── Input row ───────────────────────────────────────────── */
#input-row { gap: 0.5rem; align-items: flex-end; }
#msg-box textarea { border-radius: 10px !important; }
#send-btn {
    border-radius: 10px !important;
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    font-weight: 600 !important;
    height: 48px;
    min-width: 90px;
}
#send-btn:hover { opacity: 0.88 !important; }
#clear-btn {
    border-radius: 10px !important;
    height: 48px;
    min-width: 80px;
}

/* ── Sources accordion ───────────────────────────────────── */
#sources-panel {
    border-radius: 12px;
    border: 1px solid rgba(139,92,246,0.2);
    background: rgba(15,14,23,0.7);
    margin-top: 0.75rem;
}
"""

with gr.Blocks(theme=THEME, css=CSS, title="Insurellm Assistant") as demo:

    # ── Header ────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="header">
        <h1>🛡️ Insurellm Assistant</h1>
        <p>Ask me anything about Insurellm — powered by RAG + Gemini</p>
    </div>
    """)

    # ── Conversation state ────────────────────────────────────────────────
    history_state = gr.State([])   # list[dict] — source of truth

    # ── Chat window ───────────────────────────────────────────────────────
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        height=520,
        show_label=False,
        avatar_images=(
            None,                          # user — default avatar
            "https://api.dicebear.com/7.x/bottts-neutral/svg?seed=insurellm",
        ),
        layout="bubble",          # replaces deprecated bubble_full_width=False
        render_markdown=True,
    )

    # ── Input row ─────────────────────────────────────────────────────────
    with gr.Row(elem_id="input-row"):
        msg_box = gr.Textbox(
            elem_id="msg-box",
            placeholder="Ask about Insurellm…",
            show_label=False,
            scale=8,
            lines=1,
            max_lines=5,
            autofocus=True,
            submit_btn=False,
        )
        send_btn = gr.Button("Send ➤", elem_id="send-btn", scale=1, variant="primary")
        clear_btn = gr.Button("🗑 Clear", elem_id="clear-btn", scale=1, variant="secondary")

    # ── Retrieved sources ─────────────────────────────────────────────────
    with gr.Accordion("📚 Retrieved Sources", open=False, elem_id="sources-panel"):
        sources_box = gr.Markdown(
            value="_Sources will appear here after your first message._",
            label="",
        )

    # ── Event wiring ──────────────────────────────────────────────────────

    def _submit(user_msg, history):
        """Run the RAG pipeline and update UI components."""
        if not user_msg.strip():
            return history, history, "", user_msg
        new_history, sources_text = chat(user_msg, history)
        return new_history, new_history, sources_text, ""   # chatbot, state, sources, clear textbox

    def _clear():
        """Reset conversation."""
        return [], [], "_Sources will appear here after your first message._"

    # Trigger on button click
    send_btn.click(
        fn=_submit,
        inputs=[msg_box, history_state],
        outputs=[chatbot, history_state, sources_box, msg_box],
    )

    # Trigger on Enter key
    msg_box.submit(
        fn=_submit,
        inputs=[msg_box, history_state],
        outputs=[chatbot, history_state, sources_box, msg_box],
    )

    # Clear button
    clear_btn.click(
        fn=_clear,
        inputs=[],
        outputs=[chatbot, history_state, sources_box],
    )

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(inbrowser=True)