"""Streamlit UI: upload documents, chat with them, inspect what each agent did.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Chat with your documents", page_icon="📄", layout="wide")

NODE_LABELS = {
    "understand_intent": "Orchestrator: understanding intent",
    "retrieve": "Retriever Agent: searching documents",
    "analyze": "Analyst Agent: analysing evidence",
    "answer": "Answer Agent: writing the response",
}


def get_system():
    if "rag" not in st.session_state:
        from rag_agents.system import RagSystem

        st.session_state.rag = RagSystem()
    return st.session_state.rag


def render_trace(result) -> None:
    with st.expander("How this answer was produced"):
        for n, r in enumerate(result.retrievals, start=1):
            st.markdown(f"**Retriever pass {n}** — query: `{r.standalone_query}`")
            for line in r.trace:
                st.caption(line)
        for n, trace in enumerate(result.analyst_trace, start=1):
            st.markdown(f"**Analyst pass {n}**")
            if not trace:
                st.caption("No tools needed.")
            for line in trace:
                st.caption(line)
        if result.analysis is not None:
            a = result.analysis
            st.markdown(f"**Evidence sufficient:** {'yes' if a.sufficient else 'no'} · feedback loops used: {result.feedback_loops}")
            if a.missing_information:
                st.caption("Missing: " + "; ".join(a.missing_information))
    if result.evidence:
        with st.expander(f"Evidence used ({len(result.evidence)} chunks)"):
            for i, e in enumerate(result.evidence, start=1):
                where = f"page {e.page}" if e.page is not None else (e.section or "document")
                st.markdown(f"**E{i}** · {e.source} · {where} · {e.chunk_type} · score {e.score:.2f}")
                st.text(e.content[:900])


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Documents")
    if not os.getenv("OPENAI_API_KEY"):
        key = st.text_input("OpenAI API key", type="password", help="Or set OPENAI_API_KEY in a .env file.")
        if key:
            os.environ["OPENAI_API_KEY"] = key
            st.rerun()
        st.info("Enter your OpenAI API key to start.")
        st.stop()

    rag = get_system()
    files = st.file_uploader("Upload PDF, Word or text files", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if st.button("Process uploaded files", disabled=not files, use_container_width=True):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for f in files:
                p = Path(tmp) / Path(f.name).name
                p.write_bytes(f.getbuffer())
                paths.append(p)
            with st.status("Loading, cleaning, chunking and embedding...", expanded=True) as status:
                for report in rag.ingest(paths):
                    if report.ok:
                        st.write(f"✅ {report.source}: {report.text_chunks} text + {report.table_chunks} table chunks")
                    else:
                        st.write(f"❌ {report.source}: {report.error}")
                    for w in report.warnings:
                        st.write(f"⚠️ {report.source}: {w}")
                status.update(label="Done", state="complete")

    docs = rag.documents()
    if docs:
        st.subheader("In the knowledge base")
        for name, info in docs.items():
            c1, c2 = st.columns([4, 1])
            c1.write(f"📄 {name}  \n`{info['chunks']} chunks`")
            if c2.button("🗑", key=f"del-{name}", help=f"Remove {name}"):
                rag.remove_document(name)
                st.rerun()
        if st.button("Clear everything", use_container_width=True):
            rag.clear()
            st.session_state.messages = []
            st.rerun()
    else:
        st.caption("No documents yet.")

# ---------------------------------------------------------------- chat
st.title("Chat with your documents")
st.caption("Retriever Agent finds evidence → Analyst Agent reasons and calculates → Answer Agent responds with citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("result") is not None:
            render_trace(m["result"])

if question := st.chat_input("Ask a question about your documents"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        final_state: dict = {}
        with st.status("Working...", expanded=False) as status:
            try:
                for node, update in rag.stream(question, history):
                    status.update(label=NODE_LABELS.get(node, node))
                    final_state.update(update)
                    if node == "analyze" and update.get("follow_up_queries"):
                        st.write("Analyst needs more evidence → asking the Retriever: " + "; ".join(update["follow_up_queries"]))
                status.update(label="Done", state="complete")
            except Exception as exc:
                status.update(label="Failed", state="error")
                st.error(f"{type(exc).__name__}: {exc}")
                st.stop()

        # rebuild the full result from the streamed updates (retrieve/analyze keep their accumulated state)
        from rag_agents.system import RagSystem

        result = RagSystem._to_result(final_state)
        st.markdown(result.text)
        if result.evidence or result.analysis:
            render_trace(result)
    st.session_state.messages.append({"role": "assistant", "content": result.text, "result": result})
