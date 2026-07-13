import streamlit as st
from backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
    thread_latest_meta,
    delete_thread,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

st.set_page_config(page_title="LangGraph Chatbot", layout="centered")
st.title("A LangGraph Chatbot")

# ******************** Style (badges for CRAG / Self-RAG) ****************
st.markdown(
    """
    <style>
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-correct   { background: #d7f5df; color: #147a3d; }
    .badge-ambiguous { background: #fdf1cf; color: #92660b; }
    .badge-incorrect { background: #fbdcdc; color: #a3271f; }
    .badge-support-full    { background: #d7f5df; color: #147a3d; }
    .badge-support-partial { background: #fdf1cf; color: #92660b; }
    .badge-support-none    { background: #fbdcdc; color: #a3271f; }
    .badge-neutral { background: #e7e9ee; color: #3a3f4b; }
    </style>
    """,
    unsafe_allow_html=True,
)

_VERDICT_CLASS = {
    "CORRECT": "badge-correct",
    "AMBIGUOUS": "badge-ambiguous",
    "INCORRECT": "badge-incorrect",
}
_SUPPORT_CLASS = {
    "full": "badge-support-full",
    "partial": "badge-support-partial",
    "none": "badge-support-none",
}

# ******************** Utility functions ****************
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def remove_thread(thread_id):
    st.session_state['chat_threads'] = [
        t for t in st.session_state['chat_threads'] if str(t) != str(thread_id)
    ]

def load_convo(thread_id):
    return chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values.get('messages', [])

def render_crag_badge(crag_meta: dict):
    verdict = crag_meta.get("verdict", "")
    cls = _VERDICT_CLASS.get(verdict, "badge-neutral")
    st.markdown(
        f'<span class="badge {cls}">CRAG: {verdict.title() if verdict else "n/a"}</span>',
        unsafe_allow_html=True,
    )

def render_self_rag_badge(self_rag_meta: dict):
    support = self_rag_meta.get("support", "")
    utility = self_rag_meta.get("utility", "")
    cls = _SUPPORT_CLASS.get(support, "badge-neutral")
    st.markdown(
        f'<span class="badge {cls}">Self-RAG support: {support}</span>'
        f'<span class="badge badge-neutral">Utility: {utility}/5</span>',
        unsafe_allow_html=True,
    )

def render_retrieval_details(crag_meta: dict, self_rag_meta: dict | None):
    with st.expander("🔎 Retrieval details (Corrective RAG)"):
        render_crag_badge(crag_meta)
        if self_rag_meta:
            render_self_rag_badge(self_rag_meta)
        st.caption(f"Query: {crag_meta.get('query', '')}")
        cols = st.columns(2)
        cols[0].metric("Internal chunks used", crag_meta.get("internal_chunks_used", 0))
        cols[1].metric("Web results used", crag_meta.get("web_results_used", 0))
        scores = crag_meta.get("scores") or []
        if scores:
            st.write("Chunk relevance scores:", ", ".join(f"{s:.2f}" for s in scores))
        if crag_meta.get("web_query"):
            st.write(f"Rewritten web query: *{crag_meta['web_query']}*")
        web_sources = crag_meta.get("web_sources") or []
        if web_sources:
            st.write("Web sources:")
            for url in web_sources:
                st.markdown(f"- [{url}]({url})")
        refined = crag_meta.get("refined_context", "")
        if refined:
            st.text_area("Refined context passed to the generator", refined, height=160, disabled=True)
        if self_rag_meta and self_rag_meta.get("explanation"):
            st.caption(f"Self-RAG critique: {self_rag_meta['explanation']}")

# ******************** Session Setup *********************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "pending_delete" not in st.session_state:
    st.session_state["pending_delete"] = None

add_thread(st.session_state['thread_id'])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ******************* Sidebar UI *************************
st.sidebar.title("Start a new chat!!")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

with st.sidebar.expander("ℹ️ How retrieval works"):
    st.write(
        "Questions about the uploaded PDF go through **Corrective RAG**: "
        "retrieved chunks are graded for relevance, and if they aren't good "
        "enough the assistant automatically rewrites your question and "
        "searches the web instead of (or in addition to) the PDF. Every "
        "answer that used the document also gets a quick **Self-RAG** "
        "critique checking whether the answer is actually supported by the "
        "context it was given."
    )

st.sidebar.title("Your Chats")

if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for tid in threads:
        tid_key = str(tid)
        row = st.sidebar.container()
        col_open, col_del = row.columns([4, 1])
        with col_open:
            if st.button(str(tid), key=f"side-thread-{tid_key}", use_container_width=True):
                selected_thread = tid
        with col_del:
            if st.button("🗑️", key=f"del-thread-{tid_key}", help="Delete this chat"):
                st.session_state["pending_delete"] = tid_key
                st.rerun()

        if st.session_state["pending_delete"] == tid_key:
            row.warning("Delete this chat permanently?")
            c_confirm, c_cancel = row.columns(2)
            if c_confirm.button("Confirm", key=f"confirm-del-{tid_key}", use_container_width=True):
                delete_thread(tid_key)
                remove_thread(tid_key)
                st.session_state["ingested_docs"].pop(tid_key, None)
                st.session_state["pending_delete"] = None
                if tid_key == thread_key:
                    reset_chat()
                st.rerun()
            if c_cancel.button("Cancel", key=f"cancel-del-{tid_key}", use_container_width=True):
                st.session_state["pending_delete"] = None
                st.rerun()

# ******************* Main UI****************************
# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])
        if message['role'] == 'assistant' and message.get('crag_meta'):
            render_retrieval_details(message['crag_meta'], message.get('self_rag_meta'))

user_input = st.chat_input('Ask about a document or something else')

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    label = (
                        "🧠 Running Corrective RAG…"
                        if tool_name == "rag_tool"
                        else f"🔧 Using `{tool_name}` …"
                    )
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(label, expanded=True)
                    else:
                        status_holder["box"].update(label=label, state="running", expanded=True)

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

        meta = thread_latest_meta(thread_key)
        crag_meta = meta.get("crag")
        self_rag_meta = meta.get("self_rag")
        if crag_meta:
            render_retrieval_details(crag_meta, self_rag_meta)

    assistant_entry = {"role": "assistant", "content": ai_message}
    if crag_meta:
        assistant_entry["crag_meta"] = crag_meta
        assistant_entry["self_rag_meta"] = self_rag_meta
    st.session_state["message_history"].append(assistant_entry)

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_convo(selected_thread)

    temp_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            temp_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            temp_messages.append({"role": "assistant", "content": msg.content})
    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()