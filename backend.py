"""
LangGraph chatbot backend.

This module keeps the original tool-calling agent (web search, stock price,
calculator, PDF-RAG) but upgrades the RAG tool into a Corrective RAG (CRAG)
pipeline, and adds a lightweight Self-RAG grading step that critiques the
final answer against the retrieved context.

Nothing that existed before has been removed:
- calculator, get_stock_price, search_tool are unchanged
- PDF ingestion / per-thread FAISS retrievers are unchanged
- SqliteSaver checkpointing, thread listing, and thread metadata helpers
  are unchanged (with one addition: thread_latest_meta)
"""

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated, Literal, Dict, Any, Optional, List
from pydantic import BaseModel, Field
import operator
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import re
import json
import sqlite3
import requests
import tempfile

load_dotenv()

API_KEY = os.getenv("API_KEY")
STOCK_API_KEY = os.getenv("STOCK_API_KEY")

# -------------------
#  CRAG configuration
# -------------------
# Chunk-relevance thresholds (see: Yan et al., "Corrective Retrieval Augmented
# Generation"). A single chunk scoring above UPPER_TH is enough to trust the
# internal knowledge fully ("Correct"); if every chunk scores below LOWER_TH
# the internal knowledge is discarded entirely ("Incorrect"); anything in
# between blends internal + web knowledge ("Ambiguous").
CRAG_UPPER_TH = float(os.getenv("CRAG_UPPER_TH", 0.7))
CRAG_LOWER_TH = float(os.getenv("CRAG_LOWER_TH", 0.3))
CRAG_TOP_K = int(os.getenv("CRAG_TOP_K", 4))
CRAG_WEB_RESULTS = int(os.getenv("CRAG_WEB_RESULTS", 4))

model = ChatGroq(
    groq_api_key=API_KEY,
    model_name="llama-3.3-70b-versatile"
)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
# -------------------
#  PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": CRAG_TOP_K}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass

# ==========================================================================
# Tools
# ==========================================================================
search_tool = DuckDuckGoSearchRun(region="us-en")
# A second, structured search wrapper used internally by CRAG so that we can
# recover titles/links for citation instead of one opaque text blob.
_web_search_tool = DuckDuckGoSearchResults(output_format="list", num_results=CRAG_WEB_RESULTS)

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={STOCK_API_KEY}"
    r = requests.get(url)
    return r.json()

# --------------------------------------------------------------------------
# Corrective RAG (CRAG) internals
# --------------------------------------------------------------------------

class DocEvalScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Relevance of the chunk to the question, 0-1")
    reason: str = Field(description="One short sentence justifying the score")

class KeepOrDrop(BaseModel):
    keep: bool

class WebQuery(BaseModel):
    query: str

class SelfRagGrade(BaseModel):
    support: Literal["full", "partial", "none"] = Field(
        description="Is the answer fully, partially, or not supported by the given context?"
    )
    utility: int = Field(ge=1, le=5, description="How useful is the answer to the question, 1-5")
    explanation: str

_doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores. Also return a short reason.",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
)
_doc_eval_chain = _doc_eval_prompt | model.with_structured_output(DocEvalScore)

_filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the sentence directly helps answer the question.\n"
            "Use ONLY the sentence.",
        ),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ]
)
_filter_chain = _filter_prompt | model.with_structured_output(KeepOrDrop)

_rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6-14 words).\n"
            "- If the question implies recency (e.g., recent/latest/last week/last month), "
            "add a constraint like (last 30 days).\n"
            "- Do NOT answer the question.",
        ),
        ("human", "Question: {question}"),
    ]
)
_rewrite_chain = _rewrite_prompt | model.with_structured_output(WebQuery)

_grade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Self-RAG critic. Given the context that was used and the answer "
            "that was generated, judge whether the context supports the answer "
            "('full', 'partial', or 'none') and rate the answer's utility to the "
            "question on a 1-5 scale.",
        ),
        ("human", "Question: {question}\n\nContext used:\n{context}\n\nAnswer:\n{answer}"),
    ]
)
_grade_chain = _grade_prompt | model.with_structured_output(SelfRagGrade)


def _decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _eval_docs(question: str, docs: list):
    """Score each retrieved chunk and derive a CRAG verdict."""
    scores: List[float] = []
    good = []
    for d in docs:
        try:
            out = _doc_eval_chain.invoke({"question": question, "chunk": d.page_content})
            score = float(out.score)
        except Exception:
            score = 0.5  # fail-open with a neutral score if grading breaks
        scores.append(score)
        if score > CRAG_LOWER_TH:
            good.append(d)

    if scores and any(s > CRAG_UPPER_TH for s in scores):
        verdict = "CORRECT"
    elif scores and all(s < CRAG_LOWER_TH for s in scores):
        verdict = "INCORRECT"
    else:
        verdict = "AMBIGUOUS"
    return verdict, good, scores


def _web_search_docs(query: str) -> List[dict]:
    """Structured web search used for the corrective / ambiguous branches."""
    try:
        results = _web_search_tool.invoke(query)
    except Exception:
        return []

    docs: List[dict] = []
    if isinstance(results, str):
        # Some backends fall back to a plain text blob.
        if results.strip():
            docs.append({"content": results, "source_url": None, "title": "Web search"})
        return docs

    for r in (results or [])[:CRAG_WEB_RESULTS]:
        if not isinstance(r, dict):
            continue
        title = r.get("title", "")
        link = r.get("link") or r.get("url")
        snippet = r.get("snippet") or r.get("content", "")
        content = f"{title}: {snippet}".strip(": ") if title else snippet
        if content:
            docs.append({"content": content, "source_url": link, "title": title})
    return docs


def _refine(question: str, texts: List[str]) -> str:
    """Decompose -> filter -> recompose ("knowledge refinement" in the CRAG paper)."""
    context = "\n\n".join(t for t in texts if t).strip()
    if not context:
        return ""

    strips = _decompose_to_sentences(context)
    kept: List[str] = []
    for s in strips:
        try:
            if _filter_chain.invoke({"question": question, "sentence": s}).keep:
                kept.append(s)
        except Exception:
            kept.append(s)  # fail-open: don't silently drop content if grading errors

    refined = "\n".join(kept).strip()
    # If the filter over-pruned everything, fall back to the raw context
    # rather than handing the generator an empty string.
    return refined or context


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> str:
    """
    Corrective RAG (CRAG) retrieval over the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.

    Retrieves chunks from the PDF, grades each chunk's relevance, and:
    - if at least one chunk is clearly relevant ("Correct"), refines and uses
      only the internal chunks;
    - if every chunk is irrelevant ("Incorrect"), discards them and instead
      rewrites the query and searches the web;
    - otherwise ("Ambiguous"), blends refined internal chunks with web results.

    Returns a JSON string with keys: query, verdict, scores, refined_context,
    internal_chunks_used, web_results_used, web_query, web_sources, source_file.
    Base your final answer only on `refined_context`.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return json.dumps({
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        })

    docs = retriever.invoke(query)
    verdict, good_docs, scores = _eval_docs(query, docs)

    web_docs: List[dict] = []
    web_query = None
    if verdict in ("INCORRECT", "AMBIGUOUS"):
        try:
            web_query = _rewrite_chain.invoke({"question": query}).query
        except Exception:
            web_query = query
        web_docs = _web_search_docs(web_query)

    if verdict == "CORRECT":
        source_texts = [d.page_content for d in good_docs]
    elif verdict == "INCORRECT":
        source_texts = [d["content"] for d in web_docs]
    else:  # AMBIGUOUS
        source_texts = [d.page_content for d in good_docs] + [d["content"] for d in web_docs]

    refined_context = _refine(query, source_texts)

    result = {
        "query": query,
        "verdict": verdict,
        "scores": [round(s, 3) for s in scores],
        "refined_context": refined_context,
        "internal_chunks_used": len(good_docs),
        "web_results_used": len(web_docs),
        "web_query": web_query,
        "web_sources": [d.get("source_url") for d in web_docs if d.get("source_url")],
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }
    return json.dumps(result)


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = model.bind_tools(tools)

from langgraph.graph.message import add_messages


class chatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    crag_meta: Optional[dict]
    self_rag_meta: Optional[dict]


def chat_node(state: chatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    updates: dict = {}
    # A brand-new user turn starts with a fresh HumanMessage as the latest
    # message; clear any CRAG / Self-RAG metadata left over from a previous
    # turn so the UI doesn't show stale badges for turns that don't use RAG.
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        updates["crag_meta"] = None
        updates["self_rag_meta"] = None

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. The rag_tool runs Corrective RAG: it returns a JSON string "
            "with a `verdict` (CORRECT / AMBIGUOUS / INCORRECT) and a `refined_context` "
            "field. Base your answer only on `refined_context`, and mention when your "
            "answer relied partly or fully on web results rather than the document. "
            "You can also use the general web search, stock price, and calculator tools "
            "when helpful. If no document is available, ask the user to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    updates["messages"] = [response]
    return updates


def capture_crag_meta(state: chatState, config=None):
    """Runs right after the tools node; pulls rag_tool's JSON payload into state."""
    for m in reversed(state["messages"]):
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "rag_tool":
            try:
                data = json.loads(m.content)
            except Exception:
                return {}
            if "error" in data:
                return {}
            return {"crag_meta": data}
        if isinstance(m, HumanMessage):
            break
    return {}


def self_rag_grade(state: chatState, config=None):
    """Self-RAG style critique: grade the final answer against the used context."""
    crag_meta = state.get("crag_meta") or {}
    context = crag_meta.get("refined_context", "")

    last_ai = None
    last_human = None
    for m in reversed(state["messages"]):
        if last_ai is None and isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            last_ai = m
        elif isinstance(m, HumanMessage):
            last_human = m
            break

    if not last_ai or not context:
        return {}

    try:
        grade = _grade_chain.invoke({
            "question": last_human.content if last_human else "",
            "context": context,
            "answer": last_ai.content,
        })
        return {"self_rag_meta": grade.model_dump()}
    except Exception:
        return {}


def route_after_chat(state: chatState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    if state.get("crag_meta"):
        return "self_rag_grade"
    return END


tool_node = ToolNode(tools)

connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkPointer = SqliteSaver(connection)


graph = StateGraph(chatState)
# nodes
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)
graph.add_node('capture_crag_meta', capture_crag_meta)
graph.add_node('self_rag_grade', self_rag_grade)

# edges
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges(
    'chat_node',
    route_after_chat,
    {"tools": "tools", "self_rag_grade": "self_rag_grade", END: END},
)
graph.add_edge('tools', 'capture_crag_meta')
graph.add_edge('capture_crag_meta', 'chat_node')
graph.add_edge('self_rag_grade', END)

chatbot = graph.compile(checkpointer=checkPointer)

def retrieve_all_threads():
    all_threads = set()
    for cp in checkPointer.list(None):
        all_threads.add(cp.config["configurable"]["thread_id"])
    return list(all_threads)

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


def _delete_thread_raw(thread_id: str) -> None:
    """Fallback: directly delete a thread's rows from the checkpoint sqlite tables."""
    cur = connection.cursor()
    for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
        try:
            cur.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
        except sqlite3.OperationalError:
            # table doesn't exist in this langgraph-checkpoint-sqlite version
            pass
    connection.commit()


def delete_thread(thread_id: str) -> None:
    """
    Permanently delete a chat thread: its checkpointed conversation history
    and any in-memory PDF retriever/metadata built for it.
    """
    tid = str(thread_id)

    deleted_via_api = False
    if hasattr(checkPointer, "delete_thread"):
        try:
            checkPointer.delete_thread(tid)
            deleted_via_api = True
        except Exception:
            deleted_via_api = False

    if not deleted_via_api:
        _delete_thread_raw(tid)

    _THREAD_RETRIEVERS.pop(tid, None)
    _THREAD_METADATA.pop(tid, None)


def thread_latest_meta(thread_id: str) -> dict:
    """Returns the most recent CRAG verdict / Self-RAG grade for a thread, if any."""
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
    except Exception:
        return {"crag": None, "self_rag": None}
    values = state.values if state else {}
    return {
        "crag": values.get("crag_meta"),
        "self_rag": values.get("self_rag_meta"),
    }