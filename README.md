# LangGraph Corrective RAG (CRAG) Chatbot

A Streamlit chatbot backed by a [LangGraph](https://github.com/langchain-ai/langgraph) agent that combines general-purpose tool use (web search, stock prices, a calculator) with a **Corrective RAG (CRAG)** pipeline for answering questions about an uploaded PDF, plus a lightweight **Self-RAG** critique step that grades each PDF-grounded answer for factual support and usefulness.

## Features

-  **Multi-turn chat** with persistent, resumable threads (SQLite checkpointing via `SqliteSaver`).
-  **Per-thread PDF ingestion** — upload a PDF to a chat thread and it's chunked, embedded, and indexed in a dedicated FAISS vector store for that thread only.
-  **Corrective RAG (CRAG)** retrieval:
  - Retrieves top-k chunks from the thread's PDF.
  - An LLM grades each chunk's relevance to the question (0–1 score).
  - Based on the scores, picks a verdict:
    - **CORRECT** — at least one chunk is clearly relevant → use PDF chunks only.
    - **INCORRECT** — all chunks score low → discard them, rewrite the query, and search the web instead.
    - **AMBIGUOUS** — mixed signal → blend refined PDF chunks with web search results.
  - "Knowledge refinement": the combined context is decomposed into sentences, each sentence is filtered for relevance, and the surviving sentences are recomposed into the final context handed to the generator.
-  **Self-RAG critique**: after the assistant answers a PDF-grounded question, a second LLM call grades whether the answer is fully/partially/not supported by the refined context, plus a 1–5 utility score.
-  **General tools**: DuckDuckGo web search, stock price lookup (Alpha Vantage), and a basic calculator (add/sub/mul/div).
-  **Thread management UI**: create new chats, switch between past chats, and delete chats (with confirmation) from the sidebar.
-  **Transparency badges**: each PDF-grounded answer shows an expandable "Retrieval details" panel with the CRAG verdict, chunk relevance scores, internal vs. web result counts, the rewritten web query (if any), web sources, the refined context passed to the generator, and the Self-RAG support/utility grade.

## Architecture

```
START
  │
  ▼
chat_node ──(tool call requested)──► tools ──► capture_crag_meta ──┐
  │                                                                 │
  │◄────────────────────────────────────────────────────────────────┘
  │
  ├─(no tool call, PDF context was used)──► self_rag_grade ──► END
  │
  └─(no tool call, no PDF context)──► END
```

- **`chat_node`** — the main LLM node (Groq `llama-3.3-70b-versatile`), bound to all tools. Decides whether to answer directly or call a tool.
- **`tools`** — a `ToolNode` executing whichever tool the LLM requested (`rag_tool`, `search_tool`, `get_stock_price`, `calculator`).
- **`capture_crag_meta`** — after a tool call, pulls the `rag_tool` JSON payload (verdict, scores, refined context, sources, etc.) into graph state so the UI can render it.
- **`self_rag_grade`** — runs once per turn, only when PDF context (`crag_meta`) was used, and grades the final answer against that context.

State (`chatState`) tracks the running message list plus the latest `crag_meta` and `self_rag_meta`, which are cleared at the start of every new user turn.

## Project Structure

```
.
├── backend.py     # LangGraph graph, tools, CRAG/Self-RAG pipeline, checkpointing helpers
├── frontend.py         # Streamlit UI (chat, PDF upload, thread sidebar, retrieval-detail badges)
└── chatbot.db      # SQLite checkpoint database (created automatically on first run)
```

## Requirements

- Python 3.10+
- A [Groq](https://console.groq.com/) API key (for the LLM)
- An [Alpha Vantage](https://www.alphavantage.co/) API key (for stock price lookups)

### Python dependencies

```
streamlit
langgraph
langgraph-checkpoint-sqlite
langchain-groq
langchain-core
langchain-community
langchain-text-splitters
duckduckgo-search
faiss-cpu
pypdf
sentence-transformers
requests
python-dotenv
pydantic
```

Install with:

```bash
pip install streamlit langgraph langgraph-checkpoint-sqlite langchain-groq langchain-core langchain-community langchain-text-splitters duckduckgo-search faiss-cpu pypdf sentence-transformers requests python-dotenv pydantic
```

## Setup

1. Clone the repository and move into the project directory.
2. Create a `.env` file in the project root with your API keys:

   ```env
   API_KEY=your_groq_api_key
   STOCK_API_KEY=your_alpha_vantage_api_key
   ```

3. (Optional) Tune the CRAG thresholds by adding these to `.env`:

   ```env
   CRAG_UPPER_TH=0.7     # score above this on any chunk => verdict CORRECT
   CRAG_LOWER_TH=0.3     # scores below this on every chunk => verdict INCORRECT
   CRAG_TOP_K=4          # number of chunks retrieved from the PDF per query
   CRAG_WEB_RESULTS=4    # number of web results fetched for corrective/ambiguous cases
   ```

4. Install dependencies (see above).

## Running the App

```bash
streamlit run app.py
```

This starts the Streamlit UI (defaults to `http://localhost:8501`) and creates a local `chatbot.db` SQLite file to persist chat threads.

## Usage

1. **Start chatting** — type a question in the chat box. General questions are answered directly or routed to web search / calculator / stock tools as needed.
2. **Ask about a PDF** — upload a PDF from the sidebar. Once indexing completes, ask questions about it in the chat; the assistant will automatically call the CRAG-powered `rag_tool`.
3. **Inspect retrieval** — expand " Retrieval details (Corrective RAG)" under any PDF-grounded answer to see the CRAG verdict, chunk scores, refined context, web sources (if used), and the Self-RAG support/utility grade.
4. **Manage chats** — use "New Chat" to start a fresh thread, click a thread in the sidebar to reopen it, or use the 🗑️ button (with confirmation) to permanently delete a thread and its indexed PDF.

## Notes & Limitations

- PDF retrievers are stored **in memory** (`_THREAD_RETRIEVERS`), so uploaded documents are lost on process restart even though chat history persists in `chatbot.db`.
- Web search uses DuckDuckGo and has no API key requirement, but is subject to rate limiting/availability changes.
- All LLM-based grading steps (chunk relevance, sentence filtering, query rewriting, Self-RAG grading) fail open — if a grading call errors, the pipeline falls back to a neutral score or keeps the content rather than dropping it silently.

## Credits

Built with [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), [Groq](https://groq.com/), and [Streamlit](https://streamlit.io/). CRAG methodology based on Yan et al., *"Corrective Retrieval Augmented Generation"*.