# Chat with your documents: 3-agent RAG system + FastAPI, voice, OCR and PDF reports

Implementation of the Project 3 brief: a document-question-answering system where three specialised agents
(Retriever, Analyst, Answer) work as a team, coordinated by an Orchestrator built with **LangGraph**.
LLM: **ChatOpenAI** (LangChain). Vector DB: **ChromaDB**. UI: **Streamlit** (plus a CLI).

```
UPLOAD ─► Loader ─► Parser ─► Cleaning ─► Chunking ─► Embeddings ─► Vector DB
                                                                        │
USER ─► Orchestrator (understand intent)                                │
             │                                                          ▼
             └──► Retriever Agent ──evidence──► Analyst Agent ──► Evidence assessment
                       ▲                              │                  │
                       └──── feedback loop ◄── NOT enough evidence       │ enough
                                                                         ▼
                                                                   Answer Agent ─► response + citations
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then put your OPENAI_API_KEY in it
```

## Run

```bash
streamlit run app.py                                   # web UI: upload files, chat, inspect each agent's work

python cli.py ingest sample_docs/*.txt                 # or use the CLI
python cli.py ask "What is the average CNN accuracy across the three papers?" --trace
python cli.py chat
```

`sample_docs/` holds three tiny synthetic "papers" (CNN accuracy 92 / 95 / 89) matching the example in the brief.
Try: *"What is the average CNN accuracy across the three papers?"*, then *"Which one is cheapest to run?"*
(a follow-up that relies on chat history), then *"Compare the three papers' methodology."*

## How it maps to the brief

| Brief | Where |
|---|---|
| Document pipeline: loader, parser, cleaning, chunking, embeddings, vector DB | `rag_agents/ingestion/`, `rag_agents/vectorstore.py` |
| **Retriever Agent**: Query Rewriter, Semantic Search, Keyword Search, Metadata Filter, Reranker, Context Selector | `rag_agents/agents/retriever.py`, `rag_agents/tools/retrieval.py` |
| **Analyst Agent**: Calculator, Table Extractor, Document Comparison, Data Analysis, Search/Retrieve More Evidence | `rag_agents/agents/analyst.py`, `rag_agents/tools/analysis.py` |
| **Answer Agent**: Citation Formatter, Source Formatter, Response Formatter | `rag_agents/agents/answer.py`, `rag_agents/tools/answer_tools.py` |
| Orchestrator, evidence assessment, feedback loop | `rag_agents/graph.py` (LangGraph `StateGraph`) |

### Details worth knowing

- **Retriever never answers.** It rewrites the question (using chat history), applies a metadata filter only when
  the user scopes the question ("in paper.pdf", "page 5", "the table"), runs semantic (Chroma, cosine) and keyword
  (BM25) search, fuses the rankings (reciprocal rank fusion), reranks with the LLM, and selects a compact,
  de-duplicated set of chunks while keeping document name and page for citations.
- **Analyst uses tools instead of guessing.** Arithmetic goes through a safe calculator (no `eval`); PDF/DOCX tables are
  extracted at ingestion as markdown table chunks and turned into rows/columns by the Table Extractor.
  It returns a structured report with a `sufficient` flag.
- **Feedback loop.** If `sufficient` is false, the Analyst's follow-up queries go back to the Retriever, new evidence is
  merged, and the Analyst runs again. Bounded by `MAX_FEEDBACK_LOOPS` (default 2); if a follow-up search finds nothing
  new the loop stops and the Answer Agent states what is missing instead of inventing it.
- **Citations.** The Answer Agent numbers sources by document and page, writes `[1]`-style markers, drops any marker
  that does not exist, and appends a **Sources** list (`file.pdf, page 5`). DOCX/TXT have no pages, so DOCX cites the
  section heading and TXT cites the file (a form feed in a TXT file counts as a page break).
- Re-uploading a file replaces its old chunks. Scanned PDFs without a text layer are reported with a warning (no OCR).

## Configuration (environment variables / `.env`)

`OPENAI_API_KEY`, `OPENAI_CHAT_MODEL` (default `gpt-4o-mini`), `OPENAI_EMBEDDING_MODEL` (`text-embedding-3-small`),
`CHROMA_DIR`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K_SEMANTIC`, `TOP_K_KEYWORD`, `RERANK_TOP_N`, `CONTEXT_CHUNKS`,
`MAX_FEEDBACK_LOOPS`, `MAX_ANALYST_TOOL_STEPS`. See `rag_agents/config.py`.

## Tests

```bash
python -m pytest -q
```

32 tests run fully offline (a scripted LLM and a hashing embedding model stand in for OpenAI). They cover the pipeline
(PDF pages and tables, DOCX sections, TXT), every Retriever/Analyst/Answer tool, and the LangGraph workflow including the
feedback loop, its bound, the stalled-search shortcut, small talk, and the empty knowledge base.
They do **not** call the real OpenAI API, so answer quality with `gpt-4o-mini` is untested until you run it with your key.


---

# Task 5: Agentic system (FastAPI + voice + OCR + Report Generator agent)

## Extra setup
```bash
pip install -r requirements.txt
# OCR engine (system package, not pip):  Windows: UB-Mannheim installer | macOS: brew install tesseract | Ubuntu: apt install tesseract-ocr
```

## Run
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
# UI: http://127.0.0.1:8000/     Swagger: http://127.0.0.1:8000/docs
```

## API
| Endpoint | Purpose |
|---|---|
| `GET /health` | status, document count, auth/OCR availability (public) |
| `POST /documents`, `GET /documents`, `DELETE /documents/{name}`, `DELETE /documents` | upload PDF/DOCX/TXT/MD **and images** (OCR-ed), list, remove |
| `POST /query` `{question, session_id?, include_trace?}` | ask; reuse `session_id` for follow-ups |
| `POST /voice/query` (audio, session_id?, language?) | microphone audio -> Whisper speech-to-text -> same pipeline -> answer |
| `POST /voice/transcribe` | speech-to-text only |
| `POST /images/ocr` | extract text from an image |
| `POST /images/ingest` | OCR + add to the knowledge base |
| `POST /images/query` | text in the image becomes the question |
| `POST /reports` `{session_id, instructions, title?, question?, last_n?}` | Report Generator agent builds a PDF; returns `download_url` |
| `GET /reports/{id}` | download the PDF |

Set `API_KEY` to require an `X-API-Key` header on everything except `/health` and the UI page.

## Report Generator agent
`rag_agents/agents/report.py` + `rag_agents/tools/report_design.py`. The agent receives the session's answers, analysis,
calculations and sources plus the user's design wishes, and drives 20 tools: `set_fonts`, `set_font_sizes`, `set_colors`,
`set_spacing`, `set_page_layout` (size, orientation, margins, 1-2 columns), `set_header_footer`, `add_title_block`,
`add_heading`, `add_paragraph` (bold/italic/underline/colour/font/size/alignment/highlight/indent/spacing), `add_bullets`,
`add_table`, `add_callout`, `add_bar_chart`, `add_markdown`, dividers, spacers, page breaks, `review_report`.
Invalid values (unknown font/colour...) return an error the agent corrects; low text/background contrast is flagged.
If the model produces nothing, a default professional layout is used. Example instruction:
*"Navy headings, Times body 11pt, justified, two columns, footer 'Internal', bar chart of the accuracies."*

## Voice / OCR notes
- Voice uses the browser mic (MediaRecorder; needs `localhost` or https) and OpenAI `whisper-1`.
- OCR uses Tesseract through `pytesseract` with grayscale/upscale/contrast preprocessing and a confidence score.
- Limits: Arabic/RTL text in PDF reports is not shaped (letters appear disconnected); scanned PDFs must be uploaded as images.

## Tests
`python -m pytest -q` (offline). OCR tests need the Tesseract binary; API tests need fastapi + httpx.
