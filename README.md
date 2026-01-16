GraphRAG Agentic Pipeline
=========================

This repository implements an end‑to‑end agentic retrieval pipeline that fuses a knowledge graph with a vector store. Given a short user brief (e.g., “evolution of the AI courses in ESILV”), the system automatically finds the most relevant data sources, ingests and normalizes them, extracts entities and relations to build a knowledge graph, embeds text chunks for vector search, and finally exposes a GraphRAG QA endpoint that answers questions with citations plus graph traversal paths.

Contents
--------

1. [Key capabilities](#key-capabilities)
2. [Architecture](#architecture)
3. [Project structure](#project-structure)
4. [Setup](#setup)
5. [Global index workflow](#global-index-workflow)
6. [Use-case sessions](#use-case-sessions)
7. [API reference](#api-reference)
8. [Streamlit UI](#streamlit-ui)
9. [Configuration](#configuration)
10. [Troubleshooting & tips](#troubleshooting--tips)

Key capabilities
----------------

- **Source discovery & selection** : scans `data/raw/` for PDFs, spreadsheets, HTML, Markdown, TXT, etc., then scores each file against the user brief using keyword overlap and embedding similarity.
- **Ingestion & normalization** : converts heterogeneous documents to plain text, chunks them with overlap, and tracks chunk positions for downstream citations.
- **Knowledge graph construction** : an LLM-powered graph builder extracts entities/relations per chunk, stores them in a NetworkX-based multigraph, and preserves chunk-to-source edges.
- **Vector store for RAG** : ChromaDB acts as the persistent vector store; embeddings are produced through the configurable Ollama embedding model.
- **GraphRAG QA** : answers combine graph neighborhoods, graph path summaries, and top-N vector chunks, ensuring responses cite concrete provenance.
- **Per-brief sessions** : each brief yields its own GraphStore + VectorStore pair and a dedicated QA agent; sessions persist graphs under `graph_sessions/`.
- **FastAPI + Streamlit** : the backend exposes REST endpoints for both global and per-use-case workflows, while the UI provides an interactive way to submit briefs, inspect selected sources, and ask questions.

Architecture
------------

```
                              +----------------+
                brief + data  | SourceDiscovery|--+
                              +----------------+  |
                                                  v
                        +-----------------+   +-----------+
                        |Ingestion & Chunk|-->|Document   |
                        +-----------------+   |Chunks     |
                                              +-----------+
                                                   |
                 +----------------------+          |
                 |GraphBuilderAgent     |          |
                 +----------------------+          |
                          |                        |
                          v                        v
              +----------------------+   +------------------+
              |GraphStore (NetworkX) |   |VectorStore (Chroma)|
              +----------------------+   +------------------+
                          \                        /
                           \                      /
                            +----+    +----------+
                                 v    v
                           +----------------+
                           |QAAgent (LLM)   |
                           +----------------+
                                   |
                            Answer + citations
```

- **LLM backend** : powered by `langchain-ollama` with separate chat, extraction, and embedding models defined in `src/graph_rag/config.py`.
- **Graph store** : `GraphStore` serializes to JSON (see `graph_store.json` for the global index and `graph_sessions/*.json` for per-use-case graphs).
- **Vector store** : `VectorStore` uses Chroma’s persistent client with per-session collection names.
- **Service orchestration** : `GraphRAGService` coordinates agent execution, manages sessions, and exposes build/QA methods.

Project structure
-----------------

```
src/
  api/server.py         FastAPI app (build + QA endpoints)
  graph_rag/
    agents.py           Source discovery, ingestion, graph builder, embeddings, QA agents
    config.py           Pydantic settings (paths, LLM models, selection thresholds)
    graph_store.py      NetworkX-based store with serialization helpers
    ingestion.py        File discovery/loading/chunking utilities
    llm.py              Ollama client wrapper
    service.py          High-level service with global index + per-use-case sessions
    vector_store.py     ChromaDB wrapper
  ui/streamlit_app.py   Streamlit front-end
  main_build_index.py   CLI entry point for the global index
data/raw/               Place your PDFs, CSVs, HTML, Markdown, TXT, etc.
graph_sessions/         Persisted per-use-case graph snapshots
```

Setup
-----

1. **Prerequisites**
   - Python 3.10+
   - [Ollama](https://ollama.com/) running locally with the models referenced in `config.py` (defaults: `llama3` for chat/extraction and `nomic-embed-text` for embeddings).
   - Optional: `make`, `uvicorn`, `streamlit` (already in `requirements.txt`).

2. **Install dependencies**

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate        # PowerShell
   pip install -r requirements.txt
   ```

3. **Environment variables**
   - Create `.env` (or set env vars) if you need to override defaults defined in `src/graph_rag/config.py`. Example:

     ```
     OLLAMA_BASE_URL=http://localhost:11434
     DATA_DIR=data/raw
     CHROMA_DB_DIR=chroma_db
     ```

4. **Data directory**
   - Drop all relevant files under `data/raw/`. Supported extensions: `.pdf`, `.csv`, `.tsv`, `.xlsx`, `.html`, `.htm`, `.md`, `.txt`.

Global index workflow
---------------------

The legacy “ingest everything” pipeline remains available and is helpful for benchmarking.

```bash
python -m src.main_build_index
```

This performs:

1. Discover every supported file under `data/raw/`.
2. Ingest + chunk the content.
3. Extract entities/relations to build a global `graph_store.json`.
4. Embed all chunks and rebuild the `graphrag_global` Chroma collection.

Once completed you can query the global QA endpoint:

```bash
cd src
uvicorn api.server:app --reload
# POST http://localhost:8000/qa {"question": "..."}
```

Use-case sessions
-----------------

Per-brief sessions provide a targeted graph/vector index built only from the most relevant documents.

1. **Build a session**

   ```bash
   curl -X POST http://localhost:8000/use_case/build \
        -H "Content-Type: application/json" \
        -d '{"brief": "evolution of the AI courses in ESILV", "max_sources": 8}'
   ```

   Response:

   ```json
   {
     "session_id": "542d9514-cfce-4025-97d0-a43c0b4b1df0",
     "brief": "evolution of the AI courses in ESILV",
     "num_sources": 5,
     "num_chunks": 48,
     "selected_sources": [...],
     "graph_path": "graph_sessions/542d9514-cfce-4025-97d0-a43c0b4b1df0.json"
   }
   ```

2. **Ask questions**

   ```bash
   curl -X POST http://localhost:8000/use_case/qa \
        -H "Content-Type: application/json" \
        -d '{"session_id": "542d9514-cfce-4025-97d0-a43c0b4b1df0", "question": "Which professors shaped the AI curriculum?"}'
   ```

   The answer payload includes:
   - `answer`: natural-language response.
   - `citations`: list of chunks with formatted citations (e.g., `path@start-end`).
   - `graph_context`: nodes/edges retrieved around question entities.
   - `graph_paths`: shortest paths showing how entities are connected.

Sessions stay in memory for the server process and their graphs are serialized to `graph_sessions/<session>.json` for inspection or later reuse.

API reference
-------------

| Endpoint            | Method | Body                                        | Description                                      |
|---------------------|--------|---------------------------------------------|--------------------------------------------------|
| `/health`           | GET    | —                                           | Simple readiness check.                          |
| `/build_index`      | POST   | —                                           | Builds the global graph/vector index.            |
| `/qa`               | POST   | `{"question": "..."}`                       | Global GraphRAG QA (requires global index).      |
| `/use_case/build`   | POST   | `{"brief": "...", "max_sources": 8}`        | Brief-driven session builder; returns session_id.|
| `/use_case/qa`      | POST   | `{"session_id": "...", "question": "..."}`  | Ask questions using a specific session.          |

- Responses conform to Pydantic models defined in `src/api/server.py`.
- Errors (e.g., missing brief, unknown session) return HTTP 400 with a helpful message.

Streamlit UI
------------

Launch the UI to interact with the pipeline without crafting curl commands:

```bash
streamlit run src/ui/streamlit_app.py
```

Features:

- Input a brief and pick the maximum number of sources to keep.
- Inspect the scored sources table (path, lexical overlap, embedding score).
- Ask questions against the freshly built session; see answer, citations, and graph paths.
- Optional section for building/querying the global corpus.

Configuration
-------------

`src/graph_rag/config.py` exposes the main knobs via environment variables:

- **Data & persistence**
  - `data_dir`: default `data/raw`.
  - `chroma_db_dir`: persistent directory for Chroma.
  - `graph_store_path`: `graph_store.json` for the global index.
  - `usecase_graph_dir` + `usecase_vector_prefix`: per-session storage locations.
- **LLM models**
  - `ollama_base_url`, `ollama_chat_model`, `ollama_extraction_model`, `ollama_embedding_model`.
- **Extraction tuning**
  - `extraction_max_concurrency`, `extraction_batch_size`, `extraction_debug_samples`.
- **Source selection**
  - `max_sources_per_use_case`, `min_relevance_score`, `source_preview_chars`.
- **QA**
  - `max_graph_paths_per_answer`, `max_neighbors_per_entity`.

Override them via `.env` or shell variables as needed.

Troubleshooting & tips
----------------------

- **Missing models** : ensure the Ollama models referenced in `config.py` are downloaded (`ollama run llama3`, etc.).
- **Empty responses** : verify `data/raw/` contains supported files and that chunking produced data (see console logs for `[INGEST]` output).
- **LangChain errors about `{ "entities" }`** : indicates template braces are not escaped; the current prompts already double-brace JSON literals, so keep that pattern in custom prompts.
- **Chroma persistence** : old session collections accumulate inside `chroma_db/`; remove unused ones if disk space becomes an issue.
- **Graph inspection** : open any JSON snapshot under `graph_sessions/` to visualize nodes/edges or import into tools like Gephi/NetworkX for deeper analysis.
- **Logging** : ingestion, selection, and extraction agents emit `[INGEST]`, `[SOURCE_SELECTION]`, and `[GRAPH_BUILDER]` logs respectively; keep console visible when diagnosing issues.
