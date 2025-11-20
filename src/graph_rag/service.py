# src/graph_rag/service.py

from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from pathlib import Path

from .llm import LLMClient
from .graph_store import GraphStore
from .vector_store import VectorStore
from .agents import (
    SourceDiscoveryAgent,
    IngestionAgent,
    GraphBuilderAgent,
    EmbeddingAgent,
    QAAgent,
)
from .config import settings


@dataclass
class UseCaseSession:
    id: str
    brief: str
    graph_store: GraphStore
    vector_store: VectorStore
    qa_agent: QAAgent
    sources: List[Dict[str, Any]]
    num_chunks: int
    graph_path: Path | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class GraphRAGService:
    def __init__(self):
        self.llm = LLMClient()
        self.extraction_llm = LLMClient(chat_model=settings.ollama_extraction_model)
        self.graph_store = self._load_graph_store()
        self.vector_store = VectorStore(collection_name="graphrag_global")
        self.source_agent = SourceDiscoveryAgent(self.llm)
        self.ingestion_agent = IngestionAgent()
        self.graph_builder_agent = GraphBuilderAgent(
            self.extraction_llm,
            self.graph_store,
            max_concurrency=settings.extraction_max_concurrency,
            batch_size=settings.extraction_batch_size,
        )
        self.embedding_agent = EmbeddingAgent(self.llm, self.vector_store)
        self.qa_agent = QAAgent(self.llm, self.graph_store, self.vector_store)
        self.index_built = self._detect_index_built()
        self.sessions: Dict[str, UseCaseSession] = {}

    def _load_graph_store(self) :
        path = settings.graph_store_path
        if path.exists():
            try:
                return GraphStore.load(path)
            except Exception as e:
                print(f"[WARN] Failed to load existing graph store: {e}")
        return GraphStore()

    def _detect_index_built(self) :
        graph_ready = not self.graph_store.is_empty()
        vector_ready = self.vector_store.has_data()
        # Consider the index usable if at least one store is populated.
        return graph_ready or vector_ready

    def build_index(self) :
        """
        Universal ingestion for the global index:
        - discover all supported files under data/raw
        - ingest and chunk everything
        - build one global graph
        - build one global vector store
        """
        sources = self.source_agent.run()

        chunks = self.ingestion_agent.run(sources)

        # Reset any previous in-memory graph before rebuild
        self.graph_store.graph.clear()

        self.graph_builder_agent.run(chunks)
        self.embedding_agent.run(chunks)

        try:
            self.graph_store.save(settings.graph_store_path)
        except Exception as e:
            print(f"[WARN] Failed to save graph store: {e}")

        self.index_built = self._detect_index_built()

        return {
            "num_sources": len(sources),
            "num_chunks": len(chunks),
            "sources": sources,
        }

    def build_use_case(
        self, brief: str, max_sources: int | None = None
    ) :
        """
        Build a graph/vector index scoped to a user brief.
        Returns a session id that is required for follow-up questions.
        """
        if not brief or not brief.strip():
            raise ValueError("brief cannot be empty")

        selected_sources = self.source_agent.select_relevant(
            brief, max_sources=max_sources
        )
        if not selected_sources:
            raise ValueError("no supported sources found for this brief")

        source_paths = [s["path"] for s in selected_sources]
        chunks = self.ingestion_agent.run(source_paths)

        session_id = str(uuid.uuid4())

        graph_store = GraphStore()
        graph_builder = GraphBuilderAgent(
            self.extraction_llm,
            graph_store,
            max_concurrency=settings.extraction_max_concurrency,
            batch_size=settings.extraction_batch_size,
        )
        graph_builder.run(chunks)

        vector_store = VectorStore(
            collection_name=f"{settings.usecase_vector_prefix}{session_id}"
        )
        embedding_agent = EmbeddingAgent(self.llm, vector_store)
        embedding_agent.run(chunks)

        graph_path = settings.usecase_graph_dir / f"{session_id}.json"
        try:
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            graph_store.save(graph_path)
        except Exception as e:
            print(f"[WARN] Failed to persist use-case graph: {e}")
            graph_path = None

        qa_agent = QAAgent(self.llm, graph_store, vector_store)
        session = UseCaseSession(
            id=session_id,
            brief=brief,
            graph_store=graph_store,
            vector_store=vector_store,
            qa_agent=qa_agent,
            sources=selected_sources,
            num_chunks=len(chunks),
            graph_path=graph_path,
        )
        self.sessions[session_id] = session

        return {
            "session_id": session_id,
            "brief": brief,
            "num_sources": len(selected_sources),
            "num_chunks": len(chunks),
            "selected_sources": selected_sources,
            "graph_path": str(graph_path) if graph_path else None,
        }

    def answer(self, question: str) :
        if not self.index_built:
            return {
                "answer": "Global index not built yet. Please build it first.",
                "citations": [],
                "graph_context": {},
                "graph_paths": [],
            }
        return self.qa_agent.run(question)

    def answer_use_case(self, session_id: str, question: str) :
        session = self.sessions.get(session_id)
        if not session:
            return {
                "answer": f"Unknown session_id {session_id}. Build the use-case index first.",
                "citations": [],
                "graph_context": {},
                "graph_paths": [],
            }
        return session.qa_agent.run(question)
