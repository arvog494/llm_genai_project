from __future__ import annotations
from typing import List, Dict, Any
from textwrap import dedent
import json
import uuid
import time
import math
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from .llm import LLMClient
from .schemas import DocumentChunk, Entity, Relation
from .graph_store import GraphStore
from .vector_store import VectorStore
from .ingestion import (
    discover_all_sources,
    ingest_sources_to_chunks,
    load_file_preview,
)
from .config import settings


DEBUG_EXTRACTION = True
DEBUG_SELECTION = True


def _extraction_debug(msg: str) :
    if DEBUG_EXTRACTION:
        print(f"[GRAPH_BUILDER] {msg}", flush=True)


def _selection_debug(msg: str) :
    if DEBUG_SELECTION:
        print(f"[SOURCE_SELECTION] {msg}", flush=True)


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) :
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    denom = norm_a * norm_b
    if denom == 0:
        return 0.0
    return dot / denom


def _keyword_overlap(brief: str, candidate: str) :
    brief_tokens = {w for w in brief.lower().split() if len(w) > 3}
    if not brief_tokens:
        return 0.0
    candidate_tokens = {w for w in candidate.lower().split() if len(w) > 3}
    if not candidate_tokens:
        return 0.0
    matches = brief_tokens.intersection(candidate_tokens)
    return len(matches) / len(brief_tokens)


class SourceDiscoveryAgent:
    def __init__(self, llm: LLMClient):
        self.data_dir = settings.data_dir
        self.llm = llm

    def run(self) :
        """Backward-compatible discovery without filtering."""
        paths = discover_all_sources(self.data_dir)
        return [str(p) for p in paths]

    def select_relevant(
        self, brief: str, max_sources: int | None = None
    ) :
        """
        Discover all sources and keep only those relevant to the brief.
        Combines lightweight previews, keyword overlap, and embedding similarity.
        """
        brief = (brief or "").strip()
        paths = discover_all_sources(self.data_dir)
        if not paths:
            return []

        limit = max_sources or settings.max_sources_per_use_case
        query_embedding = None
        try:
            query_embedding = self.llm.embed_query(brief)
        except Exception as e:
            _selection_debug(f"Failed to embed brief, fallback to lexical match: {e}")

        scored: List[Dict[str, Any]] = []
        for path in paths:
            preview = load_file_preview(
                Path(path),
                max_chars=settings.source_preview_chars,
            )
            candidate_text = f"{Path(path).name}\n{preview}"
            overlap = _keyword_overlap(brief, candidate_text)
            embed_sim = 0.0
            if query_embedding:
                try:
                    doc_emb = self.llm.embed_texts([candidate_text])[0]
                    embed_sim = _cosine_similarity(query_embedding, doc_emb)
                except Exception as e:
                    _selection_debug(f"Embedding failed for {path}: {e}")

            score = 0.6 * embed_sim + 0.4 * overlap
            scored.append(
                {
                    "path": str(path),
                    "score": score,
                    "overlap": overlap,
                    "embedding_score": embed_sim,
                    "preview": preview[:400],
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        filtered = [
            s for s in scored if s["score"] >= settings.min_relevance_score
        ]
        if not filtered:
            filtered = scored

        selected = filtered[:limit] if limit else filtered

        _selection_debug(
            f"Selected {len(selected)}/{len(paths)} sources for brief '{brief}'"
        )
        for s in selected:
            _selection_debug(
                f"- {Path(s['path']).name}: score={s['score']:.2f} "
                f"(overlap={s['overlap']:.2f}, emb={s['embedding_score']:.2f})"
            )
        return selected


class IngestionAgent:
    def run(self, paths: List[str]) :
        return ingest_sources_to_chunks(Path(p) for p in paths)


class GraphBuilderAgent:
    """
    Uses LLM to extract entities and relations from chunks and fill GraphStore.
    """

    def __init__(
        self,
        llm: LLMClient,
        graph_store: GraphStore,
        max_concurrency: int = 4,
        batch_size: int = 8,
    ):
        self.llm = llm
        self.graph_store = graph_store
        self.max_concurrency = max(1, max_concurrency)
        self.batch_size = max(1, batch_size)
        self.debug_sample_limit = max(0, settings.extraction_debug_samples)
        self._debug_samples = 0

        system_prompt = dedent(
            """
            You are an information extraction agent. Given a text chunk, extract key entities and relations.
            Respond ONLY with valid JSON of the form:
            '''json
            {{
              "entities": [
                {{"name": "...", "type": "...", "properties": {{...}}}}, ...
              ],
              "relations": [
                {{"source": "<entity name>", "target": "<entity name>", "type": "...", "properties": {{...}}}}, ...
              ]
            }}
            '''
            """
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "{chunk}"),
            ]
        )
        self.chain = self.prompt | self.llm.chat

    def _parse_response(
        self, response_str: str
    ) :
        try:
            return json.loads(response_str)
        except json.JSONDecodeError:
            start = response_str.find("{")
            end = response_str.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(response_str[start : end + 1])
                except Exception:
                    pass
        return {"entities": [], "relations": []}

    def _batched(self, items: List[DocumentChunk]):
        for i in range(0, len(items), self.batch_size):
            yield items[i : i + self.batch_size]

    def _maybe_log_sample(
        self,
        chunk: DocumentChunk,
        raw_response: str,
        parsed: Dict[str, List[Dict[str, Any]]],
    ) :
        if not DEBUG_EXTRACTION:
            return
        if self.debug_sample_limit <= 0:
            return
        if self._debug_samples >= self.debug_sample_limit:
            return
        preview = chunk.text[:200].replace("\n", " ")
        if len(chunk.text) > 200:
            preview += "..."
        parsed_preview = json.dumps(parsed, ensure_ascii=False)[:500]
        _extraction_debug(
            "Sample extraction "
            f"(chunk_id={chunk.id}, source={chunk.source}):\n"
            f"  chunk_preview: {preview}\n"
            f"  raw_response: {raw_response}\n"
            f"  parsed: {parsed_preview}"
        )
        self._debug_samples += 1

    def _ensure_chunk_entities(
        self,
        chunk: DocumentChunk,
        source_cache: Dict[str, str],
    ) :
        """
        Ensure each chunk is represented in the graph even if extraction fails.
        Returns the chunk entity id.
        """
        source_key = chunk.source or "unknown_source"
        display_source = (
            Path(chunk.source).name if chunk.source else "Unknown source"
        )
        if not display_source:
            display_source = source_key

        source_id = source_cache.get(source_key)
        if source_id is None or source_id not in self.graph_store.graph:
            source_id = str(uuid.uuid4())
            source_cache[source_key] = source_id
            self.graph_store.upsert_entity(
                Entity(
                    id=source_id,
                    name=f"Source: {display_source}",
                    type="SourceDocument",
                    properties={
                        "path": source_key,
                    },
                )
            )

        chunk_entity_id = chunk.id
        chunk_exists = chunk_entity_id in self.graph_store.graph
        if not chunk_exists:
            start_word = chunk.metadata.get("start_word")
            end_word = chunk.metadata.get("end_word")
            if start_word is None:
                start_word = 0
            if end_word is None:
                end_word = start_word
            chunk_label = f"{display_source} chunk {start_word}-{end_word}"
            preview = chunk.text[:200]
            if len(chunk.text) > 200:
                preview += "..."
            self.graph_store.upsert_entity(
                Entity(
                    id=chunk_entity_id,
                    name=chunk_label,
                    type="DocumentChunk",
                    properties={
                        "chunk_id": chunk.id,
                        "source_file": source_key,
                        "start_word": start_word,
                        "end_word": end_word,
                        "preview": preview,
                    },
                )
            )
            relation = Relation(
                id=str(uuid.uuid4()),
                source=source_id,
                target=chunk_entity_id,
                type="contains_chunk",
                properties={
                    "source_file": source_key,
                    "chunk_id": chunk.id,
                },
            )
            self.graph_store.add_relation(relation)
        return chunk_entity_id

    def _process_parsed(
        self,
        chunk: DocumentChunk,
        parsed: Dict[str, List[Dict[str, Any]]],
        name_to_id: Dict[str, str],
        source_cache: Dict[str, str],
    ) :
        self._ensure_chunk_entities(chunk, source_cache)
        for ent in parsed.get("entities", []):
            name = ent.get("name") or ent.get("label")
            if not name:
                continue
            e_id = name_to_id.get(name)
            if e_id is None:
                e_id = str(uuid.uuid4())
                name_to_id[name] = e_id
            entity = Entity(
                id=e_id,
                name=name,
                type=ent.get("type", "Unknown"),
                properties={
                    **ent.get("properties", {}),
                    "source_chunk_id": chunk.id,
                    "source_file": chunk.source,
                },
            )
            self.graph_store.upsert_entity(entity)
        # Relations may contain single names or lists (model-dependent).
        # Normalize to lists and create a relation for each source/target pair.
        def _to_name_list(val):
            if val is None:
                return []
            if isinstance(val, list):
                # keep only simple scalar items
                out = []
                for v in val:
                    if isinstance(v, (str, int)):
                        out.append(str(v))
                return out
            if isinstance(val, (str, int)):
                return [str(val)]
            # unknown type, attempt to stringify
            try:
                return [str(val)]
            except Exception:
                return []

        for rel in parsed.get("relations", []):
            src_names = _to_name_list(rel.get("source"))
            tgt_names = _to_name_list(rel.get("target"))
            if not src_names or not tgt_names:
                continue

            for src_name in src_names:
                for tgt_name in tgt_names:
                    if not src_name or not tgt_name:
                        continue
                    if src_name not in name_to_id or tgt_name not in name_to_id:
                        # skip relations where entity names weren't extracted/seen
                        continue
                    relation = Relation(
                        id=str(uuid.uuid4()),
                        source=name_to_id[src_name],
                        target=name_to_id[tgt_name],
                        type=rel.get("type", "related_to"),
                        properties={
                            **rel.get("properties", {}),
                            "source_chunk_id": chunk.id,
                            "source_file": chunk.source,
                        },
                    )
                    self.graph_store.add_relation(relation)

    def run(self, chunks: List[DocumentChunk]) :
        if not chunks:
            return
        name_to_id: Dict[str, str] = {}
        source_cache: Dict[str, str] = {}
        self._debug_samples = 0
        total_chunks = len(chunks)
        _extraction_debug(
            f"Starting extraction on {total_chunks} chunks "
            f"(batch_size={self.batch_size}, max_concurrency={self.max_concurrency})"
        )
        for batch_index, batch in enumerate(self._batched(chunks), start=1):
            batch_start = time.perf_counter()
            start_idx = (batch_index - 1) * self.batch_size + 1
            end_idx = start_idx + len(batch) - 1
            _extraction_debug(
                f"Batch {batch_index}: indices {start_idx}-{end_idx} ({len(batch)} chunks)"
            )
            inputs = [{"chunk": chunk.text} for chunk in batch]
            responses = self.chain.batch(
                inputs,
                config={"max_concurrency": self.max_concurrency},
            )
            for chunk, resp in zip(batch, responses):
                content = getattr(resp, "content", resp)
                parsed = self._parse_response(content)
                self._maybe_log_sample(chunk, content, parsed)
                self._process_parsed(chunk, parsed, name_to_id, source_cache)
            batch_time = time.perf_counter() - batch_start
            _extraction_debug(
                f"Batch {batch_index} finished in {batch_time:.2f}s "
                f"(processed {min(end_idx, total_chunks)}/{total_chunks})"
            )
        _extraction_debug("Extraction complete.")


class EmbeddingAgent:
    def __init__(self, llm: LLMClient, vector_store: VectorStore):
        self.llm = llm
        self.vector_store = vector_store

    def run(self, chunks: List[DocumentChunk]) :
        self.vector_store.rebuild(chunks, self.llm)


class QAAgent:
    """
    Uses both graph neighborhood and vector store to answer questions.
    """

    def __init__(
        self,
        llm: LLMClient,
        graph_store: GraphStore,
        vector_store: VectorStore,
    ):
        self.llm = llm
        self.graph_store = graph_store
        self.vector_store = vector_store

        self.entity_extraction_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Extract a list of important entity names mentioned in the question. "
                    'Return JSON: {{"entities": ["..."]}}. '
                ),
                ("user", "{question}"),
            ]
        )

        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a GraphRAG question answering agent.\n"
                        "You receive:\n"
                        "1) graph_context: JSON with nodes and edges from a knowledge graph\n"
                        "2) graph_paths: possible connection paths between the mentioned entities\n"
                        "3) vector_context: a list of text chunks with metadata\n\n"
                        "Use BOTH sources to answer the user's question. "
                        "For citations, each vector_context item has a 'citation' string already formatted; cite as [citation] and do NOT expose raw chunk ids. "
                        "If graph_context is empty, be explicit that no graph evidence was available. "
                        "Also briefly describe which entities/relations were most relevant.\n"
                    ),
                ),
                (
                    "user",
                    "Question:\n{question}\n\n"
                    "Graph context:\n{graph_context}\n\n"
                    "Graph paths:\n{graph_paths}\n\n"
                    "Vector context:\n{vector_context}\n",
                ),
            ]
        )

    def _extract_entities_from_question(self, question: str) :
        chain = self.entity_extraction_prompt | self.llm.chat
        resp = chain.invoke({"question": question})
        try:
            data = json.loads(resp.content)
            ents = data.get("entities", [])
            return [e for e in ents if isinstance(e, str)]
        except Exception:
            return []

    def _format_paths(self, raw_paths: List[List[str]]) :
        formatted = []
        for ids in raw_paths:
            nodes = []
            for node_id in ids:
                ent = self.graph_store.get_entity(node_id)
                nodes.append(
                    {
                        "id": node_id,
                        "name": ent.name if ent else node_id,
                        "type": ent.type if ent else "",
                    }
                )
            formatted.append({"node_ids": ids, "nodes": nodes})
        return formatted

    def run(self, question: str) :
        # 1) Get candidate entities from question
        entity_names = self._extract_entities_from_question(question)
        entity_ids: List[str] = []
        for n in entity_names:
            entity_ids.extend(self.graph_store.find_entities_by_name(n))

        # 2) Get graph neighborhood or fallback overview
        if entity_ids:
            graph_context = self.graph_store.get_neighborhood(
                entity_ids, radius=1, max_nodes=50
            )
        else:
            graph_context = self.graph_store.get_overview_subgraph(max_nodes=20)

        if not graph_context.get("nodes"):
            graph_context = self.graph_store.get_overview_subgraph(max_nodes=20)

        graph_paths_ids: List[List[str]] = []
        if len(entity_ids) > 1:
            graph_paths_ids = self.graph_store.find_paths_between(
                [entity_ids[0]],
                entity_ids[1:],
                max_paths=settings.max_graph_paths_per_answer,
            )
        graph_paths = self._format_paths(graph_paths_ids)

        # 3) Vector store retrieval
        vector_docs = self.vector_store.similarity_search(
            question, self.llm, k=5
        )

        # 4) Ask LLM with combined context
        chain = self.answer_prompt | self.llm.chat
        resp = chain.invoke(
            {
                "question": question,
                "graph_context": json.dumps(graph_context, ensure_ascii=False),
                "graph_paths": json.dumps(graph_paths, ensure_ascii=False),
                "vector_context": json.dumps(vector_docs, ensure_ascii=False),
            }
        )

        # Prepare structured answer with citations & paths
        answer_text = resp.content
        citations = [
            {
                "chunk_id": d["id"],
                "source": d["metadata"].get("source"),
                "label": d.get("citation") or d["metadata"].get("source"),
                "span": {
                    "start_word": d["metadata"].get("start_word"),
                    "end_word": d["metadata"].get("end_word"),
                },
                "metadata": d["metadata"],
            }
            for d in vector_docs
        ]

        return {
            "answer": answer_text,
            "citations": citations,
            "graph_context": graph_context,
            "graph_paths": graph_paths,
        }
