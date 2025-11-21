from __future__ import annotations
from typing import List
from chromadb import PersistentClient

from .schemas import DocumentChunk
from .config import settings
from .llm import LLMClient


class VectorStore:
    """
    Thin wrapper around ChromaDB persistent collection.
    """

    def __init__(self, collection_name: str = "graphrag"):
        self.client = PersistentClient(path=str(settings.chroma_db_dir))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def has_data(self) :
        try:
            return self.collection.count() > 0
        except Exception:
            return False

    def rebuild(
        self, chunks: List[DocumentChunk], llm_client: LLMClient
    ) :
        # Clear existing collection
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

        if not chunks:
            # No content to index; keep an empty collection
            return

        texts = [c.text for c in chunks]
        ids = [c.id for c in chunks]
        metadatas = [
            {"source": c.source, **c.metadata} for c in chunks
        ]
        embeddings = llm_client.embed_texts(texts)
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    @staticmethod
    def _format_citation(meta: dict) :
        source = meta.get("source", "unknown_source")
        start = meta.get("start_word")
        end = meta.get("end_word")
        if start is not None and end is not None:
            return f"{source}@{start}-{end}"
        return source

    def similarity_search(
        self, query: str, llm_client: LLMClient, k: int = 5
    ):
        try:
            if self.collection.count() == 0:
                return []
        except Exception:
            return []

        q_emb = llm_client.embed_query(query)
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
        )
        docs = res.get("documents", [[]])[0]
        ids = res.get("ids", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        out = []
        for doc, _id, meta in zip(docs, ids, metas):
            meta = meta or {}
            out.append(
                {
                    "id": _id,
                    "text": doc,
                    "metadata": meta,
                    "citation": self._format_citation(meta),
                }
            )
        return out
