# src/main_build_index.py

from __future__ import annotations
from graph_rag.service import GraphRAGService


def main():
    """
    Build a universal index:
    - Ingest all supported files under data/raw
    - Normalize and chunk
    - Extract entities/relations on all chunks
    - Build one global graph
    - Build one global vector store
    """
    service = GraphRAGService()
    result = service.build_index()

    print("Universal index built")
    print(f"- Number of sources   : {result['num_sources']}")
    print(f"- Number of chunks    : {result['num_chunks']}")
    print("- Sources:")
    for src in result["sources"]:
        print(f"  - {src}")


if __name__ == "__main__":
    main()
