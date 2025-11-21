from __future__ import annotations
from typing import Iterable, List
from pathlib import Path
import re
import uuid

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .schemas import DocumentChunk
from .config import settings


SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".tsv", ".xlsx", ".html", ".htm", ".md", ".txt"}

DEBUG_INGESTION = True


def _debug(msg: str) :
    if DEBUG_INGESTION:
        print(f"[INGEST] {msg}", flush=True)


def discover_all_sources(data_dir: Path | None = None) :
    """Universal ingestion: return all supported files under data_dir."""
    data_dir = data_dir or settings.data_dir
    _debug(f"Scanning {data_dir} for supported files...")
    paths: List[Path] = []
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(path)
            if len(paths) % 5 == 0:
                _debug(f"  Found {len(paths)} files so far (latest: {path.name})")
    _debug(f"Total sources discovered: {len(paths)}")
    return paths


def _load_text_from_pdf(path: Path) :
    _debug(f"Loading PDF: {path}")
    reader = PdfReader(str(path))
    texts = []
    for page_num, page in enumerate(reader.pages, start=1):
        texts.append(page.extract_text() or "")
        if page_num % 10 == 0:
            _debug(f"  Parsed {page_num} pages in {path.name}")
    return "\n".join(texts)


def _load_text_from_csv(path: Path) :
    _debug(f"Loading CSV: {path}")
    df = pd.read_csv(path)
    return df.to_csv(index=False)


def _load_text_from_tsv(path: Path) :
    _debug(f"Loading TSV: {path}")
    df = pd.read_csv(path, sep="\t")
    return df.to_csv(index=False)


def _load_text_from_xlsx(path: Path) :
    _debug(f"Loading XLSX: {path}")
    df = pd.read_excel(path)
    return df.to_csv(index=False)


def _load_text_from_html(path: Path) :
    _debug(f"Loading HTML: {path}")
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    return soup.get_text(separator="\n")


def _load_text_from_md_or_txt(path: Path) :
    _debug(f"Loading text/markdown: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def load_file_to_text(path: Path) :
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_text_from_pdf(path)
    if ext == ".csv":
        return _load_text_from_csv(path)
    if ext == ".tsv":
        return _load_text_from_tsv(path)
    if ext == ".xlsx":
        return _load_text_from_xlsx(path)
    if ext in {".html", ".htm"}:
        return _load_text_from_html(path)
    if ext in {".md", ".txt"}:
        return _load_text_from_md_or_txt(path)
    raise ValueError(f"Unsupported extension: {ext}")


def load_file_preview(
    path: Path,
    max_chars: int = 1_500,
    max_pdf_pages: int = 3,
    max_table_rows: int = 40,
) :
    """
    Lightweight preview loader used for source selection.
    Reads a shallow slice of the file instead of the full content to keep
    discovery cheap.
    """
    ext = path.suffix.lower()
    text = ""
    try:
        if ext == ".pdf":
            reader = PdfReader(str(path))
            texts = []
            for idx, page in enumerate(reader.pages, start=1):
                texts.append(page.extract_text() or "")
                if idx >= max_pdf_pages:
                    break
            text = "\n".join(texts)
        elif ext == ".csv":
            df = pd.read_csv(path, nrows=max_table_rows)
            text = df.to_csv(index=False)
        elif ext == ".tsv":
            df = pd.read_csv(path, sep="\t", nrows=max_table_rows)
            text = df.to_csv(index=False)
        elif ext == ".xlsx":
            df = pd.read_excel(path, nrows=max_table_rows)
            text = df.to_csv(index=False)
        elif ext in {".html", ".htm"}:
            html = path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator="\n")
        elif ext in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
        else:
            _debug(f"[WARN] Unsupported preview extension for {path}")
    except Exception as e:
        _debug(f"[WARN] Failed to load preview for {path}: {e}")
        text = ""

    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def simple_chunk_text(
    text: str,
    source: str,
    max_tokens: int = 400,
    overlap: int = 50,
    max_chars: int = 2_000_000,
) :
    """Memory-aware chunking with debug instrumentation."""
    if len(text) > max_chars:
        _debug(f"Text too large ({len(text)} chars) for {source}; truncating to {max_chars}")
        text = text[:max_chars]

    chunks: List[DocumentChunk] = []
    buffer: List[str] = []
    start_word = 0

    for word in re.finditer(r"\S+", text):
        buffer.append(word.group(0))
        if len(buffer) >= max_tokens:
            end_word = start_word + len(buffer)
            chunk_text = " ".join(buffer)
            chunk_id = str(uuid.uuid4())
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    text=chunk_text,
                    source=source,
                    metadata={"start_word": start_word, "end_word": end_word},
                )
            )
            start_word = max(0, end_word - overlap)
            buffer = buffer[-overlap:] if overlap > 0 else []

    if buffer:
        end_word = start_word + len(buffer)
        chunk_text = " ".join(buffer)
        chunk_id = str(uuid.uuid4())
        chunks.append(
            DocumentChunk(
                id=chunk_id,
                text=chunk_text,
                source=source,
                metadata={"start_word": start_word, "end_word": end_word},
            )
        )

    _debug(f"Chunked {source} into {len(chunks)} chunks")
    return chunks


def ingest_sources_to_chunks(paths: Iterable[Path]) :
    all_chunks: List[DocumentChunk] = []
    path_list = list(paths)
    if not path_list:
        _debug("No paths provided to ingest.")
        return all_chunks

    _debug(f"Starting ingestion for {len(path_list)} files")
    for idx, p in enumerate(path_list, start=1):
        _debug(f"[{idx}/{len(path_list)}] Reading {p}")
        try:
            text = load_file_to_text(p)
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
            continue
        _debug(f"Loaded {len(text)} characters from {p}")
        chunks = simple_chunk_text(text, source=str(p))
        all_chunks.extend(chunks)
        _debug(f"Accumulated total chunks: {len(all_chunks)}")

    _debug(f"Ingestion complete. Total chunks: {len(all_chunks)}")
    return all_chunks
