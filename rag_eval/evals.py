from __future__ import annotations

import asyncio
import time
import csv
import json
import os
import re
import sys
import threading
import httpx
from pathlib import Path

# Add the current directory to import sibling rag module when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from rag import default_rag_client

from openai import OpenAI

from ragas import Dataset

client = OpenAI(
    api_key="ollama",  # Ollama doesn't require a real key
    base_url="http://localhost:11434/v1"
)
rag_client = default_rag_client(llm_client=client, logdir="evals/logs")


def load_dataset(csv_path: Path | str | None = None):
    """Load the CSV dataset that already exists under evals/datasets.

    The CSV is expected to have at least: question, grading_note (or grading_notes).
    Optional passthrough columns like evidence and evidence_triples are preserved.
    """

    csv_path = (
        Path(csv_path)
        if csv_path
        else Path(__file__).parent / "evals" / "datasets" / "test_dataset.csv"
    )

    dataset = Dataset(
        name="test_dataset",
        backend="local/csv",
        root_dir="evals",
    )

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            question = (row.get("question") or "").strip()
            grading_notes = (row.get("grading_notes") or row.get("grading_note") or "").strip()
            evidence = (row.get("evidence") or "").strip()
            evidence_triples = (row.get("evidence_triples") or "").strip()

            dataset.append(
                {
                    "question": question,
                    "grading_notes": grading_notes,
                    "evidence": evidence,
                    "evidence_triples": evidence_triples,
                }
            )

    dataset.save()
    return dataset


EVAL_SYSTEM_PROMPT = (
        "You are a strict JSON grader for a RAG system. "
        "Return ONLY a JSON object (no prose, no markdown, no code fences). "
        "Use exactly the required keys and one of the allowed values for each key. "
        "Never invent new keys. If uncertain, choose the closest label (do not output 'unknown')."
)

EVAL_USER_PROMPT = """
Grade the answer against the question and grading notes.

Return EXACTLY this JSON schema (single JSON object):
{{
    "coverage": "high" | "partial" | "low",
    "grounding": "grounded" | "hallucinated",
    "faithfulness": "faithful" | "unfaithful",
    "answer_relevancy": "on-topic" | "partial" | "off-topic"
}}

Question: {question}
Grading Notes: {grading_notes}
Answer: {response}

JSON only.
"""


ALLOWED_VALUES: dict[str, set[str]] = {
    "coverage": {"high", "partial", "low"},
    "grounding": {"grounded", "hallucinated"},
    "faithfulness": {"faithful", "unfaithful"},
    "answer_relevancy": {"on-topic", "partial", "off-topic"},
}

DEFAULT_LABELS = {
    "coverage": "unknown",
    "grounding": "unknown",
    "faithfulness": "unknown",
    "answer_relevancy": "unknown",
}


_query_cache: dict[str, dict] = {}
_query_lock = threading.Lock()

_classification_cache: dict[tuple[str, str, str], dict] = {}
_classification_lock = threading.Lock()

_grade_semaphore: asyncio.Semaphore | None = None


def _extract_json_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _normalize_label(key: str, value: object) -> str | None:
    if value is None:
        return None
    v = str(value).strip().lower()
    # common normalizations
    v = v.replace("_", "-")
    if key == "answer_relevancy" and v in {"on topic", "ontopic"}:
        v = "on-topic"
    if v in ALLOWED_VALUES.get(key, set()):
        return v
    return None


def _parse_labels_from_text(text: str) -> dict:
    """Best-effort parsing to reduce 'unknown' when the model output is slightly malformed."""
    cleaned = (text or "").strip()
    if not cleaned:
        return {}

    # Try JSON extraction first.
    try:
        extracted = _extract_json_block(cleaned)
        parsed = json.loads(extracted)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Fallback: regex by key/value (handles missing quotes / extra text).
    out: dict[str, str] = {}
    for key, allowed in ALLOWED_VALUES.items():
        # Look for: key: value  (quotes optional)
        m = re.search(rf"\b{re.escape(key)}\b\s*[:=]\s*\"?([a-zA-Z_-]+)\"?", cleaned, re.IGNORECASE)
        if m:
            norm = _normalize_label(key, m.group(1))
            if norm:
                out[key] = norm
                continue

        # Last resort: if key is missing, try to infer by scanning allowed labels in text.
        # (We do this only when JSON parsing failed; it's better than returning unknown.)
        lowered = cleaned.lower().replace("_", "-")
        for candidate in allowed:
            if candidate in lowered:
                out[key] = candidate
                break

    return out


def _classify_response_sync(question: str, grading_notes: str, response_text: str) -> dict:
    if not response_text.strip():
        return {**DEFAULT_LABELS}

    # A lot of 'unknown' previously came from network hiccups or slightly malformed JSON.
    # We retry a few times and use tolerant parsing to keep labels usable.
    model_name = os.getenv("RAG_EVAL_MODEL", "Gpt-oss:20b")
    max_attempts = int(os.getenv("RAG_EVAL_GRADE_RETRIES", "3"))

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                temperature=0,
                max_tokens=120,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": EVAL_USER_PROMPT.format(
                            question=question,
                            grading_notes=grading_notes,
                            response=response_text,
                        ),
                    },
                ],
            )

            message = completion.choices[0].message.content if completion.choices else ""
            parsed = _parse_labels_from_text(message or "")

            labels = {**DEFAULT_LABELS}
            for key in labels.keys():
                norm = _normalize_label(key, parsed.get(key))
                if norm:
                    labels[key] = norm

            return labels
        except (httpx.ReadError, httpx.ConnectError) as e:
            last_error = e
            if attempt + 1 < max_attempts:
                time.sleep(1.0 * (attempt + 1))
                continue
        except Exception as e:
            last_error = e
            if attempt + 1 < max_attempts:
                time.sleep(0.5 * (attempt + 1))
                continue

    # If we couldn't grade after retries, keep unknown but include a hint for debugging.
    if last_error is not None:
        print(f"Grading failed after {max_attempts} attempts: {type(last_error).__name__}: {last_error}", flush=True)
    return {**DEFAULT_LABELS}


async def classify_response(question: str, grading_notes: str, response_text: str) -> dict:
    cache_key = (question, grading_notes, response_text)
    with _classification_lock:
        cached = _classification_cache.get(cache_key)
    if cached is not None:
        return cached

    global _grade_semaphore
    if _grade_semaphore is None:
        grade_concurrency = int(os.getenv("RAG_EVAL_GRADE_CONCURRENCY", "2"))
        _grade_semaphore = asyncio.Semaphore(max(1, grade_concurrency))

    loop = asyncio.get_running_loop()
    async with _grade_semaphore:
        labels = await loop.run_in_executor(
            None, _classify_response_sync, question, grading_notes, response_text
        )
    with _classification_lock:
        _classification_cache[cache_key] = labels
    return labels


def measure_conciseness(response_text: str) -> str:
    word_count = len(response_text.split())
    if word_count <= 12:
        return "terse"
    if word_count >= 120:
        return "verbose"
    return "concise"


async def query_with_cache(question: str) -> dict:
    with _query_lock:
        cached = _query_cache.get(question)
    if cached is not None:
        return cached

    loop = asyncio.get_running_loop()
    max_attempts = 3
    last_error = None
    
    for attempt in range(max_attempts):
        try:
            result = await loop.run_in_executor(None, rag_client.query, question)
            with _query_lock:
                _query_cache[question] = result
            return result
        except (ConnectionResetError, BrokenPipeError, EOFError, httpx.ReadError, httpx.ConnectError) as e:
            last_error = e
            if attempt + 1 < max_attempts:
                wait_time = (2 ** attempt) + (attempt * 2) # slightly more aggressive backoff
                print(f"Query attempt {attempt+1}/{max_attempts} failed for '{question[:50]}...': {type(e).__name__}. Retrying in {wait_time}s...", flush=True)
                await asyncio.sleep(wait_time)
            else:
                print(f"Query failed after {max_attempts} attempts for '{question[:50]}...'", flush=True)
        except Exception as e:
            # Catch-all for other unexpected errors to prevent total crash
            print(f"Unexpected error during query: {type(e).__name__}: {e}", flush=True)
            last_error = e
            break
    
    if isinstance(last_error, BaseException):
        raise last_error
    raise RuntimeError("Query failed but no exception was captured")


def dataset_to_rows(dataset: Dataset) -> list[dict]:
    try:
        rows = list(dataset)
        if rows:
            return rows
    except TypeError:
        pass

    data_attr = getattr(dataset, "data", None)
    if data_attr is not None:
        return list(data_attr)

    return []


async def run_experiment(row):
    t0 = time.monotonic()
    response = await query_with_cache(row["question"])
    t1 = time.monotonic()

    answer = response.get("answer", "")

    labels = await classify_response(
        question=row["question"],
        grading_notes=row["grading_notes"],
        response_text=answer,
    )
    t2 = time.monotonic()

    conciseness = measure_conciseness(answer)

    return {
        **row,
        "response": answer,
        "coverage": labels["coverage"],
        "grounding": labels["grounding"],
        "faithfulness": labels["faithfulness"],
        "answer_relevancy": labels["answer_relevancy"],
        "conciseness": conciseness,
        "query_latency_s": round(t1 - t0, 4),
        "grade_latency_s": round(t2 - t1, 4),
        "total_latency_s": round(t2 - t0, 4),
        "log_file": response.get("logs", " "),
    }


async def evaluate_dataset(rows, concurrency: int) -> list[dict]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    total = len(rows)
    results: list[dict] = []

    counters = {"started": 0, "completed": 0, "in_flight": 0}
    start_time = time.monotonic()

    async def worker(idx, row):
        async with semaphore:
            counters["started"] += 1
            counters["in_flight"] += 1
            q = (row.get("question") or "").replace("\n", " ")[:120]
            print(f"Starting {idx+1}/{total}: {q}", flush=True)
            res = await run_experiment(row)
            counters["in_flight"] -= 1
            counters["completed"] += 1
            elapsed = time.monotonic() - start_time
            avg = elapsed / counters["completed"] if counters["completed"] else 0
            eta = avg * (total - counters["completed"])
            print(
                f"Finished {counters['completed']}/{total} (idx={idx+1}) elapsed={int(elapsed)}s avg={avg:.1f}s ETA={int(eta)}s",
                flush=True,
            )
            return res

    tasks = [asyncio.create_task(worker(i, row)) for i, row in enumerate(rows)]

    async def heartbeat():
        while True:
            await asyncio.sleep(10)
            elapsed = time.monotonic() - start_time
            completed = counters["completed"]
            in_flight = counters["in_flight"]
            pct = (completed / total * 100) if total else 0
            print(
                f"Heartbeat: completed={completed}/{total} ({pct:.1f}%) in_flight={in_flight} elapsed={int(elapsed)}s",
                flush=True,
            )

    hb_task = asyncio.create_task(heartbeat())

    try:
        for future in asyncio.as_completed(tasks):
            result = await future
            results.append(result)
    finally:
        hb_task.cancel()

    return results


def save_results_csv(results: list[dict], dataset_name: str) -> Path:
    out_dir = Path(__file__).parent / "evals" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_name}.csv"

    if not results:
        out_path.touch()
        return out_path

    fieldnames = sorted({key for row in results for key in row.keys()})
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return out_path


async def main():
    dataset = load_dataset()
    rows = dataset_to_rows(dataset)

    total = len(rows)
    if total == 0:
        print("Dataset is empty; nothing to evaluate.")
        return

    concurrency = int(os.getenv("RAG_EVAL_CONCURRENCY", "6"))
    print(f"Loaded {total} rows. Evaluating with concurrency={concurrency}...")

    results = await evaluate_dataset(rows, concurrency)
    print("Evaluation completed successfully!")

    dataset_name = getattr(dataset, "name", "experiment_results")
    csv_path = save_results_csv(results, f"{dataset_name}_results")
    print(f"Results saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
