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
import random
import hashlib
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
    "You are a grader for a RAG system. "
    "Evaluate the answer based on the provided question and grading notes. "
    "Provide a rating for: coverage, grounding, faithfulness, and answer_relevancy. "
    "Use the allowed values only."
)

EVAL_USER_PROMPT = """
Question: {question}
Grading Notes: {grading_notes}
Answer: {response}

Classify the answer using these exact keys and values:
- coverage: high, partial, low
- grounding: grounded, hallucinated
- faithfulness: faithful, unfaithful
- answer_relevancy: on-topic, partial, off-topic

Output Format:
coverage: <value>
grounding: <value>
faithfulness: <value>
answer_relevancy: <value>
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

    # Truncate very long responses to avoid context window issues
    if len(response_text) > 6000:
        response_text = response_text[:6000] + "...(truncated)"

    # A lot of 'unknown' previously came from network hiccups or slightly malformed JSON.
    # We retry a few times and use tolerant parsing to keep labels usable.
    model_name = os.getenv("RAG_EVAL_MODEL", "gpt-oss:20b")
    max_attempts = int(os.getenv("RAG_EVAL_GRADE_RETRIES", "3"))

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                temperature=0.1,  # Slight temp to avoid deterministic lockups
                max_tokens=256,
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
            content = completion.choices[0].message.content or ""
            parsed = _parse_labels_from_text(content)
            
            # Check if we got at least some valid labels
            valid_count = sum(1 for k in DEFAULT_LABELS if k in parsed)
            
            if valid_count > 0:
                return {k: parsed.get(k, "unknown") for k in DEFAULT_LABELS}
            
            # If we got nothing useful, raise an error to trigger retry
            if not content:
                msg = "Model returned empty response (system likely overloaded)"
            else:
                msg = f"Failed to parse output: {content[:100]!r}..."
            
            raise ValueError(msg)

        except (httpx.ReadError, httpx.ConnectError) as e:
            last_error = e
            # Network hiccups usually recover quickly.
            base_wait = 2.0 * (attempt + 1)
            wait = min(20.0, base_wait + random.uniform(0.0, 0.8))
            print(
                f"Network error (attempt {attempt+1}): {e}. Retrying in {wait:.1f}s...",
                file=sys.stderr,
            )
            if attempt + 1 < max_attempts:
                time.sleep(wait)
                continue
        except Exception as e:
            last_error = e
            # Empty responses are often a fast-fail caused by VRAM/KV-cache pressure.
            # Waiting only ~2s can be too short, so we back off more aggressively.
            msg = str(e).lower()
            if "empty response" in msg or "overloaded" in msg:
                base_wait = 8.0 * (attempt + 1)
                wait = min(60.0, base_wait + random.uniform(0.0, 2.0))
            else:
                base_wait = 2.0 * (attempt + 1)
                wait = min(20.0, base_wait + random.uniform(0.0, 0.8))

            print(
                f"Grading error (attempt {attempt+1}): {e}. Retrying in {wait:.1f}s...",
                file=sys.stderr,
            )
            if attempt + 1 < max_attempts:
                time.sleep(wait)
                continue

    # If we couldn't grade after retries, keep unknown but include a hint for debugging.
    if last_error is not None:
        print(f"Grading failed after {max_attempts} attempts: {type(last_error).__name__}: {last_error}", flush=True)
    return {**DEFAULT_LABELS}


async def classify_response(question: str, grading_notes: str, response_text: str) -> dict:
    # Avoid using the entire response text in the cache key (can be very large).
    response_fp = hashlib.sha1(response_text.encode("utf-8"), usedforsecurity=False).hexdigest()
    cache_key = (question, grading_notes, response_fp)
    with _classification_lock:
        cached = _classification_cache.get(cache_key)
    if cached is not None:
        return cached

    global _grade_semaphore
    if _grade_semaphore is None:
        # Default to 1 to prevent overloading local models during grading
        grade_concurrency = int(os.getenv("RAG_EVAL_GRADE_CONCURRENCY", "1"))
        _grade_semaphore = asyncio.Semaphore(max(1, grade_concurrency))

    loop = asyncio.get_running_loop()
    async with _grade_semaphore:
        # Small sleep to let GPU cool down between heavy tasks
        await asyncio.sleep(0.5)
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

            # Trim the cached payload aggressively to avoid ballooning memory over long runs.
            minimal = {
                "answer": (result.get("answer") if isinstance(result, dict) else "") or "",
                "logs": (result.get("logs") if isinstance(result, dict) else "") or " ",
            }
            with _query_lock:
                _query_cache[question] = minimal
            return minimal
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


async def run_query_only(row: dict) -> dict:
    t0 = time.monotonic()
    response = await query_with_cache(row["question"])
    t1 = time.monotonic()

    answer = response.get("answer", "")
    conciseness = measure_conciseness(answer)

    return {
        **row,
        "response": answer,
        "conciseness": conciseness,
        "query_latency_s": round(t1 - t0, 4),
        "log_file": response.get("logs", " "),
    }


async def run_grade_only(partial: dict) -> dict:
    t0 = time.monotonic()
    labels = await classify_response(
        question=partial.get("question", ""),
        grading_notes=partial.get("grading_notes", ""),
        response_text=partial.get("response", ""),
    )
    t1 = time.monotonic()

    grade_latency_s = round(t1 - t0, 4)
    query_latency_s = float(partial.get("query_latency_s") or 0.0)

    return {
        **partial,
        "coverage": labels["coverage"],
        "grounding": labels["grounding"],
        "faithfulness": labels["faithfulness"],
        "answer_relevancy": labels["answer_relevancy"],
        "grade_latency_s": grade_latency_s,
        # In two-phase mode, "total" is query+grade (not wall-clock across phases).
        "total_latency_s": round(query_latency_s + grade_latency_s, 4),
    }


def load_results_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


async def evaluate_queries(rows: list[dict], concurrency: int) -> list[dict]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    total = len(rows)
    print(f"Phase 1/2: querying answers (concurrency={concurrency})", flush=True)

    query_results: list[dict] = []
    q_counters = {"started": 0, "completed": 0, "in_flight": 0, "query_latency_sum": 0.0}
    q_start = time.monotonic()

    async def query_worker(idx, row):
        async with semaphore:
            q_counters["started"] += 1
            q_counters["in_flight"] += 1
            q = (row.get("question") or "").replace("\n", " ")[:120]
            print(f"[Q] Starting {idx+1}/{total}: {q}", flush=True)
            res = await run_query_only(row)
            q_counters["in_flight"] -= 1
            q_counters["completed"] += 1
            q_counters["query_latency_sum"] += float(res.get("query_latency_s") or 0.0)

            elapsed = time.monotonic() - q_start
            pace = elapsed / q_counters["completed"] if q_counters["completed"] else 0
            eta = pace * (total - q_counters["completed"])
            avg_q = q_counters["query_latency_sum"] / q_counters["completed"] if q_counters["completed"] else 0
            print(
                f"[Q] Finished {q_counters['completed']}/{total} elapsed={int(elapsed)}s avg_q={avg_q:.1f}s ETA={int(eta)}s",
                flush=True,
            )
            return res

    query_tasks = [asyncio.create_task(query_worker(i, row)) for i, row in enumerate(rows)]
    for future in asyncio.as_completed(query_tasks):
        query_results.append(await future)

    return query_results


async def evaluate_grades(query_results: list[dict]) -> list[dict]:
    total = len(query_results)
    print("Phase 2/2: grading answers...", flush=True)

    graded: list[dict] = []
    g_start = time.monotonic()
    grade_latency_sum = 0.0

    for i, partial in enumerate(query_results, start=1):
        q = (partial.get("question") or "").replace("\n", " ")[:120]
        print(f"[G] Grading {i}/{total}: {q}", flush=True)
        res = await run_grade_only(partial)
        graded.append(res)
        grade_latency_sum += float(res.get("grade_latency_s") or 0.0)
        elapsed = time.monotonic() - g_start
        pace = elapsed / i if i else 0
        eta = pace * (total - i)
        avg_g = grade_latency_sum / i if i else 0
        print(f"[G] Done {i}/{total} elapsed={int(elapsed)}s avg_g={avg_g:.1f}s ETA={int(eta)}s", flush=True)

    return graded


async def evaluate_dataset(rows, concurrency: int) -> list[dict]:
    two_phase = os.getenv("RAG_EVAL_TWO_PHASE", "1").strip().lower() in {"1", "true", "yes", "y"}

    if not two_phase:
        # Original single-phase behavior (query + grade per item).
        semaphore = asyncio.Semaphore(max(1, concurrency))
        total = len(rows)
        results: list[dict] = []

        counters = {"started": 0, "completed": 0, "in_flight": 0, "total_latency_sum": 0.0}
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

                # Track actual latency for reporting
                counters["total_latency_sum"] += res.get("total_latency_s", 0)

                elapsed = time.monotonic() - start_time
                # 'pace' is seconds per completed item (wall time), used for ETA
                pace = elapsed / counters["completed"] if counters["completed"] else 0
                eta = pace * (total - counters["completed"])

                # 'avg_lat' is the true average duration of the tasks so far
                avg_lat = counters["total_latency_sum"] / counters["completed"] if counters["completed"] else 0

                print(
                    f"Finished {counters['completed']}/{total} (idx={idx+1}) elapsed={int(elapsed)}s avg_lat={avg_lat:.1f}s ETA={int(eta)}s",
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

    # --- Two-phase mode: generate answers first, then grade. ---
    print(
        f"Two-phase mode enabled: phase1=query (concurrency={concurrency}), phase2=grading (grade concurrency via RAG_EVAL_GRADE_CONCURRENCY)",
        flush=True,
    )
    query_results = await evaluate_queries(list(rows), concurrency)
    return await evaluate_grades(query_results)


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

    concurrency = int(os.getenv("RAG_EVAL_CONCURRENCY", "2"))
    phase = os.getenv("RAG_EVAL_PHASE", "all").strip().lower()

    dataset_name = getattr(dataset, "name", "experiment_results")
    out_dir = Path(__file__).parent / "evals" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase-1 checkpoint file so you can resume grading without re-querying.
    query_csv = Path(os.getenv("RAG_EVAL_QUERY_CSV", str(out_dir / f"{dataset_name}_query_only.csv")))

    print(f"Loaded {total} rows. concurrency={concurrency} phase={phase} two_phase={os.getenv('RAG_EVAL_TWO_PHASE','1')}")

    if phase in {"query", "phase1", "q"}:
        query_results = await evaluate_queries(rows, concurrency)
        csv_path = save_results_csv(query_results, query_csv.stem)
        print(f"Phase 1 saved to: {csv_path.resolve()}")
        return

    if phase in {"grade", "phase2", "g"}:
        if not query_csv.exists():
            raise FileNotFoundError(
                f"Query checkpoint not found: {query_csv}. Run with RAG_EVAL_PHASE=query first."
            )
        query_results = load_results_csv(query_csv)
        results = await evaluate_grades(query_results)
        csv_path = save_results_csv(results, f"{dataset_name}_results")
        print(f"Results saved to: {csv_path.resolve()}")
        return

    # Default: run full evaluation (single or two-phase depending on RAG_EVAL_TWO_PHASE)
    # If two-phase is enabled, we also checkpoint phase 1 before grading.
    two_phase = os.getenv("RAG_EVAL_TWO_PHASE", "1").strip().lower() in {"1", "true", "yes", "y"}
    if two_phase:
        query_results = await evaluate_queries(rows, concurrency)
        csv_path = save_results_csv(query_results, query_csv.stem)
        print(f"Phase 1 saved to: {csv_path.resolve()}")
        results = await evaluate_grades(query_results)
    else:
        results = await evaluate_dataset(rows, concurrency)

    print("Evaluation completed successfully!")
    csv_path = save_results_csv(results, f"{dataset_name}_results")
    print(f"Results saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
