from __future__ import annotations

import asyncio
import time
import csv
import json
import os
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
    "You are an evaluation assistant. Always return STRICT JSON with the requested fields."
)

EVAL_USER_PROMPT = """
Evaluate the response using the grading notes and question.
Return a JSON object with exactly these keys and allowed values:
"coverage": one of ["high", "partial", "low"]
"grounding": one of ["grounded", "hallucinated"]
"faithfulness": one of ["faithful", "unfaithful"]
"answer_relevancy": one of ["on-topic", "partial", "off-topic"]

Question: {question}
Grading Notes: {grading_notes}
Response: {response}

JSON only, no explanations.
"""


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


def _extract_json_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _classify_response_sync(question: str, grading_notes: str, response_text: str) -> dict:
    if not response_text.strip():
        return {**DEFAULT_LABELS}

    try:
        completion = client.chat.completions.create(
            model="Gpt-oss:20b",
            temperature=0,
            max_tokens=200,
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
        parsed = json.loads(_extract_json_block(message or ""))
    except Exception:
        parsed = {}

    labels = {**DEFAULT_LABELS}
    labels.update({k: str(v) for k, v in parsed.items() if k in labels})
    return labels


async def classify_response(question: str, grading_notes: str, response_text: str) -> dict:
    cache_key = (question, grading_notes, response_text)
    with _classification_lock:
        cached = _classification_cache.get(cache_key)
    if cached is not None:
        return cached

    loop = asyncio.get_running_loop()
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
    
    raise last_error


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
    response = await query_with_cache(row["question"])
    answer = response.get("answer", "")

    labels = await classify_response(
        question=row["question"],
        grading_notes=row["grading_notes"],
        response_text=answer,
    )

    conciseness = measure_conciseness(answer)

    return {
        **row,
        "response": answer,
        "coverage": labels["coverage"],
        "grounding": labels["grounding"],
        "faithfulness": labels["faithfulness"],
        "answer_relevancy": labels["answer_relevancy"],
        "conciseness": conciseness,
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
