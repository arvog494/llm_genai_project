import time
import json
import requests
import pandas as pd
from pathlib import Path
import random

BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

# ========= A REMPLACER =========
SESSION_ID = "6fff736c-e9e4-461a-a5bb-43ffe2178cc0"   # ex: "8a3dd2b8-..."
DOC_ID = "2410.14077v2"            # le PDF que tu as téléchargé
# ===============================

# 1) Charger le benchmark
BENCH = Path("data/open_ragbench")
queries = json.load(open(BENCH / "queries.json", encoding="utf-8"))
answers = json.load(open(BENCH / "answers.json", encoding="utf-8"))
qrels   = json.load(open(BENCH / "qrels.json", encoding="utf-8"))

# 2) Filtrer UNIQUEMENT les queries liées à ce PDF
filtered_qids = [
    qid for qid, rel in qrels.items()
    if rel["doc_id"] == DOC_ID and qid in queries
]

print("Queries compatibles:", len(filtered_qids))
if len(filtered_qids) == 0:
    raise RuntimeError(
        "0 query trouvée pour ce DOC_ID. "
        "Vérifie que DOC_ID est bien celui de pdf_urls.json et que le fichier s'appelle DOC_ID.pdf"
    )

# échantillon (10 max recommandé)
random.seed(42)
sample_qids = random.sample(filtered_qids, min(10, len(filtered_qids)))

# 3) Boucle d’évaluation
rows = []

target_pdf_token = f"{DOC_ID}.pdf"   # ex: "2410.14077v2.pdf"

for i, qid in enumerate(sample_qids, 1):
    question = queries[qid]["query"]

    t0 = time.time()
    resp = requests.post(
        f"{BASE_URL}/use_case/qa",
        headers=HEADERS,
        json={
            "session_id": SESSION_ID,
            "question": question
        },
        timeout=600
    ).json()
    latency = time.time() - t0

    citations = resp.get("citations", []) or []
    cited_sources = [c.get("source", "") for c in citations]

    # HIT si au moins une citation vient du bon PDF
    hit_doc = any(target_pdf_token in (src or "") for src in cited_sources)

    rows.append({
        "query_id": qid,
        "question": question,
        "pred_answer": resp.get("answer", ""),
        "gt_answer": answers.get(qid, ""),
        "gt_doc_id": qrels[qid]["doc_id"],
        "gt_section_id": qrels[qid].get("section_id", None),
        "latency_s": round(latency, 2),
        "num_citations": len(citations),
        "hit_doc": hit_doc,
        "query_type": queries[qid].get("type"),
        "query_source": queries[qid].get("source"),
        "cited_sources": ";".join(sorted(set([s for s in cited_sources if s])))
    })

    print(f"[{i}/{len(sample_qids)}] latency={latency:.2f}s hit_doc={hit_doc}")

# 4) Sauvegarde
out_dir = Path("evaluation/results")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(rows)
df.to_csv(out_dir / f"eval_{DOC_ID}.csv", index=False, encoding="utf-8")

print("✅ Évaluation terminée. CSV généré :", out_dir / f"eval_{DOC_ID}.csv")

