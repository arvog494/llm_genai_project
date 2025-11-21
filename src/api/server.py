# src/api/server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from graph_rag.service import GraphRAGService

app = FastAPI(title="GraphRAG Multi-Agent API", version="0.2.0")
service = GraphRAGService()


class QARequest(BaseModel):
    question: str


class QAResponse(BaseModel):
    answer: str
    citations: list[dict]
    graph_context: dict
    graph_paths: list[dict]


class BuildIndexResponse(BaseModel):
    num_sources: int
    num_chunks: int
    sources: list[str]


class UseCaseBuildRequest(BaseModel):
    brief: str
    max_sources: int | None = None


class UseCaseBuildResponse(BaseModel):
    session_id: str
    brief: str
    num_sources: int
    num_chunks: int
    selected_sources: list[dict]
    graph_path: str | None = None


class UseCaseQARequest(BaseModel):
    session_id: str
    question: str


@app.post("/build_index", response_model=BuildIndexResponse)
def build_index():
    """
    Build global index by ingesting ALL data/raw files.
    """
    result = service.build_index()
    return BuildIndexResponse(**result)


@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest):
    result = service.answer(req.question)
    return QAResponse(**result)


@app.post("/use_case/build", response_model=UseCaseBuildResponse)
def build_use_case(req: UseCaseBuildRequest):
    """
    Build a use-case specific GraphRAG index given a user brief.
    """
    try:
        result = service.build_use_case(
            brief=req.brief, max_sources=req.max_sources
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return UseCaseBuildResponse(**result)


@app.post("/use_case/qa", response_model=QAResponse)
def qa_use_case(req: UseCaseQARequest):
    result = service.answer_use_case(req.session_id, req.question)
    return QAResponse(**result)


@app.get("/health")
def health():
    return {"status": "ok"}
