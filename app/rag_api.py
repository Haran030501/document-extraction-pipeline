from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import anthropic
from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_session
from app.rag import Filters, Hit, answer_question, retrieve

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]
ModeParam = Literal["hybrid", "vector", "keyword"]


def get_answer_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key)


class HitOut(BaseModel):
    n: int
    document_id: int
    chunk_id: int
    page: int
    case_name: str | None
    court: str | None
    decision_date: date | None
    filename: str | None
    text: str
    score: float
    vector_rank: int | None
    keyword_rank: int | None

    @classmethod
    def of(cls, n: int, h: Hit) -> "HitOut":
        return cls(n=n, document_id=h.document_id, chunk_id=h.chunk_id, page=h.page, case_name=h.case_name,
                   court=h.court, decision_date=h.decision_date, filename=h.filename, text=h.text,
                   score=round(h.score, 5), vector_rank=h.vector_rank, keyword_rank=h.keyword_rank)


class FiltersIn(BaseModel):
    court: str | None = None
    disposition: str | None = None
    date_from: date | None = None
    date_to: date | None = None


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(8, ge=1, le=20)
    mode: ModeParam = "hybrid"
    filters: FiltersIn = FiltersIn()


class AskOut(BaseModel):
    answer: str
    status: str
    cited: list[int]
    invalid_citations: list[int]
    sources: list[HitOut]
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


@router.get("/ask", include_in_schema=False)
def ask_page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "ask.html")


@router.get("/search/semantic", response_model=list[HitOut], tags=["rag"])
def semantic_search(
    session: SessionDep,
    q: Annotated[str, Query(min_length=2, max_length=1000)],
    k: Annotated[int, Query(ge=1, le=50)] = 8,
    mode: ModeParam = "hybrid",
    court: str | None = None,
    disposition: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> list[HitOut]:
    """Passage retrieval only (no LLM call, no cost)."""
    hits = retrieve(session, q, k=k, mode=mode, filters=Filters(court, disposition, date_from, date_to))
    return [HitOut.of(i, h) for i, h in enumerate(hits, start=1)]


@router.post("/ask", response_model=AskOut, tags=["rag"])
def ask(body: AskIn, session: SessionDep,
        client: Annotated[anthropic.Anthropic, Depends(get_answer_client)]) -> AskOut:
    """Retrieve passages, then have Claude answer from them with [n] citations."""
    hits = retrieve(session, body.question, k=body.k, mode=body.mode, filters=Filters(**body.filters.model_dump()))
    a = answer_question(body.question, hits, client=client)
    return AskOut(answer=a.text, status=a.status, cited=a.cited, invalid_citations=a.invalid_citations,
                  sources=[HitOut.of(i, h) for i, h in enumerate(hits, start=1)], model=a.model,
                  input_tokens=a.input_tokens, output_tokens=a.output_tokens, latency_ms=a.latency_ms)
