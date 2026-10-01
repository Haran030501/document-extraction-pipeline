import logging
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Annotated

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_session, init_db
from app.extractor import Extractor
from app.ingest import extract_text
from app.models import AmountRow, Document, Extraction, PartyRow
from app.rag import index_document
from app.rag_api import router as rag_router
from app.review import router as review_router

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Document Entity Extraction Pipeline", version="1.0.0", lifespan=lifespan)

SessionDep = Annotated[Session, Depends(get_session)]


def get_extractor() -> Extractor:
    return Extractor()


ExtractorDep = Annotated[Extractor, Depends(get_extractor)]


def get_indexer():
    return index_document


IndexerDep = Annotated[object, Depends(get_indexer)]


class ExtractionOut(BaseModel):
    id: int
    prompt_version: str
    model: str
    status: str
    attempts: int
    errors: list
    entities: dict | None
    input_tokens: int
    output_tokens: int
    latency_ms: int
    created_at: datetime | None

    @classmethod
    def of(cls, e: Extraction) -> "ExtractionOut":
        return cls(
            id=e.id, prompt_version=e.prompt_version, model=e.model, status=e.status, attempts=e.attempts,
            errors=e.errors or [], entities=e.raw_json, input_tokens=e.input_tokens,
            output_tokens=e.output_tokens, latency_ms=e.latency_ms, created_at=e.created_at,
        )


class DocumentOut(BaseModel):
    id: int
    filename: str
    sha256: str
    page_count: int
    ocr_used: bool
    latest_extraction: ExtractionOut | None

    @classmethod
    def of(cls, d: Document) -> "DocumentOut":
        latest = d.extractions[-1] if d.extractions else None
        return cls(
            id=d.id, filename=d.filename, sha256=d.sha256, page_count=d.page_count, ocr_used=d.ocr_used,
            latest_extraction=ExtractionOut.of(latest) if latest else None,
        )


STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.include_router(review_router)
app.include_router(rag_router)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health(session: SessionDep) -> dict:
    session.execute(text("select 1"))
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentOut, status_code=201)
def upload_document(
    file: UploadFile,
    session: SessionDep,
    extractor: ExtractorDep,
    indexer: IndexerDep,
    prompt_version: Annotated[str | None, Query(pattern="^v[123]$")] = None,
) -> DocumentOut:
    data = file.file.read()
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "file must be a PDF")
    ingested = extract_text(data)

    doc = session.scalar(select(Document).where(Document.sha256 == ingested.sha256))
    if doc is None:
        doc = Document(
            filename=file.filename or "upload.pdf", sha256=ingested.sha256, page_count=ingested.page_count,
            ocr_used=ingested.ocr_used, raw_text=ingested.text,
        )
        session.add(doc)
        session.flush()

    result = extractor.extract(doc.raw_text, prompt_version or get_settings().prompt_version)
    session.add(Extraction.from_result(doc.id, result))
    session.commit()
    session.refresh(doc)
    try:
        indexer(session, doc)  # chunk + embed for /ask; uses the new extraction as chunk context
        session.commit()
    except Exception:
        session.rollback()
        logging.getLogger(__name__).exception("indexing failed for document %s", doc.id)
    return DocumentOut.of(doc)


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(session: SessionDep, limit: int = 50, offset: int = 0) -> list[DocumentOut]:
    docs = session.scalars(select(Document).order_by(Document.id).limit(limit).offset(offset)).all()
    return [DocumentOut.of(d) for d in docs]


@app.get("/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: int, session: SessionDep) -> DocumentOut:
    doc = session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, "document not found")
    return DocumentOut.of(doc)


@app.get("/documents/{doc_id}/text", response_class=PlainTextResponse)
def get_document_text(doc_id: int, session: SessionDep) -> str:
    doc = session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, "document not found")
    return doc.raw_text


@app.get("/documents/{doc_id}/extractions", response_model=list[ExtractionOut])
def get_extractions(doc_id: int, session: SessionDep) -> list[ExtractionOut]:
    doc = session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, "document not found")
    return [ExtractionOut.of(e) for e in doc.extractions]


@app.post("/documents/{doc_id}/extractions", response_model=ExtractionOut, status_code=201)
def rerun_extraction(
    doc_id: int,
    session: SessionDep,
    extractor: ExtractorDep,
    prompt_version: Annotated[str, Query(pattern="^v[123]$")] = "v3",
) -> ExtractionOut:
    doc = session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, "document not found")
    row = Extraction.from_result(doc.id, extractor.extract(doc.raw_text, prompt_version))
    session.add(row)
    session.commit()
    session.refresh(row)
    return ExtractionOut.of(row)


class SearchHit(BaseModel):
    document_id: int
    extraction_id: int
    case_name: str | None
    court: str | None
    decision_date: date | None
    disposition: str | None


@app.get("/search", response_model=list[SearchHit])
def search(
    session: SessionDep,
    party: str | None = None,
    court: str | None = None,
    disposition: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    min_amount: float | None = None,
    limit: int = 50,
) -> list[SearchHit]:
    """Search the latest successful extraction of each document."""
    latest = (
        select(Extraction.document_id, Extraction.id.label("eid"))
        .where(Extraction.status.in_(["ok", "invalid"]))
        .ext(distinct_on(Extraction.document_id))
        .order_by(Extraction.document_id, Extraction.id.desc())
        .subquery()
    )
    q = select(Extraction).join(latest, Extraction.id == latest.c.eid)
    if party:
        q = q.where(Extraction.parties.any(PartyRow.name.ilike(f"%{party}%")))
    if court:
        q = q.where(Extraction.court.ilike(f"%{court}%"))
    if disposition:
        q = q.where(Extraction.disposition == disposition)
    if date_from:
        q = q.where(Extraction.decision_date >= date_from)
    if date_to:
        q = q.where(Extraction.decision_date <= date_to)
    if min_amount is not None:
        q = q.where(Extraction.amounts.any(AmountRow.value >= min_amount))
    rows = session.scalars(q.order_by(Extraction.decision_date.desc().nulls_last()).limit(limit)).all()
    return [
        SearchHit(document_id=e.document_id, extraction_id=e.id, case_name=e.case_name, court=e.court,
                  decision_date=e.decision_date, disposition=e.disposition)
        for e in rows
    ]

