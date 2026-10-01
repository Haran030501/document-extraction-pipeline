from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Computed, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import get_settings
from app.db import Base
from app.extractor import ExtractionResult

JSONType = JSON().with_variant(JSONB(), "postgresql")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(512))
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    page_count: Mapped[int] = mapped_column(Integer)
    ocr_used: Mapped[bool] = mapped_column(default=False)
    raw_text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    extractions: Mapped[list["Extraction"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Extraction.id"
    )
    chunks: Mapped[list["ChunkRow"]] = relationship(cascade="all, delete-orphan", order_by="ChunkRow.chunk_index")


class Extraction(Base):
    __tablename__ = "extractions"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    prompt_version: Mapped[str] = mapped_column(String(16))
    model: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), index=True)
    attempts: Mapped[int] = mapped_column(Integer)
    errors: Mapped[list] = mapped_column(JSONType, default=list)
    raw_json: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Denormalized scalar fields for querying.
    case_name: Mapped[str | None] = mapped_column(String(512))
    docket_number: Mapped[str | None] = mapped_column(String(128), index=True)
    court: Mapped[str | None] = mapped_column(String(256), index=True)
    decision_date: Mapped[date | None] = mapped_column(Date, index=True)
    author_judge: Mapped[str | None] = mapped_column(String(128))
    disposition: Mapped[str | None] = mapped_column(String(32), index=True)

    document: Mapped[Document] = relationship(back_populates="extractions")
    parties: Mapped[list["PartyRow"]] = relationship(cascade="all, delete-orphan")
    judges: Mapped[list["JudgeRow"]] = relationship(cascade="all, delete-orphan")
    amounts: Mapped[list["AmountRow"]] = relationship(cascade="all, delete-orphan")
    statutes: Mapped[list["StatuteRow"]] = relationship(cascade="all, delete-orphan")

    @classmethod
    def from_result(cls, document_id: int, r: ExtractionResult) -> "Extraction":
        row = cls(
            document_id=document_id, prompt_version=r.version, model=r.model, status=r.status,
            attempts=r.attempts, errors=r.errors, raw_json=r.raw_json,
            input_tokens=r.input_tokens, output_tokens=r.output_tokens, latency_ms=r.latency_ms,
        )
        op = r.opinion
        if op:
            row.case_name, row.docket_number, row.court = op.case_name, op.docket_number, op.court
            row.decision_date, row.author_judge, row.disposition = op.decision_date, op.author_judge, op.disposition
            row.parties = [PartyRow(name=p.name, role=p.role) for p in op.parties]
            row.judges = [JudgeRow(name=j) for j in op.judges]
            row.amounts = [AmountRow(value=a.value, currency=a.currency, context=a.context) for a in op.monetary_amounts]
            row.statutes = [StatuteRow(citation=s) for s in op.cited_statutes]
        return row


class PartyRow(Base):
    __tablename__ = "parties"

    id: Mapped[int] = mapped_column(primary_key=True)
    extraction_id: Mapped[int] = mapped_column(ForeignKey("extractions.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(512), index=True)
    role: Mapped[str] = mapped_column(String(32))


class JudgeRow(Base):
    __tablename__ = "judges"

    id: Mapped[int] = mapped_column(primary_key=True)
    extraction_id: Mapped[int] = mapped_column(ForeignKey("extractions.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(256), index=True)


class AmountRow(Base):
    __tablename__ = "amounts"

    id: Mapped[int] = mapped_column(primary_key=True)
    extraction_id: Mapped[int] = mapped_column(ForeignKey("extractions.id", ondelete="CASCADE"), index=True)
    value: Mapped[float] = mapped_column(Float, index=True)
    currency: Mapped[str] = mapped_column(String(8))
    context: Mapped[str] = mapped_column(Text)


class StatuteRow(Base):
    __tablename__ = "statutes"

    id: Mapped[int] = mapped_column(primary_key=True)
    extraction_id: Mapped[int] = mapped_column(ForeignKey("extractions.id", ondelete="CASCADE"), index=True)
    citation: Mapped[str] = mapped_column(String(256), index=True)


class ChunkRow(Base):
    """A page-scoped passage of a document, embedded for semantic search and indexed for full-text search."""

    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_embedding_hnsw", "embedding", postgresql_using="hnsw",
              postgresql_ops={"embedding": "vector_cosine_ops"}),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    page: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    embedding = mapped_column(Vector(get_settings().embedding_dim))
    tsv = mapped_column(TSVECTOR, Computed("to_tsvector('english', text)", persisted=True))
