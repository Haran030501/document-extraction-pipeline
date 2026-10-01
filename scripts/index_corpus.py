"""Load the labeled corpus into the database and build the RAG index.

    python -m scripts.index_corpus            # 30 labeled opinions + any un-indexed uploads
    python -m scripts.index_corpus --extract  # also run Claude extraction where no cached result exists

Extractions are reused from the eval cache (eval/cache/<model>-<effort>/v3) when available,
so indexing the corpus makes no API calls by default. Re-running is safe: documents are
deduplicated by SHA-256 and their chunks are rebuilt.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal, init_db
from app.extractor import ExtractionResult, Extractor
from app.ingest import extract_text
from app.models import ChunkRow, Document, Extraction
from app.rag import index_document
from app.schemas import CourtOpinion

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "eval" / "labels"
PDFS = ROOT / "data" / "pdfs"

log = logging.getLogger("index")


def cached_extraction(doc_id: str) -> ExtractionResult | None:
    s = get_settings()
    path = ROOT / "eval" / "cache" / f"{s.model}-{s.effort}" / "v3" / f"{doc_id}.json"
    if not path.is_file():
        return None
    rec = json.loads(path.read_text())
    if not rec.get("entities"):
        return None
    fields = {k: rec[k] for k in ("version", "model", "status", "attempts", "errors",
                                  "input_tokens", "output_tokens", "latency_ms")}
    return ExtractionResult(opinion=CourtOpinion.model_validate(rec["entities"]), **fields)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="call Claude when no cached extraction exists")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    init_db()
    extractor = Extractor() if args.extract else None

    with SessionLocal() as session:
        for label_path in sorted(LABELS.glob("*.json")):
            doc_id = label_path.stem
            pdf = PDFS / json.loads(label_path.read_text())["source_pdf"]
            data = pdf.read_bytes()
            doc = session.scalar(select(Document).where(Document.sha256 == hashlib.sha256(data).hexdigest()))
            if doc is None:
                ing = extract_text(data)
                doc = Document(filename=pdf.name, sha256=ing.sha256, page_count=ing.page_count,
                               ocr_used=ing.ocr_used, raw_text=ing.text)
                session.add(doc)
                session.flush()
            if not doc.extractions:
                result = cached_extraction(doc_id) or (extractor.extract(doc.raw_text, "v3") if extractor else None)
                if result:
                    session.add(Extraction.from_result(doc.id, result))
                    session.flush()
                    session.refresh(doc)
            n = index_document(session, doc)
            session.commit()
            log.info("%-16s %3d chunks  %s", doc_id, n, "(OCR)" if doc.ocr_used else "")

        # Uploaded documents that were never indexed (e.g. uploaded before RAG existed).
        unindexed = session.scalars(select(Document).where(~Document.chunks.any())).all()
        for doc in unindexed:
            n = index_document(session, doc)
            session.commit()
            log.info("upload %-12s %3d chunks", doc.filename[:30], n)

        total = session.query(ChunkRow).count()
        docs = session.query(Document).count()
    log.info("index ready: %d chunks across %d documents", total, docs)


if __name__ == "__main__":
    main()
