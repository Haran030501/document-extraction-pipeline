import io
import os

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Point the app at the test database before any app module creates its engine.
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg:///extraction_test?host=/var/run/postgresql")

OPINION_TEXT = [
    "UNITED STATES COURT OF APPEALS FOR THE NINTH CIRCUIT",
    "JANE DOE, Plaintiff-Appellant, v. ACME CORP., Defendant-Appellee.",
    "No. 21-1234",
    "Filed March 3, 2023",
    "Before: SMITH, JONES, and LEE, Circuit Judges. Opinion by Judge Smith.",
    "The jury awarded $250,000 in damages under 42 U.S.C. § 1983. AFFIRMED.",
]


def make_pdf(lines: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    y = 720
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture
def opinion_pdf() -> bytes:
    return make_pdf(OPINION_TEXT)


@pytest.fixture
def gold_dict() -> dict:
    return {
        "case_name": "Doe v. Acme Corp.",
        "docket_number": "21-1234",
        "court": "United States Court of Appeals for the Ninth Circuit",
        "decision_date": "2023-03-03",
        "judges": ["Smith", "Jones", "Lee"],
        "author_judge": "Smith",
        "parties": [{"name": "Jane Doe", "role": "appellant"}, {"name": "Acme Corp.", "role": "appellee"}],
        "disposition": "affirmed",
        "monetary_amounts": [{"value": 250000, "currency": "USD", "context": "jury award"}],
        "cited_statutes": ["42 U.S.C. § 1983"],
    }


def fake_embed(text: str):
    """Deterministic bag-of-words hashing embedding: texts sharing words get similar vectors."""
    import hashlib
    import re

    import numpy as np

    v = np.zeros(384, dtype=np.float32)
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 384] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v + 1e-3


@pytest.fixture(autouse=True)
def offline_embeddings(monkeypatch):
    """Never download or run the real embedding model in tests."""
    from app import rag

    monkeypatch.setattr(rag, "embed_passages", lambda texts: [fake_embed(t) for t in texts])
    monkeypatch.setattr(rag, "embed_query", fake_embed)
