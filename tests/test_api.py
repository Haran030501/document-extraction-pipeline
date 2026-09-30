import pytest
from fastapi.testclient import TestClient

from app.api import app, get_extractor
from app.db import Base, engine
from app.extractor import ExtractionResult
from app.schemas import CourtOpinion


class StubExtractor:
    model = "stub"

    def __init__(self, opinion: CourtOpinion):
        self.opinion = opinion
        self.calls = 0

    def extract(self, text: str, version: str = "v3") -> ExtractionResult:
        self.calls += 1
        return ExtractionResult(version=version, model="stub", opinion=self.opinion, status="ok",
                                attempts=1, input_tokens=10, output_tokens=5)


@pytest.fixture
def client(gold_dict):
    from app import models  # noqa: F401

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    stub = StubExtractor(CourtOpinion.model_validate(gold_dict))
    app.dependency_overrides[get_extractor] = lambda: stub
    with TestClient(app) as c:
        c.stub = stub
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_and_fetch(client, opinion_pdf):
    r = client.post("/documents", files={"file": ("op.pdf", opinion_pdf, "application/pdf")})
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["page_count"] == 1
    ext = doc["latest_extraction"]
    assert ext["status"] == "ok" and ext["entities"]["docket_number"] == "21-1234"

    # Re-uploading the same file dedupes the document but records a new extraction.
    r2 = client.post("/documents?prompt_version=v2", files={"file": ("op.pdf", opinion_pdf, "application/pdf")})
    assert r2.json()["id"] == doc["id"]
    exts = client.get(f"/documents/{doc['id']}/extractions").json()
    assert [e["prompt_version"] for e in exts] == ["v3", "v2"]


def test_rejects_non_pdf(client):
    r = client.post("/documents", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_search(client, opinion_pdf):
    client.post("/documents", files={"file": ("op.pdf", opinion_pdf, "application/pdf")})
    assert len(client.get("/search", params={"party": "acme"}).json()) == 1
    assert len(client.get("/search", params={"disposition": "affirmed", "min_amount": 100000}).json()) == 1
    assert client.get("/search", params={"disposition": "reversed"}).json() == []
    assert client.get("/search", params={"date_from": "2024-01-01"}).json() == []


def test_404(client):
    assert client.get("/documents/999").status_code == 404


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Opinion Extractor" in r.text


def test_document_text(client, opinion_pdf):
    doc = client.post("/documents", files={"file": ("op.pdf", opinion_pdf, "application/pdf")}).json()
    r = client.get(f"/documents/{doc['id']}/text")
    assert r.status_code == 200 and "21-1234" in r.text
    assert client.get("/documents/999/text").status_code == 404
