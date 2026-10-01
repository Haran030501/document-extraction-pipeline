from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api import app, get_extractor
from app.chunking import chunk_text, split_pages
from app.db import Base, SessionLocal, engine, init_db
from app.extractor import ExtractionResult
from app.models import Document, Extraction
from app.rag import Filters, index_document, keyword_terms, parse_citations, retrieve, rrf
from app.rag_api import get_answer_client
from app.schemas import CourtOpinion

# ---------- pure functions ----------


def test_split_pages_and_chunking():
    text = "[Page 1]\nFirst page line.\n\n[Page 2]\n" + "\n".join(f"Sentence number {i} about restitution." for i in range(80))
    assert [p for p, _ in split_pages(text)] == [1, 2]
    chunks = chunk_text(text, size=300, overlap=60, min_size=10)
    assert chunks[0].page == 1 and chunks[0].text == "First page line."
    page2 = [c for c in chunks if c.page == 2]
    assert len(page2) > 3 and all(len(c.text) <= 300 for c in page2)
    # consecutive chunks overlap, and no chunk mixes pages
    assert page2[0].text.split(". ")[-1] in page2[1].text
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunk_text("no page markers here, but long enough to keep around") [0].page == 1


def test_parse_citations():
    assert parse_citations("A [1][3], B [2, 4], C [5-6] and [9].", n_hits=6) == ([1, 3, 2, 4, 5, 6], [9])
    assert parse_citations("no citations", 3) == ([], [])


def test_rrf_prefers_items_ranked_by_both():
    scores = rrf([[1, 2, 3], [3, 4, 1]])
    assert max(scores, key=scores.get) in (1, 3) and scores[1] > scores[2] and scores[3] > scores[4]


# ---------- retrieval against Postgres (pgvector + tsvector) ----------

OPINION_A = ("[Page 1]\nUNITED STATES v. SMITH\nThe district court ordered restitution of $57,044.96 to the gun store owners "
             "after the defendant stole firearms. This case concerns the restitution order.\n"
             "[Page 2]\nThe court of appeals vacated the lost income component of the restitution award.")
OPINION_B = ("[Page 1]\nDOE v. ACME HOTEL\nGuests claimed the hotel concealed Legionella bacteria in the water system. "
             "This case concerns consumer fraud and the resort fee.\n[Page 2]\nThe dismissal is affirmed in this case.")


def _add(session, name, text, court, disposition):
    doc = Document(filename=f"{name}.pdf", sha256=name.ljust(64, "0"), page_count=2, ocr_used=False, raw_text=text)
    session.add(doc)
    session.flush()
    op = CourtOpinion(case_name=name, court=court, disposition=disposition, parties=[])
    session.add(Extraction.from_result(doc.id, ExtractionResult(version="v3", model="m", opinion=op, status="ok", attempts=1)))
    session.flush()
    session.refresh(doc)
    index_document(session, doc)
    session.commit()
    return doc


@pytest.fixture
def corpus():
    from app import models  # noqa: F401

    Base.metadata.drop_all(engine)
    init_db()
    with SessionLocal() as s:
        a = _add(s, "United States v. Smith", OPINION_A, "Third Circuit", "affirmed_in_part")
        b = _add(s, "Doe v. Acme Hotel", OPINION_B, "Ninth Circuit", "affirmed")
        yield s, a.id, b.id


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_retrieve_modes(corpus, mode):
    s, a, b = corpus
    hits = retrieve(s, "How much restitution did the gun store owners get?", k=3, mode=mode)
    assert hits and hits[0].document_id == a and hits[0].page == 1
    assert hits[0].case_name == "United States v. Smith"
    hits = retrieve(s, "Legionella in the hotel water", k=3, mode=mode)
    assert hits[0].document_id == b


def test_filters_and_stopwords(corpus, monkeypatch):
    from app import rag

    s, a, b = corpus
    hits = retrieve(s, "restitution case", k=5, filters=Filters(court="Ninth"))
    assert hits and {h.document_id for h in hits} == {b}
    assert retrieve(s, "restitution", k=5, mode="keyword", filters=Filters(disposition="affirmed")) == []
    # 4 chunks: 'case' is in 3 (75%), 'restitut' in 2 (50%). With a 60% cutoff only 'case' is dropped.
    monkeypatch.setattr(rag, "MAX_DF", 0.6)
    assert keyword_terms(s, "which case involved restitution") == ["restitut"]
    # If every term is too common, fall back to all of them rather than matching nothing.
    monkeypatch.setattr(rag, "MAX_DF", 0.1)
    assert keyword_terms(s, "which case involved restitution") == ["case", "restitut"]


def test_reindex_replaces_chunks(corpus):
    s, a, _ = corpus
    doc = s.get(Document, a)
    n1 = index_document(s, doc)
    s.commit()
    assert len(s.get(Document, a).chunks) == n1


# ---------- API ----------


class FakeAnswerClient:
    def __init__(self, text):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.text = text

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(stop_reason="end_turn", usage=SimpleNamespace(input_tokens=500, output_tokens=60),
                               content=[SimpleNamespace(type="text", text=self.text)])


def test_ask_and_semantic_search_endpoints(corpus):
    fake = FakeAnswerClient("Smith was ordered to pay $57,044.96 [1]. Unsupported [7].")
    app.dependency_overrides[get_answer_client] = lambda: fake
    try:
        with TestClient(app) as c:
            r = c.get("/search/semantic", params={"q": "restitution for stolen guns", "k": 3})
            assert r.status_code == 200 and r.json()[0]["case_name"] == "United States v. Smith"
            r = c.post("/ask", json={"question": "How much restitution was ordered?", "k": 4})
            body = r.json()
            assert r.status_code == 200 and body["status"] == "ok"
            assert body["cited"] == [1] and body["invalid_citations"] == [7]
            assert body["sources"][0]["n"] == 1
            prompt = fake.calls[0]["messages"][0]["content"]
            assert '<excerpt n="1"' in prompt and "Question: How much restitution was ordered?" in prompt
            assert c.post("/ask", json={"question": "x"}).status_code == 422
            assert c.get("/ask").status_code == 200
    finally:
        app.dependency_overrides.pop(get_answer_client, None)


def test_ask_with_no_matches_skips_llm(corpus):
    fake = FakeAnswerClient("should not be called")
    app.dependency_overrides[get_answer_client] = lambda: fake
    try:
        with TestClient(app) as c:
            body = c.post("/ask", json={"question": "restitution", "filters": {"court": "Nonexistent"}}).json()
            assert body["status"] == "no_results" and fake.calls == []
    finally:
        app.dependency_overrides.pop(get_answer_client, None)


def test_upload_indexes_document(corpus, opinion_pdf, gold_dict):
    stub = SimpleNamespace(extract=lambda text, v="v3": ExtractionResult(
        version=v, model="stub", opinion=CourtOpinion.model_validate(gold_dict), status="ok", attempts=1))
    app.dependency_overrides[get_extractor] = lambda: stub
    try:
        with TestClient(app) as c:
            doc = c.post("/documents", files={"file": ("op.pdf", opinion_pdf, "application/pdf")}).json()
            hits = c.get("/search/semantic", params={"q": "Acme jury damages", "k": 5}).json()
            assert any(h["document_id"] == doc["id"] for h in hits)
    finally:
        app.dependency_overrides.pop(get_extractor, None)
