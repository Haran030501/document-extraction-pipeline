import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import review


@pytest.fixture
def client(tmp_path, monkeypatch, gold_dict, opinion_pdf):
    labels, pdfs, preds = tmp_path / "labels", tmp_path / "pdfs", tmp_path / "preds"
    for d in (labels, pdfs, preds):
        d.mkdir()
    (labels / "doc_a.json").write_text(json.dumps({"source_pdf": "doc_a.pdf", "split": "test", "entities": gold_dict}))
    (labels / "doc_b.json").write_text(json.dumps({"source_pdf": "missing.pdf", "split": "dev", "entities": gold_dict}))
    (pdfs / "doc_a.pdf").write_bytes(opinion_pdf)
    prios = tmp_path / "prios.json"
    prios.write_text(json.dumps({"doc_a": {"tags": ["judgment"], "notes": ["check the author"]}}))
    wrong = {**gold_dict, "disposition": "reversed"}
    (preds / "doc_a.json").write_text(json.dumps({"entities": wrong}))
    monkeypatch.setattr(review, "LABELS", labels)
    monkeypatch.setattr(review, "PDFS", pdfs)
    monkeypatch.setattr(review, "PRIORITIES", prios)
    monkeypatch.setattr(review, "PREDICTIONS", preds)
    app = FastAPI()
    app.include_router(review.router)
    c = TestClient(app)
    c.labels = labels
    return c


def test_list_and_get(client):
    items = {i["id"]: i for i in client.get("/api/labels").json()}
    assert items["doc_a"]["tags"] == ["judgment"] and items["doc_a"]["status"] == "unreviewed"
    assert items["doc_b"]["tags"] == []
    label = client.get("/api/labels/doc_a").json()
    assert label["priority_notes"] == ["check the author"] and label["entities"]["docket_number"] == "21-1234"


def test_pdf_and_path_safety(client):
    r = client.get("/api/labels/doc_a/pdf")
    assert r.status_code == 200 and r.content.startswith(b"%PDF")
    assert client.get("/api/labels/doc_b/pdf").status_code == 404
    assert client.get("/api/labels/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get("/api/labels/nope").status_code == 404


def test_disagreements(client):
    d = client.get("/api/labels/doc_a/disagreements").json()
    assert d["available"] and list(d["fields"]) == ["disposition"]
    assert d["fields"]["disposition"]["model"] == "reversed"
    assert client.get("/api/labels/doc_b/disagreements").json()["available"] is False


def test_verify(client):
    r = client.put("/api/labels/doc_a", json={"action": "verify", "note": "checked p.1"})
    assert r.status_code == 200 and r.json()["status"] == "verified"
    saved = json.loads((client.labels / "doc_a.json").read_text())
    assert saved["review"]["note"] == "checked p.1" and "adjudication_notes" not in saved


def test_correct(client, gold_dict):
    new = {**gold_dict, "author_judge": None}
    assert client.put("/api/labels/doc_a", json={"action": "correct", "entities": new}).status_code == 422  # note required
    assert client.put("/api/labels/doc_a", json={"action": "correct", "entities": gold_dict, "note": "x"}).status_code == 422  # no change
    bad = {**gold_dict, "disposition": "won"}
    assert client.put("/api/labels/doc_a", json={"action": "correct", "entities": bad, "note": "x"}).status_code == 422
    r = client.put("/api/labels/doc_a", json={"action": "correct", "entities": new, "note": "p.1: per curiam"})
    assert r.status_code == 200 and r.json()["changed_fields"] == ["author_judge"]
    saved = json.loads((client.labels / "doc_a.json").read_text())
    assert saved["entities"]["author_judge"] is None and saved["review"]["status"] == "corrected"
    assert "author_judge: p.1: per curiam" in saved["adjudication_notes"][-1]
