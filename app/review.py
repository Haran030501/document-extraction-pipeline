"""Gold-label review: browse eval labels next to their PDFs, verify them, or save corrections.

Writes to eval/labels/*.json, so it is meant for local use and should not be exposed publicly.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError

from app.schemas import CourtOpinion

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "eval" / "labels"
PDFS = ROOT / "data" / "pdfs"
PRIORITIES = ROOT / "eval" / "review_priorities.json"
# Reference predictions shown only when the reviewer asks for them.
PREDICTIONS = ROOT / "eval" / "cache" / "claude-opus-5-5-high" / "v3"

router = APIRouter()


def _label_path(doc_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", doc_id) or not (LABELS / f"{doc_id}.json").is_file():
        raise HTTPException(404, "label not found")
    return LABELS / f"{doc_id}.json"


def _priorities() -> dict:
    return json.loads(PRIORITIES.read_text()) if PRIORITIES.exists() else {}


def _status(label: dict) -> str:
    return label.get("review", {}).get("status", "unreviewed")


@router.get("/review", include_in_schema=False)
def review_page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "review.html")


@router.get("/api/labels", tags=["review"])
def list_labels() -> list[dict]:
    prios = _priorities()
    out = []
    for p in sorted(LABELS.glob("*.json")):
        label = json.loads(p.read_text())
        out.append({
            "id": p.stem,
            "case_name": label["entities"].get("case_name"),
            "split": label.get("split"),
            "status": _status(label),
            "tags": prios.get(p.stem, {}).get("tags", []),
        })
    return out


@router.get("/api/labels/{doc_id}", tags=["review"])
def get_label(doc_id: str) -> dict:
    label = json.loads(_label_path(doc_id).read_text())
    prio = _priorities().get(doc_id, {})
    return {**label, "id": doc_id, "tags": prio.get("tags", []), "priority_notes": prio.get("notes", [])}


@router.get("/api/labels/{doc_id}/pdf", tags=["review"])
def get_label_pdf(doc_id: str) -> FileResponse:
    label = json.loads(_label_path(doc_id).read_text())
    pdf = PDFS / label["source_pdf"]
    if not pdf.is_file():
        raise HTTPException(404, "PDF not found; run scripts/fetch_opinions.py")
    return FileResponse(pdf, media_type="application/pdf")


@router.get("/api/labels/{doc_id}/disagreements", tags=["review"])
def get_disagreements(doc_id: str) -> dict:
    """Fields where the cached Opus 5.5 v3 extraction disagrees with the gold label."""
    from eval.scoring import score_document  # eval is only needed for this optional view

    gold = CourtOpinion.model_validate(json.loads(_label_path(doc_id).read_text())["entities"])
    cached = PREDICTIONS / f"{doc_id}.json"
    if not cached.is_file():
        return {"available": False, "fields": {}}
    entities = json.loads(cached.read_text()).get("entities")
    if not entities:
        return {"available": False, "fields": {}}
    pred = CourtOpinion.model_validate(entities)
    scores = score_document(pred, gold)
    predicted = pred.model_dump(mode="json")
    return {
        "available": True,
        "fields": {f: {"score": s, "model": predicted[f]} for f, s in scores.items() if s < 1.0},
    }


class ReviewIn(BaseModel):
    action: Literal["verify", "correct"]
    entities: dict | None = None
    note: str = ""


@router.put("/api/labels/{doc_id}", tags=["review"])
def save_review(doc_id: str, body: ReviewIn) -> dict:
    path = _label_path(doc_id)
    label = json.loads(path.read_text())
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if body.action == "correct":
        if body.entities is None:
            raise HTTPException(422, "entities are required for a correction")
        if not body.note.strip():
            raise HTTPException(422, "a correction needs a note citing the source text")
        try:
            new = CourtOpinion.model_validate(body.entities).model_dump(mode="json")
        except ValidationError as e:
            raise HTTPException(422, e.errors(include_url=False)) from e
        changed = sorted(f for f in new if new[f] != label["entities"].get(f))
        if not changed:
            raise HTTPException(422, "no fields changed; use verify instead")
        label["entities"] = new
        label.setdefault("adjudication_notes", []).append(f"[{now[:10]} human review] {', '.join(changed)}: {body.note.strip()}")
        label["review"] = {"status": "corrected", "reviewed_at": now, "changed_fields": changed}
    else:
        label["review"] = {"status": "verified", "reviewed_at": now}
        if body.note.strip():
            label["review"]["note"] = body.note.strip()
    path.write_text(json.dumps(label, indent=2, ensure_ascii=False) + "\n")
    return {"id": doc_id, **label["review"]}
