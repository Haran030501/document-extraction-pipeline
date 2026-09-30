"""Run prompt versions against the hand-labeled set and report field-level accuracy.

    python -m eval.run_eval --versions v1 v2 v3 [--split test] [--no-cache]

Each label in eval/labels/<id>.json looks like:
    {"source_pdf": "<file in data/pdfs>", "split": "dev"|"test", "entities": {<CourtOpinion>}}

Extractions are cached in eval/cache/<model>/<version>/<id>.json so re-runs only call
the API for new (document, version) pairs.
"""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean

from app.extractor import VERSIONS, ExtractionResult, Extractor
from app.ingest import extract_text
from app.schemas import CourtOpinion
from eval.scoring import FIELDS, score_document

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "eval" / "labels"
CACHE = ROOT / "eval" / "cache"
RESULTS = ROOT / "eval" / "results"
PDFS = ROOT / "data" / "pdfs"

# USD per million tokens (input, output). Used only for the cost column.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5": (5.0, 25.0), "claude-haiku-4-5": (1.0, 5.0),
}

log = logging.getLogger("eval")


def load_labels(split: str | None) -> dict[str, dict]:
    labels = {}
    for p in sorted(LABELS.glob("*.json")):
        label = json.loads(p.read_text())
        if split and label.get("split") != split:
            continue
        labels[p.stem] = label
    return labels


def document_text(doc_id: str, pdf_name: str) -> str:
    cached = CACHE / "text" / f"{doc_id}.txt"
    if cached.exists():
        return cached.read_text()
    text = extract_text((PDFS / pdf_name).read_bytes()).text
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(text)
    return text


def run_one(extractor: Extractor, doc_id: str, label: dict, version: str, use_cache: bool) -> dict:
    path = CACHE / f"{extractor.model}-{extractor.effort}" / version / f"{doc_id}.json"
    if use_cache and path.exists():
        return json.loads(path.read_text())
    result: ExtractionResult = extractor.extract(document_text(doc_id, label["source_pdf"]), version)
    record = {k: v for k, v in asdict(result).items() if k != "opinion"}
    record["entities"] = result.raw_json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str))
    log.info("%s %s -> %s (%d attempts)", version, doc_id, result.status, result.attempts)
    return record


def summarize(version: str, model: str, rows: list[dict]) -> dict:
    per_field = {f: mean(r["scores"][f] for r in rows) for f in FIELDS}
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    cost = [(r["input_tokens"] * price_in + r["output_tokens"] * price_out) / 1e6 for r in rows]
    return {
        "version": version,
        "n_docs": len(rows),
        "field_accuracy": mean(per_field.values()),
        "exact_field_rate": mean(float(r["scores"][f] == 1.0) for r in rows for f in FIELDS),
        "per_field": per_field,
        "ok_rate": mean(r["status"] == "ok" for r in rows),
        "failure_rate": mean(r["status"] in ("failed", "refused") for r in rows),
        "avg_attempts": mean(r["attempts"] for r in rows),
        "retry_rate": mean(r["attempts"] > 1 for r in rows),
        "avg_latency_s": mean(r["latency_ms"] for r in rows) / 1000,
        "cost_per_doc_usd": mean(cost),
    }


def markdown_table(summaries: list[dict]) -> str:
    cols = [s["version"] for s in summaries]
    lines = ["| Metric | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]

    def row(name, fn):
        lines.append(f"| {name} | " + " | ".join(fn(s) for s in summaries) + " |")

    row("**Field-level accuracy**", lambda s: f"**{s['field_accuracy']:.1%}**")
    row("Fields exactly correct", lambda s: f"{s['exact_field_rate']:.1%}")
    for f in FIELDS:
        row(f"&nbsp;&nbsp;{f}", lambda s, f=f: f"{s['per_field'][f]:.1%}")
    row("Schema-valid & passes checks", lambda s: f"{s['ok_rate']:.0%}")
    row("Hard failures", lambda s: f"{s['failure_rate']:.0%}")
    row("Docs needing a retry", lambda s: f"{s['retry_rate']:.0%}")
    row("Avg attempts", lambda s: f"{s['avg_attempts']:.2f}")
    row("Avg latency (s)", lambda s: f"{s['avg_latency_s']:.1f}")
    row("Cost / doc (USD)", lambda s: f"${s['cost_per_doc_usd']:.3f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=list(VERSIONS), choices=VERSIONS)
    ap.add_argument("--split", choices=["dev", "test"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    labels = load_labels(args.split)
    if not labels:
        raise SystemExit(f"no labels found in {LABELS}")
    extractor = Extractor(model=args.model, effort=args.effort)

    summaries, details = [], {}
    for version in args.versions:
        with ThreadPoolExecutor(args.workers) as pool:
            records = dict(zip(labels, pool.map(
                lambda item: run_one(extractor, item[0], item[1], version, not args.no_cache), labels.items()
            )))
        rows = []
        for doc_id, rec in records.items():
            gold = CourtOpinion.model_validate(labels[doc_id]["entities"])
            pred = CourtOpinion.model_validate(rec["entities"]) if rec["entities"] else None
            if rec["status"] in ("failed", "refused"):
                pred = None
            rows.append({**rec, "doc_id": doc_id, "scores": score_document(pred, gold)})
        summaries.append(summarize(version, extractor.model, rows))
        details[version] = rows

    table = markdown_table(summaries)
    print(f"\nModel: {extractor.model} (effort={extractor.effort})   Docs: {len(labels)}   Split: {args.split or 'all'}\n")
    print(table)

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"{stamp}-{extractor.model}-{extractor.effort}-{args.split or 'all'}"
    out = RESULTS / f"{tag}.json"
    out.write_text(json.dumps(
        {"model": extractor.model, "effort": extractor.effort, "split": args.split, "summaries": summaries, "details": details},
        indent=2, default=str,
    ))
    (RESULTS / f"{tag}.md").write_text(table + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
