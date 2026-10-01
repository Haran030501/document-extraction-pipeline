"""Retrieval evaluation for RAG search. Runs locally with no API calls.

    python -m scripts.index_corpus      # once, to load and index the labeled corpus
    python -m eval.run_rag_eval

Question sets (eval/rag_questions.json):
  lookup  which opinion is a question about?  -> document Recall@1/@5 and MRR
  facts   is the passage with a specific fact retrieved?  -> evidence Recall@3 / @8
  multi   are all relevant cases retrieved for a cross-case question?  -> evidence coverage@8

Evidence = a short string that must appear in a retrieved chunk of the named document.
Every evidence string is first checked against the index, so a question cannot pass or
fail because of a typo in the test set.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean

from sqlalchemy import select

from app import rag
from app.db import SessionLocal
from app.models import ChunkRow, Document

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "rag_questions.json"
LABELS = ROOT / "eval" / "labels"
RESULTS = ROOT / "eval" / "results"

CONFIGS = {
    "Keyword (no stopword filter)": {"mode": "keyword", "max_df": 1.0},
    "Keyword": {"mode": "keyword"},
    "Vector": {"mode": "vector"},
    "Hybrid + cap of 3 chunks per case": {"mode": "hybrid", "max_per_doc": 3},
    "Hybrid (RRF, default)": {"mode": "hybrid"},
}
K = 8  # chunks passed to the answer model by default


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).casefold()


def found(hits: list[rag.Hit], doc_id: int, text: str) -> bool:
    t = norm(text)
    return any(h.document_id == doc_id and t in norm(h.text) for h in hits)


def main() -> None:
    qs = json.loads(QUESTIONS.read_text())
    with SessionLocal() as session:
        id_by_file = dict(session.execute(select(Document.filename, Document.id)).all())

        def doc_pk(doc_id: str) -> int:
            pdf = json.loads((LABELS / f"{doc_id}.json").read_text())["source_pdf"]
            if pdf not in id_by_file:
                raise SystemExit(f"{pdf} is not indexed; run: python -m scripts.index_corpus")
            return id_by_file[pdf]

        # Validate every evidence string against the indexed chunks.
        for item in qs["facts"] + qs["multi"]:
            for ev in item["evidence"]:
                texts = session.scalars(select(ChunkRow.text).where(ChunkRow.document_id == doc_pk(ev["doc_id"]))).all()
                if not any(norm(ev["text"]) in norm(t) for t in texts):
                    raise SystemExit(f"evidence not in index: {ev} for {item['question']!r}")

        rows, misses = [], {}
        default_df = rag.MAX_DF
        for name, cfg in CONFIGS.items():
            rag.MAX_DF = cfg.get("max_df", default_df)
            cap = cfg.get("max_per_doc")
            retrieve = lambda q, k: rag.retrieve(session, q, k=k, mode=cfg["mode"], max_per_doc=cap)  # noqa: E731

            # lookup: rank documents by first appearance among the top 30 chunks
            ranks = []
            for item in qs["lookup"]:
                order = list(dict.fromkeys(h.document_id for h in rag.retrieve(session, item["question"], k=30, mode=cfg["mode"], max_per_doc=None)))
                t = doc_pk(item["doc_id"])
                ranks.append(order.index(t) + 1 if t in order else None)

            fact3, fact8, missed = [], [], []
            for item in qs["facts"]:
                ev = item["evidence"][0]
                hits = retrieve(item["question"], K)
                fact8.append(found(hits, doc_pk(ev["doc_id"]), ev["text"]))
                fact3.append(found(hits[:3], doc_pk(ev["doc_id"]), ev["text"]))
                if not fact8[-1]:
                    missed.append(ev["doc_id"])

            cover, complete = [], []
            for item in qs["multi"]:
                hits = retrieve(item["question"], K)
                got = [found(hits, doc_pk(ev["doc_id"]), ev["text"]) for ev in item["evidence"]]
                cover.append(mean(got))
                complete.append(all(got))
            rag.MAX_DF = default_df

            rows.append({
                "config": name,
                "lookup_r1": mean(1.0 if r == 1 else 0.0 for r in ranks),
                "lookup_r5": mean(1.0 if r and r <= 5 else 0.0 for r in ranks),
                "lookup_mrr": mean(1 / r if r else 0.0 for r in ranks),
                "fact_r3": mean(fact3), "fact_r8": mean(fact8),
                "multi_coverage": mean(cover), "multi_complete": mean(complete),
            })
            misses[name] = missed

    n = {k: len(v) for k, v in qs.items() if isinstance(v, list)}
    head = (f"| Retrieval | Case lookup R@1 ({n['lookup']}) | Lookup MRR | Fact passage R@3 ({n['facts']}) | "
            f"Fact passage R@{K} | Multi-case coverage@{K} ({n['multi']}) | Multi-case all found |")
    lines = [head, "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['lookup_r1']:.1%} | {r['lookup_mrr']:.3f} | {r['fact_r3']:.1%} | "
                     f"{r['fact_r8']:.1%} | {r['multi_coverage']:.1%} | {r['multi_complete']:.0%} |")
    table = "\n".join(lines)
    print("\n" + table + "\n\nFact passages not in top", K, ":")
    for name, m in misses.items():
        print(f"  {name}: {m}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS / f"{stamp}-rag-retrieval.json").write_text(json.dumps({"summary": rows, "fact_misses": misses}, indent=2))
    (RESULTS / f"{stamp}-rag-retrieval.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
