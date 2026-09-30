"""Download a varied sample of public court opinion PDFs from CourtListener.

    python -m scripts.fetch_opinions --per-court 3 --max-pages 25

Writes PDFs to data/pdfs/ and CourtListener metadata to data/manifest.json.
Set COURTLISTENER_TOKEN for higher rate limits (optional).
"""

import argparse
import io
import json
import os
import re
import time
from pathlib import Path

import httpx
import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
PDFS = ROOT / "data" / "pdfs"
MANIFEST = ROOT / "data" / "manifest.json"
API = "https://www.courtlistener.com/api/rest/v4/search/"
STORAGE = "https://storage.courtlistener.com/"

# Federal appellate, federal district, and state high courts; query biases toward entity-rich opinions.
COURTS = {
    "ca1": "damages", "ca2": "damages", "ca3": "restitution", "ca4": "judgment",
    "ca5": "damages", "ca6": "verdict", "ca7": "damages", "ca8": "sentence",
    "ca9": "damages", "ca10": "judgment", "ca11": "damages", "cadc": "penalty",
    "cal": "damages", "tex": "judgment", "ny": "damages", "fla": "judgment",
}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-court", type=int, default=2)
    ap.add_argument("--max-pages", type=int, default=25)
    ap.add_argument("--min-pages", type=int, default=3)
    ap.add_argument("--filed-after", default="2018-01-01")
    args = ap.parse_args()

    headers = {"User-Agent": "document-extraction-pipeline/1.0 (research project)"}
    if token := os.environ.get("COURTLISTENER_TOKEN"):
        headers["Authorization"] = f"Token {token}"
    PDFS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

    with httpx.Client(headers=headers, timeout=120, follow_redirects=True) as http:
        for court, query in COURTS.items():
            have = sum(1 for m in manifest.values() if m["court_id"] == court)
            if have >= args.per_court:
                continue
            params = {"type": "o", "court": court, "q": query, "stat_Published": "on",
                      "filed_after": args.filed_after, "order_by": "score desc"}
            time.sleep(5)  # the search API rate-limits aggressively without a token
            try:
                results = http.get(API, params=params).raise_for_status().json()["results"]
            except httpx.HTTPError as e:
                print(f"[{court}] search failed: {e}")
                continue
            for r in results:
                if have >= args.per_court:
                    break
                op = next((o for o in r["opinions"] if (o.get("local_path") or "").endswith(".pdf")), None)
                if not op:
                    continue
                doc_id = f"{court}_{slug(r['docketNumber'] or str(r['cluster_id']))}"
                if doc_id in manifest:
                    continue
                try:
                    pdf = http.get(STORAGE + op["local_path"]).raise_for_status().content
                    with pdfplumber.open(io.BytesIO(pdf)) as p:
                        n_pages = len(p.pages)
                except Exception as e:
                    print(f"[{court}] download failed for {doc_id}: {e}")
                    continue
                if not args.min_pages <= n_pages <= args.max_pages:
                    continue
                (PDFS / f"{doc_id}.pdf").write_bytes(pdf)
                manifest[doc_id] = {
                    "court_id": court, "court": r["court"], "case_name": r["caseName"],
                    "docket_number": r["docketNumber"], "date_filed": r["dateFiled"],
                    "pages": n_pages, "source_url": op.get("download_url"),
                    "courtlistener_url": "https://www.courtlistener.com" + r["absolute_url"],
                }
                have += 1
                print(f"[{court}] saved {doc_id} ({n_pages} pages)")
                MANIFEST.write_text(json.dumps(manifest, indent=2))
                time.sleep(1)
    print(f"{len(manifest)} documents in manifest")


if __name__ == "__main__":
    main()
