# Document Entity Extraction Pipeline

Ingests court-opinion PDFs, extracts text (with OCR fallback for scanned pages), uses Claude with a
pydantic schema to extract structured entities, and stores them in PostgreSQL behind a FastAPI service.
An evaluation harness measures field-level accuracy against hand-labeled opinions across prompt versions.

**Stack:** Python 3.13 · FastAPI · Claude API (Opus 5.5 / Sonnet 5.5, structured outputs) · pydantic v2 ·
SQLAlchemy 2 / PostgreSQL + pgvector · fastembed · pdfplumber + Tesseract OCR · Docker

```
          ┌──────────┐    ┌──────────────────────┐    ┌───────────────────────────┐    ┌────────────┐
PDF ────▶ │ ingest   │──▶ │ extractor            │──▶ │ validate                  │──▶ │ PostgreSQL │
upload    │ pdfplumber│   │ Claude + pydantic    │    │ schema + business rules   │    │ documents  │
          │ ↳ OCR if │    │ schema (structured   │    │ (dates, author ∈ judges,  │    │ extractions│
          │   no text│    │ outputs)             │◀───│  parties, docket, amounts)│    │ parties …  │
          └──────────┘    └──────────────────────┘    └── retry with errors ──────┘    └────────────┘
```

## What gets extracted

`app/schemas.py` defines `CourtOpinion`:

| Field | Type | Notes |
|---|---|---|
| case_name, docket_number, court | str | docket normalized (strips "No.") |
| decision_date | date | filed/decided, not argued |
| judges, author_judge | list[str], str | author null for per curiam |
| parties | list[{name, role}] | role in *this* court (appellant/appellee/petitioner/…) |
| disposition | enum | affirmed · reversed · vacated · remanded · affirmed_in_part · dismissed · other |
| monetary_amounts | list[{value, currency, context}] | only amounts at issue in this case |
| cited_statutes | list[str] | statutes only, not cases, rules or regulations |

## Extraction strategies (prompt versions)

| Version | What changes |
|---|---|
| **v1** baseline | Short prompt listing the fields and allowed values; free-form JSON parsed with `json.loads`; no schema enforcement, guidelines, or retries |
| **v2** | API-enforced structured output (`messages.parse(output_format=CourtOpinion)`), detailed field-by-field guidelines, explicit "null over guessing" |
| **v3** | v2 + few-shot edge cases + **semantic validation with corrective retries**: if the output fails business rules (future date, author not on panel, no parties, …) the errors are fed back and the model re-extracts (max 2 retries) |

## Evaluation

- **Dataset:** 30 published opinions from 15 courts (federal circuits, D.C. Cir., and the California,
  New York, and Texas high courts), downloaded from CourtListener (`scripts/fetch_opinions.py`).
  3 of them are rasterized, blurred, and rotated into image-only "scans" (`scripts/make_scanned.py`) so the OCR path is part of the eval.
- **Labels:** `eval/labels/*.json`, one per opinion, in the same schema as the extractor output. Each
  label was written by reading the opinion text independently of the pipeline, following the same
  field rules as the prompt. The data is split 12 dev / 18 test by hash; prompt iteration used only the dev split.
- **Scoring** (`eval/scoring.py`): scalar fields score 1/0 after normalization (fuzzy ≥90 for case name
  and court, surname match for the author, exact for date, docket, and disposition). List fields score F1 with
  item matching: parties must match name *and* role, amounts must be within 1%, and statutes match on
  title and section. **Field-level accuracy** is the mean score over every (document, field) pair.
  A failed extraction (unparseable or invalid JSON) scores 0 on every field.

```bash
python -m eval.run_eval --versions v1 v2 v3              # all 30 docs
python -m eval.run_eval --versions v1 v2 v3 --split test  # held-out only
```

### Results

<!-- RESULTS -->
**Held-out test split (18 opinions), Claude Opus 5.5, effort `high`:**

| Metric | v1 | v2 | v3 |
|---|---:|---:|---:|
| **Field-level accuracy** | **91.9%** | **97.9%** | **98.1%** |
| Fields exactly correct | 86.7% | 96.1% | 97.2% |
| &nbsp;&nbsp;case_name | 100.0% | 100.0% | 100.0% |
| &nbsp;&nbsp;docket_number | 88.9% | 100.0% | 100.0% |
| &nbsp;&nbsp;court | 100.0% | 100.0% | 100.0% |
| &nbsp;&nbsp;decision_date | 100.0% | 100.0% | 100.0% |
| &nbsp;&nbsp;author_judge | 94.4% | 94.4% | 94.4% |
| &nbsp;&nbsp;disposition | 83.3% | 100.0% | 100.0% |
| &nbsp;&nbsp;judges | 100.0% | 100.0% | 100.0% |
| &nbsp;&nbsp;parties | 92.6% | 96.9% | 97.8% |
| &nbsp;&nbsp;monetary_amounts | 89.3% | 97.2% | 97.2% |
| &nbsp;&nbsp;cited_statutes | 70.2% | 90.7% | 91.1% |
| Schema-valid & passes checks | 100% | 100% | 100% |
| Hard failures | 0% | 0% | 0% |
| Docs needing a retry | 0% | 0% | 0% |
| Avg attempts | 1.00 | 1.00 | 1.00 |
| Avg latency (s) | 8.5 | 7.1 | 6.5 |
| Cost / doc (USD) | $0.032 | $0.036 | $0.037 |

**Model comparison, v1 → v3 field-level accuracy (effort `high`):**

| Model | All 30 docs | Test split (18) | Cost / doc (v3, all docs) | Avg latency (v3, all docs) |
|---|---:|---:|---:|---:|
| Claude Opus 5.5 | 93.6% → 98.4% | 91.9% → 98.1% | $0.037 | 6.5 s |
| Claude Sonnet 5.5 | 93.7% → 97.3% | 92.4% → 96.8% | $0.016 | 2.7 s |

**Opus 5.5 vs. Sonnet 5.5, per field (v3, held-out test split, 18 opinions):**

| Field | Opus 5.5 | Sonnet 5.5 | Δ |
|---|---:|---:|---:|
| case_name, docket_number, court, decision_date, judges | 100.0% | 100.0% | — |
| author_judge | 94.4% | 94.4% | — |
| parties | 97.8% | 98.1% | +0.4 |
| monetary_amounts | 97.2% | 94.8% | −2.4 |
| disposition | 100.0% | 94.4% | −5.6 |
| cited_statutes | 91.1% | 85.7% | −5.4 |
| **Field-level accuracy** | **98.1%** | **96.8%** | **−1.3** |
| Fields exactly correct | 97.2% | 94.4% | −2.8 |
| Cost / doc | $0.037 | $0.016 | 44% of Opus |
| Avg latency | 6.5 s | 2.6 s | 2.5× faster |

The models are identical on the header fields (case name, docket, court, date and judges). Sonnet
loses accuracy on fields that need reading the whole opinion. It **misses** more statute citations (7 missed
vs. 4 for Opus; neither model added citations that weren't there). It also reports **statutory thresholds**
as amounts at issue, such as the $0.75-per-page copy cap and the $10,000 aggravated-felony threshold, which
the prompt says to exclude and Opus did. Its disposition gap is a single
document (`tex_23_0408`, a statement about a denied rehearing) where Sonnet returned null instead of
`dismissed`. **For production:** Sonnet 5.5 suits high-volume ingestion where header fields are what
matter. Opus 5.5 is worth the cost when statutes and amounts need to be complete, or Sonnet can be used
with an Opus re-check on low-confidence documents.

**How to read these numbers**

- Nearly all of the gain comes from **v1 → v2**: API-enforced structured output plus field guidelines.
  The biggest improvements were in cited statutes (70% → 91%), disposition (83% → 100%), and docket numbers (89% → 100%).
  On the test split, the field-level error rate fell from 8.1% to 1.9%.
- **v2 → v3 is within noise** at this sample size (30 documents). The semantic-validation retry loop
  **never fired** on this dataset, because every structured output already passed the business-rule checks. The
  loop is still tested (`tests/test_core.py`) as a safety net, but none of the measured gain comes from it.
- On the held-out split, Sonnet 5.5 comes within 1.3 points of Opus 5.5 (96.8% vs. 98.1%) at 44% of the cost and 2.5× the speed.
- Most remaining misses are judgment calls rather than clear errors: for example, whether a non-appellate
  co-defendant's role is `defendant` or `other`, and whether CPLR counts as a statute or a procedural rule.
- Labels: after the first run, every disagreement was checked against the source text. Two labels were
  objectively wrong (a statute the label missed, and an unexpanded section range) and were corrected. The
  corrections are recorded in each file's `adjudication_notes`. Ambiguous cases were left as originally labeled.
<!-- /RESULTS -->

## Web UI

Open **http://localhost:8000/**. You can drag in a PDF and pick a strategy (v1–v3). The PDF appears next to
the extracted case details, judges, parties, amounts and statutes, along with the model, tokens, latency and
cost of each run. Past documents are listed in the sidebar, and each one can be re-run with another strategy to
compare results side by side. The UI is plain HTML/JS in `app/static/index.html` with no build step.

## Natural-language search (RAG)

Open **http://localhost:8000/ask** (or `POST /ask`) and ask questions across every ingested opinion, such as
*"Which cases involved restitution, and how much was owed?"* Claude answers only from retrieved passages and
cites each claim as `[n]`. Each citation links to the case, court, and page it came from.

```
question ─┬─► vector search (pgvector, HNSW, cosine) ─┐
          │   bge-small-en-v1.5 local embeddings      ├─► reciprocal rank fusion ─► top-k passages ─► Claude ─► cited answer
          └─► keyword search (Postgres tsvector, GIN) ┘          ▲
              common terms dropped as stopwords        filters on extracted entities
                                                       (court, disposition, decision date)
```

- **Indexing:** each opinion is split into ~1,200-character chunks that never cross a page boundary, so
  citations point to an exact page. Chunks are embedded with the case name and court prepended for context, and
  also indexed for Postgres full-text search. Uploads are indexed automatically.
  `python -m scripts.index_corpus` loads the labeled corpus, reusing cached extractions (no API calls).
- **Embeddings run locally** (fastembed/ONNX on CPU, 65 MB model baked into the Docker image), so search
  needs no extra API key and costs nothing. Only the final answer calls Claude.
- **Keyword stopwords:** Postgres ranking has no IDF, so query terms found in more than 20% of chunks (such as
  "case" and "order" in legal text) are dropped before matching.
- **Structured + unstructured:** search can be filtered by the entities the extraction pipeline produced.
  For example, ask only about reversed Ninth Circuit opinions.
- `GET /search/semantic?q=…` returns passages without calling the LLM.

### Retrieval evaluation

`python -m eval.run_rag_eval` (local, free) runs 65 questions over the 30-opinion corpus:

- **Case lookup (30):** which opinion is the question about?
- **Fact passage (30):** is the passage containing a specific fact retrieved? Evidence strings are checked against the index.
- **Multi-case (5):** for questions spanning several cases, is every relevant case's passage retrieved?

| Retrieval | Case lookup R@1 (30) | Lookup MRR | Fact passage R@3 (30) | Fact passage R@8 | Multi-case coverage@8 (5) | Multi-case all found |
|---|---:|---:|---:|---:|---:|---:|
| Keyword (no stopword filter) | 93.3% | 0.953 | 86.7% | 96.7% | 66.7% | 60% |
| Keyword | 93.3% | 0.961 | 86.7% | 96.7% | 93.3% | 80% |
| Vector | 90.0% | 0.944 | 86.7% | 90.0% | 73.3% | 60% |
| Hybrid + cap of 3 chunks per case | 93.3% | 0.967 | 86.7% | 90.0% | 80.0% | 60% |
| Hybrid (RRF, default) | 93.3% | 0.967 | 86.7% | 96.7% | 80.0% | 60% |

What this showed:
- **Hybrid retrieval is the best default** for single-fact questions (96.7% of answer passages in the top 8).
- **The stopword filter raised keyword multi-case coverage** from 66.7% to 93.3%.
- **A per-case chunk cap made things worse.** It was meant to stop one long opinion from crowding out other
  cases, but it lowered fact recall to 90.0% and didn't help multi-case questions, so it's off by default.
- **Cross-case questions are the weak spot.** Keyword search beats hybrid there (93% vs. 80% coverage), but
  with only 5 questions the fusion wasn't re-weighted, to avoid tuning to the test set.

**Limitations:**
- **Lookup is easy at this scale.** With 30 opinions, all methods find the right case by rank 5. The fact
  and multi-case sets are the ones that tell the methods apart.
- **The questions were drafted with an AI assistant** and the same set guided the defaults above, so treat
  the numbers as indicative rather than a held-out benchmark.
- **The small embedding model misses paraphrases with no distinctive terms.** For example, "Can hotel guests
  recover damages if they got full value?" ranks the right case third, behind a gun-theft restitution opinion
  that shares generic words like "recover" and "value".

## Label review

Open **http://localhost:8000/review** to check the gold labels against their source PDFs. The sidebar
tracks progress and flags 18 priority labels: judgment calls, long opinions with many statutes, scanned
copies, and short opinions for a full read. Each flag explains what to check. For every label you can:

- **Verify** it as correct, with an optional note, or
- **Correct** fields in the form. A note citing the source text is required, and the change is recorded in the label's `adjudication_notes`.

An off-by-default toggle shows where the Opus 5.5 extraction disagrees, so the reviewer's first pass isn't
anchored on the model's answer. Reviews are written straight to `eval/labels/*.json`, so this page is for
local use and shouldn't be exposed publicly. After making corrections, rescore from the cache with no API calls:
`python -m eval.run_eval --versions v1 v2 v3 --split test`.

## API

| Method | Path | |
|---|---|---|
| POST | `/documents?prompt_version=v3` | Upload a PDF (multipart `file`). The document is deduplicated by SHA-256, then text is extracted and the entities are stored |
| GET | `/documents`, `/documents/{id}` | Document with its latest extraction |
| GET | `/documents/{id}/extractions` | Every extraction run, with tokens, latency, attempts, and errors |
| POST | `/documents/{id}/extractions?prompt_version=v2` | Re-run extraction with a different strategy |
| GET | `/search?party=&court=&disposition=&date_from=&date_to=&min_amount=` | Query over the normalized tables |
| GET | `/documents/{id}/text` | Extracted (or OCR'd) text of the document |
| POST | `/ask` | RAG question answering with cited sources (`question`, `k`, `mode`, `filters`) |
| GET | `/search/semantic?q=&mode=&k=&court=&disposition=` | Passage retrieval only, no LLM call |
| GET | `/`, `/ask`, `/review`, `/health`, `/docs` | Web UIs, health check, OpenAPI UI |

```bash
curl -F file=@data/pdfs/ca7_18_3392.pdf localhost:8000/documents
curl "localhost:8000/search?disposition=reversed&court=seventh"
```

## Running

**Docker** (verified: the image builds, OCR runs in the container, and extraction results persist to the Compose Postgres)
```bash
cp .env.example .env   # set ANTHROPIC_API_KEY
docker compose up --build
docker compose exec api python -m scripts.index_corpus   # load + index the 30 opinions for /ask (no API calls)
```

**Local**
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
sudo apt-get install tesseract-ocr poppler-utils postgresql-17-pgvector   # OCR fallback + vector search
createdb extraction && createdb extraction_test
export ANTHROPIC_API_KEY=...  DATABASE_URL=postgresql+psycopg:///extraction?host=/var/run/postgresql
python -m scripts.index_corpus    # load + index the labeled corpus for /ask
uvicorn app.api:app --reload
pytest   # no API key or model download needed; the Anthropic client and embeddings are faked
```

**Rebuild the dataset**
```bash
python -m scripts.fetch_opinions --per-court 2
python -m scripts.make_scanned ca2_22_282 cal_s239777 ca9_21_56237
```

## Layout

```
app/        config, schemas (pydantic), ingest (PDF/OCR), extractor (v1–v3), models (SQLAlchemy), api (FastAPI),
            chunking + embeddings + rag + rag_api (search and Q&A), review (label review), static/ (web UIs)
eval/       labels/ (gold), scoring.py, run_eval.py, rag_questions.json, run_rag_eval.py, results/
scripts/    fetch_opinions.py, make_scanned.py, index_corpus.py
tests/      unit tests (schema, scoring, ingest, extractor retry loop), API tests against Postgres, label-review tests
```
