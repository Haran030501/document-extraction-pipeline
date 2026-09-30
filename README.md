# Document Entity Extraction Pipeline

Ingests court-opinion PDFs, extracts text (with OCR fallback for scanned pages), uses Claude with a
pydantic schema to extract structured entities, and stores them in PostgreSQL behind a FastAPI service.
An evaluation harness measures field-level accuracy against hand-labeled opinions across prompt versions.

**Stack:** Python 3.13 · FastAPI · Claude API (Opus 5.5 / Sonnet 5.5, structured outputs) · pydantic v2 ·
SQLAlchemy 2 / PostgreSQL · pdfplumber + Tesseract OCR · Docker

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
- Sonnet 5.5 comes within about 1 point of Opus 5.5 at 43% of the cost and 2.4× the speed.
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

## API

| Method | Path | |
|---|---|---|
| POST | `/documents?prompt_version=v3` | Upload a PDF (multipart `file`). The document is deduplicated by SHA-256, then text is extracted and the entities are stored |
| GET | `/documents`, `/documents/{id}` | Document with its latest extraction |
| GET | `/documents/{id}/extractions` | Every extraction run, with tokens, latency, attempts, and errors |
| POST | `/documents/{id}/extractions?prompt_version=v2` | Re-run extraction with a different strategy |
| GET | `/search?party=&court=&disposition=&date_from=&date_to=&min_amount=` | Query over the normalized tables |
| GET | `/documents/{id}/text` | Extracted (or OCR'd) text of the document |
| GET | `/`, `/health`, `/docs` | Web UI, health check, OpenAPI UI |

```bash
curl -F file=@data/pdfs/ca7_18_3392.pdf localhost:8000/documents
curl "localhost:8000/search?disposition=reversed&court=seventh"
```

## Running

**Docker** (verified: the image builds, OCR runs in the container, and extraction results persist to the Compose Postgres)
```bash
cp .env.example .env   # set ANTHROPIC_API_KEY
docker compose up --build
```

**Local**
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
sudo apt-get install tesseract-ocr poppler-utils   # OCR fallback
createdb extraction && createdb extraction_test
export ANTHROPIC_API_KEY=...  DATABASE_URL=postgresql+psycopg:///extraction?host=/var/run/postgresql
uvicorn app.api:app --reload
pytest   # no API key needed; the Anthropic client is mocked
```

**Rebuild the dataset**
```bash
python -m scripts.fetch_opinions --per-court 2
python -m scripts.make_scanned ca2_22_282 cal_s239777 ca9_21_56237
```

## Layout

```
app/        config, schemas (pydantic), ingest (PDF/OCR), extractor (v1–v3), models (SQLAlchemy), api (FastAPI)
eval/       labels/ (gold), scoring.py, run_eval.py, results/
scripts/    fetch_opinions.py, make_scanned.py
tests/      unit tests (schema, scoring, ingest, extractor retry loop) + API tests against Postgres
```
