import json
from datetime import date, timedelta
from types import SimpleNamespace

from app.extractor import Extractor
from app.ingest import extract_text
from app.schemas import CourtOpinion, semantic_errors
from eval.scoring import FIELDS, score_document

# ---------- schema ----------


def test_schema_normalizes(gold_dict):
    gold_dict.update(docket_number="No. 21-1234", judges=["Smith", " smith ", "Jones"], case_name="  Doe  v. Acme ")
    op = CourtOpinion.model_validate(gold_dict)
    assert op.docket_number == "21-1234"
    assert CourtOpinion(docket_number="18‐3392").docket_number == "18-3392"
    assert op.judges == ["Smith", "Jones"]
    assert op.case_name == "Doe v. Acme"


def test_semantic_errors(gold_dict):
    assert semantic_errors(CourtOpinion.model_validate(gold_dict)) == []
    gold_dict.update(author_judge="Wardlaw", parties=[], decision_date=str(date.today() + timedelta(days=5)))
    errs = semantic_errors(CourtOpinion.model_validate(gold_dict))
    assert len(errs) == 3


# ---------- ingest ----------


def test_extract_text(opinion_pdf):
    r = extract_text(opinion_pdf)
    assert r.page_count == 1
    assert "21-1234" in r.text and "ACME" in r.text
    assert not r.ocr_used
    assert len(r.sha256) == 64


# ---------- scoring ----------


def test_perfect_score(gold_dict):
    gold = CourtOpinion.model_validate(gold_dict)
    assert all(v == 1.0 for v in score_document(gold, gold).values())


def test_scoring_tolerates_formatting(gold_dict):
    gold = CourtOpinion.model_validate(gold_dict)
    pred = gold.model_copy(update={
        "case_name": "Jane Doe v. ACME Corp",
        "docket_number": "21–1234".replace("–", "-"),
        "author_judge": "Milan D. Smith, Jr.",
        "judges": ["Milan D. Smith, Jr.", "Jones", "Lee"],
        "cited_statutes": ["42 USC 1983(a)"],
    })
    s = score_document(pred, gold)
    assert s["case_name"] == s["author_judge"] == s["judges"] == s["cited_statutes"] == 1.0


def test_scoring_penalizes(gold_dict):
    gold = CourtOpinion.model_validate(gold_dict)
    pred = gold.model_copy(update={
        "disposition": "reversed",
        "parties": [gold.parties[0].model_copy(update={"role": "plaintiff"}), gold.parties[1]],
        "monetary_amounts": [],
    })
    s = score_document(pred, gold)
    assert s["disposition"] == 0.0
    assert s["parties"] == 0.5
    assert s["monetary_amounts"] == 0.0
    assert score_document(None, gold) == {f: 0.0 for f in FIELDS}


# ---------- extractor (mocked Anthropic client) ----------


def _usage():
    return SimpleNamespace(input_tokens=1000, output_tokens=200)


class FakeClient:
    def __init__(self, create_texts=(), parsed=()):
        self._texts, self._parsed = list(create_texts), list(parsed)
        self.parse_calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _create(self, **kw):
        text = self._texts.pop(0)
        return SimpleNamespace(stop_reason="end_turn", usage=_usage(),
                               content=[SimpleNamespace(type="text", text=text)])

    def _parse(self, **kw):
        self.parse_calls.append(kw)
        return SimpleNamespace(stop_reason="end_turn", usage=_usage(), parsed_output=self._parsed.pop(0))


def test_v1_parses_loose_json(gold_dict):
    client = FakeClient(create_texts=["Here you go:\n```json\n" + json.dumps(gold_dict) + "\n```"])
    r = Extractor(client=client, model="m").extract("text", "v1")
    assert r.status == "ok" and r.opinion.docket_number == "21-1234"


def test_v1_fails_on_bad_json():
    r = Extractor(client=FakeClient(create_texts=["{not json"]), model="m").extract("text", "v1")
    assert r.status == "failed" and r.opinion is None


def test_v3_retries_on_semantic_error(gold_dict):
    good = CourtOpinion.model_validate(gold_dict)
    bad = good.model_copy(update={"author_judge": "Wardlaw"})
    client = FakeClient(parsed=[bad, good])
    r = Extractor(client=client, model="m").extract("text", "v3")
    assert r.status == "ok" and r.attempts == 2
    assert r.input_tokens == 2000
    retry_prompt = client.parse_calls[1]["messages"][0]["content"]
    assert "Wardlaw" in retry_prompt and "failed these validation checks" in retry_prompt


def test_v2_does_not_retry(gold_dict):
    bad = CourtOpinion.model_validate(gold_dict).model_copy(update={"author_judge": "Wardlaw"})
    r = Extractor(client=FakeClient(parsed=[bad]), model="m").extract("text", "v2")
    assert r.status == "ok" and r.attempts == 1


def test_v3_gives_up_after_max_retries(gold_dict):
    bad = CourtOpinion.model_validate(gold_dict).model_copy(update={"parties": []})
    r = Extractor(client=FakeClient(parsed=[bad, bad, bad]), model="m").extract("text", "v3")
    assert r.status == "invalid" and r.attempts == 3 and r.opinion is not None


def test_statute_matching():
    from eval.scoring import statute_match

    assert statute_match("42 U.S.C. §§ 12131–34", "42 U.S.C. § 12131")
    assert statute_match("NRS § 41.600", "Nev. Rev. Stat. § 41.600")
    assert statute_match("8 USC 1101(a)(43)(M)(i)", "8 U.S.C. § 1101")
    assert statute_match("CPLR 5602(a)(1)(i)", "CPLR 5602")
    assert statute_match("Tex. Civ. Prac. & Rem. Code § 27.005(c)", "Tex. Civ. Prac. & Rem. Code § 27.005")
    assert statute_match("15 U.S.C. § 717t-1", "15 U.S.C. § 717t-1")
    assert statute_match("Vt. Stat. Ann. tit. 9, § 2451a", "9 V.S.A. § 2451a(1)")
    assert not statute_match("42 U.S.C. § 1983", "28 U.S.C. § 1983")
    assert not statute_match("Tex. Civ. Prac. & Rem. Code § 27.005", "Tex. Civ. Prac. & Rem. Code § 27.009")
