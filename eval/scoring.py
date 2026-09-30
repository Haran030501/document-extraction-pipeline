"""Field-level scoring of an extraction against a gold label.

Scalar fields score 1 or 0 (normalized match); list fields score F1 over matched items.
The headline "field-level accuracy" is the mean of these per-field scores over all
(document, field) pairs. A failed extraction scores 0 on every field.
"""

import re
import unicodedata

from rapidfuzz import fuzz

from app.schemas import CourtOpinion

SCALAR_FIELDS = ("case_name", "docket_number", "court", "decision_date", "author_judge", "disposition")
LIST_FIELDS = ("judges", "parties", "monetary_amounts", "cited_statutes")
FIELDS = SCALAR_FIELDS + LIST_FIELDS

FUZZY_THRESHOLD = 90


def norm(s: str | None) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.casefold()
    s = re.sub(r"\bversus\b|\bvs\.?", "v.", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def surname(name: str | None) -> str:
    n = norm(name)
    n = re.sub(r"\b(jr|sr|ii|iii|iv|chief|judge|justice|hon)\b", "", n).strip()
    return n.split()[-1] if n else ""


def statute_key(s: str) -> tuple[str | None, str]:
    """(U.S.C. title or None, base section) -- tolerant of citation style.

    '42 U.S.C. §§ 12131-34' -> ('42', '12131'); 'NRS § 41.600' -> (None, '41.600');
    'CPLR 5602(a)(1)(i)' -> (None, '5602'); '8 USC 1101(a)(43)' -> ('8', '1101').
    """
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    title = None
    m = re.match(r"\s*(\d+)\s+U\.?\s*S\.?\s*C\.?\s*A?\.?", s, re.I)
    if m:
        title, s = m.group(1), s[m.end():]
    # First section number after '§' / 'section' / code name; drop subsections and ranges.
    num = r"(\d+[a-z]?(?:[.\-]\d+[a-z]?)*)"
    m = (re.search(r"§+\s*" + num, s)
         or re.search(r"\b(?:sec(?:tion)?s?|ch(?:apter)?)\.?\s*" + num, s, re.I)
         or re.search(num, s))
    section = m.group(1) if m else norm(s)
    section = re.sub(r"-\d+$", "", section) if title else section  # USC ranges: 12131-34 -> 12131
    return title, section.casefold()


def statute_match(p: str, g: str) -> bool:
    tp, sp = statute_key(p)
    tg, sg = statute_key(g)
    return sp == sg and (tp == tg or None in (tp, tg))


def fuzzy_eq(a: str | None, b: str | None) -> bool:
    na, nb = norm(a), norm(b)
    if not na and not nb:
        return True
    if not na or not nb:
        return False
    return na == nb or fuzz.token_set_ratio(na, nb) >= FUZZY_THRESHOLD


def score_scalar(field: str, pred, gold) -> float:
    if pred is None and gold is None:
        return 1.0
    if pred is None or gold is None:
        return 0.0
    if field in ("case_name", "court"):
        return float(fuzzy_eq(pred, gold))
    if field == "author_judge":
        return float(surname(pred) == surname(gold))
    if field == "docket_number":
        return float(re.sub(r"[^\w]", "", str(pred)).casefold() == re.sub(r"[^\w]", "", str(gold)).casefold())
    return float(str(pred) == str(gold))  # decision_date, disposition


def f1(pred: list, gold: list, match) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    unmatched = list(gold)
    tp = 0
    for p in pred:
        for i, g in enumerate(unmatched):
            if match(p, g):
                tp += 1
                del unmatched[i]
                break
    precision, recall = tp / len(pred), tp / len(gold)
    return 0.0 if tp == 0 else 2 * precision * recall / (precision + recall)


def _amount_match(p, g) -> bool:
    return abs(p.value - g.value) <= max(0.01 * abs(g.value), 0.5)


LIST_MATCHERS = {
    "judges": lambda p, g: surname(p) == surname(g),
    "parties": lambda p, g: p.role == g.role and fuzzy_eq(p.name, g.name),
    "monetary_amounts": _amount_match,
    "cited_statutes": statute_match,
}


def score_document(pred: CourtOpinion | None, gold: CourtOpinion) -> dict[str, float]:
    if pred is None:
        return {f: 0.0 for f in FIELDS}
    scores = {f: score_scalar(f, getattr(pred, f), getattr(gold, f)) for f in SCALAR_FIELDS}
    for f in LIST_FIELDS:
        scores[f] = f1(getattr(pred, f), getattr(gold, f), LIST_MATCHERS[f])
    return scores
