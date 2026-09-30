"""Pydantic schema for entities extracted from a court opinion.

The same model is used as the LLM structured-output format, the API response
shape, and the gold-label format for evaluation.
"""

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

PartyRole = Literal[
    "plaintiff", "defendant", "appellant", "appellee",
    "petitioner", "respondent", "other",
]
Disposition = Literal[
    "affirmed", "reversed", "remanded", "affirmed_in_part",
    "dismissed", "vacated", "other",
]


def _clean(s: str | None) -> str | None:
    if s is None:
        return None
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


class Party(BaseModel):
    name: str = Field(description="Party name as written, without role words like 'Plaintiff-Appellant'.")
    role: PartyRole = Field(description="Party's role in THIS proceeding (e.g. appellant/appellee on appeal).")

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _clean(v) or v


class Amount(BaseModel):
    value: float = Field(description="Numeric amount, e.g. 1250000.00 for '$1.25 million'.")
    currency: str = Field(default="USD", description="ISO currency code.")
    context: str = Field(description="Short description of what the amount is (e.g. 'jury verdict', 'restitution').")


class CourtOpinion(BaseModel):
    case_name: str | None = Field(None, description="Short case caption, e.g. 'Smith v. Jones'.")
    docket_number: str | None = Field(None, description="Primary docket/case number exactly as printed, e.g. '21-1234'.")
    court: str | None = Field(None, description="Full name of the court that issued the opinion.")
    decision_date: date | None = Field(None, description="Date the opinion was decided/filed (not argued).")
    judges: list[str] = Field(default_factory=list, description="Surnames or full names of judges on the panel.")
    author_judge: str | None = Field(None, description="Judge who authored the majority opinion; null if per curiam.")
    parties: list[Party] = Field(default_factory=list)
    disposition: Disposition | None = Field(None, description="Outcome for the judgment under review.")
    monetary_amounts: list[Amount] = Field(
        default_factory=list,
        description="Dollar amounts at issue in THIS case (damages, fines, awards). Exclude amounts from cited cases.",
    )
    cited_statutes: list[str] = Field(
        default_factory=list,
        description="Statutes cited, normalized like '42 U.S.C. § 1983'. Exclude case-law citations and rules.",
    )

    @field_validator("case_name", "docket_number", "court", "author_judge")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        return _clean(v)

    @field_validator("docket_number")
    @classmethod
    def _docket(cls, v: str | None) -> str | None:
        if v is None:
            return None
        # "No. 21-1234" / "Docket No. 21-1234" -> "21-1234"; PDF text often uses Unicode dashes.
        v = re.sub(r"[‐-―−]", "-", v)
        return re.sub(r"^(docket\s+)?(nos?\.?|number)\s*", "", v, flags=re.I).strip()

    @field_validator("judges", "cited_statutes")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        seen, out = set(), []
        for item in v:
            c = _clean(item)
            if c and c.casefold() not in seen:
                seen.add(c.casefold())
                out.append(c)
        return out


def semantic_errors(op: CourtOpinion) -> list[str]:
    """Business-rule checks beyond the JSON schema. Used by the v3 retry loop."""
    errors: list[str] = []
    if op.decision_date is not None:
        if op.decision_date > date.today():
            errors.append(f"decision_date {op.decision_date} is in the future.")
        if op.decision_date.year < 1900:
            errors.append(f"decision_date {op.decision_date} is implausibly old.")
    if op.author_judge and op.judges:
        surnames = {j.split()[-1].casefold().strip(",.") for j in op.judges}
        if op.author_judge.split()[-1].casefold().strip(",.") not in surnames:
            errors.append(f"author_judge '{op.author_judge}' is not in judges list {op.judges}.")
    if not op.parties:
        errors.append("parties is empty; every opinion has at least two parties.")
    if op.case_name is None:
        errors.append("case_name is missing.")
    if op.docket_number and not re.search(r"\d", op.docket_number):
        errors.append(f"docket_number '{op.docket_number}' contains no digits.")
    for a in op.monetary_amounts:
        if a.value <= 0:
            errors.append(f"monetary amount {a.value} ({a.context}) must be positive.")
    return errors
