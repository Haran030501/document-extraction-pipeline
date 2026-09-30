"""LLM entity extraction with three prompt/validation strategies.

v1  baseline: minimal prompt, free-form JSON parsed with json.loads, no retries.
v2  structured outputs (pydantic schema enforced by the API) + detailed field guidelines.
v3  v2 + few-shot examples + semantic validation with corrective retries.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from pydantic import ValidationError

from app.config import get_settings
from app.schemas import CourtOpinion, semantic_errors

log = logging.getLogger(__name__)

PROMPTS = Path(__file__).parent / "prompts"
VERSIONS = ("v1", "v2", "v3")
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class ExtractionResult:
    version: str
    model: str
    opinion: CourtOpinion | None = None
    status: str = "failed"  # ok | invalid | failed | refused
    attempts: int = 0
    errors: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0

    @property
    def raw_json(self) -> dict | None:
        return self.opinion.model_dump(mode="json") if self.opinion else None


def _prompt(name: str) -> str:
    return (PROMPTS / name).read_text()


def _document_message(text: str, feedback: str | None = None) -> list[dict]:
    content = f"<opinion>\n{text}\n</opinion>"
    if feedback:
        content += "\n\n" + feedback
    return [{"role": "user", "content": content}]


def _parse_loose_json(text: str) -> dict:
    """Pull the first JSON object out of a free-form response (for the v1 baseline)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in response")
    return json.loads(m.group(0))


class Extractor:
    def __init__(
        self, client: anthropic.Anthropic | None = None, model: str | None = None, effort: str | None = None
    ):
        settings = get_settings()
        self.client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = model or settings.model
        self.effort = effort or settings.effort
        self.max_retries = settings.max_retries

    def extract(self, text: str, version: str = "v3") -> ExtractionResult:
        if version not in VERSIONS:
            raise ValueError(f"unknown prompt version {version!r}")
        result = ExtractionResult(version=version, model=self.model)
        start = time.monotonic()
        try:
            if version == "v1":
                self._extract_v1(text, result)
            else:
                self._extract_structured(text, result, retries=self.max_retries if version == "v3" else 0)
        except anthropic.APIError as e:
            result.status = "failed"
            result.errors.append(f"API error: {e}")
        result.latency_ms = int((time.monotonic() - start) * 1000)
        return result

    def _record_usage(self, response, result: ExtractionResult) -> None:
        result.attempts += 1
        result.input_tokens += response.usage.input_tokens
        result.output_tokens += response.usage.output_tokens

    def _extract_v1(self, text: str, result: ExtractionResult) -> None:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            output_config={"effort": self.effort},
            system=_prompt("v1_baseline.txt"),
            messages=_document_message(text),
        )
        self._record_usage(response, result)
        if response.stop_reason == "refusal":
            result.status = "refused"
            return
        body = "".join(b.text for b in response.content if b.type == "text")
        try:
            result.opinion = CourtOpinion.model_validate(_parse_loose_json(body))
            result.status = "ok"
        except (ValueError, ValidationError) as e:
            result.status = "failed"
            result.errors.append(f"parse/validation: {e}")

    def _extract_structured(self, text: str, result: ExtractionResult, retries: int) -> None:
        system = _prompt("v2_guidelines.txt")
        if result.version == "v3":
            system += "\n" + _prompt("v3_fewshot.txt")
        feedback = None
        for _ in range(retries + 1):
            try:
                response = self.client.beta.messages.parse(
                    model=self.model,
                    max_tokens=16000,
                    thinking={"type": "adaptive"},
                    system=system,
                    messages=_document_message(text, feedback),
                    output_format=CourtOpinion,
                    output_config={"effort": self.effort},
                    betas=[FALLBACK_BETA],
                    fallbacks="default",
                )
            except ValidationError as e:
                # Output matched the JSON schema but failed a pydantic validator.
                result.attempts += 1
                result.errors.append(f"schema validation: {e}")
                feedback = f"Your previous answer failed schema validation:\n{e}\nReturn a corrected extraction."
                continue
            self._record_usage(response, result)
            if response.stop_reason == "refusal":
                result.status = "refused"
                return
            if response.stop_reason == "max_tokens" or response.parsed_output is None:
                result.errors.append(f"no parsed output (stop_reason={response.stop_reason})")
                continue
            opinion = response.parsed_output
            result.opinion = opinion
            problems = semantic_errors(opinion) if result.version == "v3" else []
            if not problems:
                result.status = "ok"
                return
            result.status = "invalid"
            result.errors.extend(problems)
            feedback = (
                "A previous extraction of this opinion was:\n"
                f"{opinion.model_dump_json(indent=2)}\n\n"
                "It failed these validation checks:\n- " + "\n- ".join(problems) +
                "\n\nRe-read the opinion carefully and return a corrected extraction. "
                "If a check fails because the opinion genuinely lacks the information, leave that field null/empty."
            )
        if result.status != "ok" and result.opinion is not None:
            # Keep the best-effort answer; status stays 'invalid'.
            log.info("extraction still invalid after %d attempts: %s", result.attempts, result.errors[-3:])
