from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from payrag.answer import validate_answer_contract
from payrag.normalize import redact_sensitive_values


Decision = Literal[
    "answerable",
    "needs_clarification",
    "insufficient_evidence",
    "needs_runtime_evidence",
    "out_of_scope",
]


class Conclusion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str
    evidence_chunk_ids: list[str]
    engineering_inference: bool


class OperationalError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_type: str
    message: str
    retryable: bool


class RAGAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Decision
    applicable_conditions: list[str]
    conclusions: list[Conclusion]
    missing_information: list[str]
    verification_steps: list[str]
    operational_error: OperationalError | None


@dataclass(frozen=True)
class GenerationResult:
    status: str
    answer: dict[str, Any] | None
    operational_error: dict[str, Any] | None
    model: str
    usage: dict[str, int | None]
    response_id: str | None


class OpenAIResponsesGenerator:
    def __init__(self, model: str, *, client: Any | None = None) -> None:
        if not model.strip():
            raise ValueError("A model name is required")
        self.model = model
        self.client = client

    def generate(self, context: dict[str, Any]) -> GenerationResult:
        try:
            safe_context = _redact_payload(context)
            allowed_chunk_ids = {
                evidence["chunk_id"]
                for evidence in safe_context.get("evidence", [])
            }
        except (KeyError, TypeError, AttributeError):
            return self._error(
                "invalid_context",
                "Generation context must contain evidence with chunk_id values.",
                retryable=False,
            )
        client = self.client
        if client is None:
            if not os.getenv("OPENAI_API_KEY"):
                return self._error(
                    "provider_not_configured",
                    "OPENAI_API_KEY is not configured.",
                    retryable=False,
                )
            from openai import OpenAI

            client = OpenAI()

        try:
            response = client.responses.parse(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": _system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "question": safe_context["question"],
                                "evidence": safe_context.get("evidence", []),
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                text_format=RAGAnswer,
            )
        except Exception as exc:  # Provider exceptions vary by SDK transport.
            return self._error(
                type(exc).__name__,
                str(exc)[:500],
                retryable=_looks_retryable(exc),
            )

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            refusal = _extract_refusal(response)
            return self._error(
                "provider_refusal" if refusal else "unparsed_response",
                refusal or "The provider returned no parsed answer.",
                retryable=False,
                response_id=getattr(response, "id", None),
                usage=_usage_record(getattr(response, "usage", None)),
            )

        answer = parsed.model_dump(mode="json")
        try:
            validate_answer_contract(answer, allowed_chunk_ids)
        except ValueError as exc:
            return self._error(
                "answer_contract_violation",
                str(exc),
                retryable=False,
                response_id=getattr(response, "id", None),
                usage=_usage_record(getattr(response, "usage", None)),
            )

        return GenerationResult(
            status="completed",
            answer=answer,
            operational_error=None,
            model=self.model,
            usage=_usage_record(getattr(response, "usage", None)),
            response_id=getattr(response, "id", None),
        )

    def _error(
        self,
        error_type: str,
        message: str,
        *,
        retryable: bool,
        response_id: str | None = None,
        usage: dict[str, int | None] | None = None,
    ) -> GenerationResult:
        return GenerationResult(
            status="operational_error",
            answer=None,
            operational_error={
                "error_type": error_type,
                "message": message,
                "retryable": retryable,
            },
            model=self.model,
            usage=usage or {"input_tokens": None, "output_tokens": None, "total_tokens": None},
            response_id=response_id,
        )


def write_generation_result(path: Path, result: GenerationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _redact_payload(value: Any) -> Any:
    if isinstance(value, str):
        return redact_sensitive_values(value)
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    if isinstance(value, dict):
        return {
            redact_sensitive_values(str(key)): _redact_payload(item)
            for key, item in value.items()
        }
    return value


def _system_prompt() -> str:
    return (
        "You are a read-only payment integration documentation assistant. "
        "Use only the supplied evidence. Keep unknown conditions unknown. "
        "Do not claim access to a Stripe account or runtime payment state. "
        "Every documentary conclusion must cite supplied chunk IDs. Mark any "
        "uncited engineering inference explicitly. Never request API keys, "
        "client secrets, card numbers, or other secrets. Return the requested "
        "structured answer."
    )


def _extract_refusal(response: Any) -> str | None:
    for output in getattr(response, "output", []) or []:
        for item in getattr(output, "content", []) or []:
            refusal = getattr(item, "refusal", None)
            if refusal:
                return str(refusal)
    return None


def _usage_record(usage: Any) -> dict[str, int | None]:
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _looks_retryable(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return any(
        marker in name or marker in message
        for marker in ("timeout", "ratelimit", "rate limit", "connection")
    )
