from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from payrag.answer import AnswerContractError, prepare_retrieval_context, validate_answer_contract
from payrag.dense import DenseIndex
from payrag.generation import OpenAIResponsesGenerator
from payrag.manifest import load_manifest
from payrag.normalize import redact_sensitive_values
from payrag.retrieval import BM25Index


@dataclass(frozen=True)
class ApiSettings:
    manifest_path: Path
    bm25_index_path: Path | None = None
    dense_index_path: Path | None = None
    dense_device: str = "cpu"
    generation_model: str | None = None


class ContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=10_000)
    retriever: Literal["bm25", "dense"] = "bm25"
    top_k: int = Field(default=5, ge=1, le=20)
    max_per_source: int | None = Field(default=None, ge=1, le=20)


class ValidateAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: dict[str, Any]
    allowed_chunk_ids: list[str]


class GenerateRequest(ContextRequest):
    pass


class _Runtime:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.manifest = load_manifest(settings.manifest_path)
        self.bm25 = (
            BM25Index.load(settings.bm25_index_path)
            if settings.bm25_index_path
            else None
        )
        self._dense: DenseIndex | None = None

    def index(self, kind: str) -> Any:
        if kind == "bm25":
            if self.bm25 is None:
                raise RuntimeError("BM25 index is not configured")
            return self.bm25
        if self.settings.dense_index_path is None:
            raise RuntimeError("Dense index is not configured")
        if self._dense is None:
            self._dense = DenseIndex.load(
                self.settings.dense_index_path,
                device=self.settings.dense_device,
            )
        return self._dense


def create_app(settings: ApiSettings) -> FastAPI:
    runtime = _Runtime(settings)
    app = FastAPI(title="Payment Integration Support RAG", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "retrievers": {
                "bm25": settings.bm25_index_path is not None,
                "dense": settings.dense_index_path is not None,
            },
            "generation": {
                "model_configured": settings.generation_model is not None,
                "provider_key_configured": bool(os.getenv("OPENAI_API_KEY")),
            },
        }

    @app.post("/v1/context", response_model=None)
    def context(request: ContextRequest) -> Any:
        redacted_question = redact_sensitive_values(request.question)
        try:
            index = runtime.index(request.retriever)
            payload = prepare_retrieval_context(
                index,
                runtime.manifest,
                redacted_question,
                top_k=request.top_k,
                max_per_source=request.max_per_source,
                retriever_kind=request.retriever,
            )
            payload["input_redacted"] = redacted_question != request.question
            return payload
        except Exception as exc:
            return _operational_error_response(exc)

    @app.post("/v1/answers/validate", response_model=None)
    def validate_answer(request: ValidateAnswerRequest) -> Any:
        try:
            validate_answer_contract(request.answer, set(request.allowed_chunk_ids))
        except AnswerContractError as exc:
            return JSONResponse(
                status_code=422,
                content={
                    "status": "invalid",
                    "error": str(exc),
                },
            )
        return {"status": "valid"}

    @app.post("/v1/generate", response_model=None)
    def generate(request: GenerateRequest) -> Any:
        if settings.generation_model is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "operational_error",
                    "operational_error": {
                        "error_type": "provider_not_configured",
                        "message": "No generation model is configured.",
                        "retryable": False,
                    },
                },
            )
        redacted_question = redact_sensitive_values(request.question)
        try:
            index = runtime.index(request.retriever)
            generation_context = prepare_retrieval_context(
                index,
                runtime.manifest,
                redacted_question,
                top_k=request.top_k,
                max_per_source=request.max_per_source,
                retriever_kind=request.retriever,
            )
            generation_context["input_redacted"] = (
                redacted_question != request.question
            )
        except Exception as exc:
            return _operational_error_response(exc)
        result = OpenAIResponsesGenerator(settings.generation_model).generate(
            generation_context
        )
        status_code = 200 if result.status == "completed" else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "status": result.status,
                "answer": result.answer,
                "operational_error": result.operational_error,
                "model": result.model,
                "usage": result.usage,
                "response_id": result.response_id,
            },
        )

    return app


def _operational_error_response(exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "status": "operational_error",
            "operational_error": {
                "error_type": type(exc).__name__,
                "message": str(exc)[:500],
                "retryable": False,
            },
        },
    )
