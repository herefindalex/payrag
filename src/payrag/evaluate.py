from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from payrag.retrieval import BM25Index, result_record


def evaluate_development_pilot(
    index: BM25Index,
    pilot_path: Path,
    output_path: Path,
    *,
    top_k: int = 5,
    max_per_source: int | None = None,
) -> dict[str, Any]:
    pilot = yaml.safe_load(pilot_path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []

    for case in pilot["cases"]:
        results = index.search(
            case["prompt_en"], top_k=top_k, max_per_source=max_per_source
        )
        required_sources = sorted(
            evidence["source_id"]
            for evidence in case.get("evidence", [])
            if evidence.get("role") == "required"
        )
        returned_sources = [result.source_id for result in results]
        retrieval_scored = bool(required_sources)
        hits = sorted(set(required_sources) & set(returned_sources))
        cases.append(
            {
                "case_id": case["case_id"],
                "family": case["family"],
                "expected_decision": case.get("expected_decision"),
                "retrieval_scored": retrieval_scored,
                "required_source_ids": required_sources,
                "required_source_hits": hits,
                "all_required_sources_present": (
                    set(required_sources) <= set(returned_sources)
                    if retrieval_scored
                    else None
                ),
                "results": [result_record(result) for result in results],
            }
        )

    scored_cases = [case for case in cases if case["retrieval_scored"]]
    diagnostic_hits = sum(case["all_required_sources_present"] for case in scored_cases)
    report = {
        "dataset_version": pilot["dataset_version"],
        "evaluation_kind": "development_pilot_source_level_diagnostic",
        "formal_benchmark": False,
        "top_k": top_k,
        "max_per_source": max_per_source,
        "case_count": len(cases),
        "diagnostic_scored_case_count": len(scored_cases),
        "diagnostic_all_required_sources_present_count": diagnostic_hits,
        "diagnostic_all_required_sources_present_rate": (
            diagnostic_hits / len(scored_cases) if scored_cases else None
        ),
        "limitations": [
            "The pilot is development data, not held-out gold or production support traffic.",
            "Expected evidence is checked at source level because snapshot-specific chunk binding and human gold review are pending.",
            "P12 has background sources only and is excluded from retrieval-miss diagnostics.",
            "No MRR or nDCG is reported because complete relevance and alternative-evidence labels do not yet exist.",
        ],
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
