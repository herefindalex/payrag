from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from payrag.answer import prepare_retrieval_context, write_context
from payrag.acquire import acquire_sources, summarize_results
from payrag.dense import DenseIndex
from payrag.expansion import probe_expansions
from payrag.evaluate import evaluate_development_pilot
from payrag.generation import OpenAIResponsesGenerator, write_generation_result
from payrag.manifest import ManifestError, load_manifest
from payrag.normalize import normalize_snapshot
from payrag.retrieval import BM25Index, result_record
from payrag.validation import validate_normalization


DEFAULT_MANIFEST = Path("config/source-manifest.json")
DEFAULT_RAW_ROOT = Path("data/raw")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="payrag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    acquire.add_argument("--output", type=Path, default=DEFAULT_RAW_ROOT)
    acquire.add_argument("--source", action="append", dest="source_ids")
    acquire.add_argument("--timeout", type=float, default=30.0)
    acquire.add_argument("--attempts", type=int, default=2)

    normalize = subparsers.add_parser("normalize")
    normalize.add_argument("snapshot", type=Path)
    normalize.add_argument("--output", type=Path)
    normalize.add_argument("--max-characters", type=int, default=6000)

    expansions = subparsers.add_parser("probe-expansions")
    expansions.add_argument("snapshot", type=Path)
    expansions.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    expansions.add_argument("--timeout", type=float, default=30.0)
    expansions.add_argument("--attempts", type=int, default=2)

    validation = subparsers.add_parser("validate-normalization")
    validation.add_argument("snapshot", type=Path)
    validation.add_argument("normalized", type=Path)

    build_bm25 = subparsers.add_parser("build-bm25")
    build_bm25.add_argument("chunks", type=Path)
    build_bm25.add_argument("--output", type=Path, required=True)
    build_bm25.add_argument("--snapshot-id")

    search_bm25 = subparsers.add_parser("search-bm25")
    search_bm25.add_argument("index", type=Path)
    search_bm25.add_argument("query")
    search_bm25.add_argument("--top-k", type=int, default=5)
    search_bm25.add_argument("--max-per-source", type=int)
    search_bm25.add_argument("--include-text", action="store_true")

    pilot = subparsers.add_parser("evaluate-pilot")
    pilot.add_argument("index", type=Path)
    pilot.add_argument("--pilot", type=Path, default=Path("config/retrieval-pilot.yaml"))
    pilot.add_argument("--output", type=Path, required=True)
    pilot.add_argument("--top-k", type=int, default=5)
    pilot.add_argument("--max-per-source", type=int)

    context = subparsers.add_parser("prepare-context")
    context.add_argument("index", type=Path)
    context.add_argument("query")
    context.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    context.add_argument("--output", type=Path, required=True)
    context.add_argument("--top-k", type=int, default=5)
    context.add_argument("--max-per-source", type=int)
    context.add_argument("--retriever", choices=("bm25", "dense"), default="bm25")
    context.add_argument("--device", default="cpu")

    build_dense = subparsers.add_parser("build-dense")
    build_dense.add_argument("chunks", type=Path)
    build_dense.add_argument("--output", type=Path, required=True)
    build_dense.add_argument(
        "--model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    build_dense.add_argument("--revision")
    build_dense.add_argument("--device", default="cpu")
    build_dense.add_argument("--batch-size", type=int, default=32)
    build_dense.add_argument("--snapshot-id")

    search_dense = subparsers.add_parser("search-dense")
    search_dense.add_argument("index", type=Path)
    search_dense.add_argument("query")
    search_dense.add_argument("--device", default="cpu")
    search_dense.add_argument("--top-k", type=int, default=5)
    search_dense.add_argument("--max-per-source", type=int)

    dense_pilot = subparsers.add_parser("evaluate-pilot-dense")
    dense_pilot.add_argument("index", type=Path)
    dense_pilot.add_argument(
        "--pilot", type=Path, default=Path("config/retrieval-pilot.yaml")
    )
    dense_pilot.add_argument("--output", type=Path, required=True)
    dense_pilot.add_argument("--device", default="cpu")
    dense_pilot.add_argument("--top-k", type=int, default=5)
    dense_pilot.add_argument("--max-per-source", type=int)

    generate = subparsers.add_parser("generate-openai")
    generate.add_argument("context", type=Path)
    generate.add_argument("--model", required=True)
    generate.add_argument("--output", type=Path, required=True)

    serve = subparsers.add_parser("serve-api")
    serve.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    serve.add_argument("--bm25-index", type=Path)
    serve.add_argument("--dense-index", type=Path)
    serve.add_argument("--dense-device", default="cpu")
    serve.add_argument("--generation-model")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "normalize":
        output = args.output or Path("data/normalized") / args.snapshot.name
        report = normalize_snapshot(
            args.snapshot,
            output,
            max_characters=args.max_characters,
        )
        print(json.dumps(report, ensure_ascii=False))
        return 1 if report["failure_count"] else 0

    if args.command == "validate-normalization":
        report = validate_normalization(args.snapshot, args.normalized)
        print(
            json.dumps(
                {
                    "snapshot_id": report["snapshot_id"],
                    "passed": report["passed"],
                    "passed_count": report["passed_count"],
                    "failed_count": report["failed_count"],
                }
            )
        )
        return 0 if report["passed"] else 1

    if args.command == "build-bm25":
        index = BM25Index.from_chunks_jsonl(args.chunks)
        index.save(args.output, snapshot_id=args.snapshot_id)
        print(
            json.dumps(
                {
                    "status": "built",
                    "index": str(args.output),
                    "chunk_count": len(index.chunks),
                    "average_document_length": round(index.average_document_length, 2),
                }
            )
        )
        return 0

    if args.command == "search-bm25":
        index = BM25Index.load(args.index)
        results = index.search(
            args.query,
            top_k=args.top_k,
            max_per_source=args.max_per_source,
        )
        print(
            json.dumps(
                [
                    result_record(result, include_text=args.include_text)
                    for result in results
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "evaluate-pilot":
        index = BM25Index.load(args.index)
        report = evaluate_development_pilot(
            index,
            args.pilot,
            args.output,
            top_k=args.top_k,
            max_per_source=args.max_per_source,
        )
        print(
            json.dumps(
                {
                    "evaluation_kind": report["evaluation_kind"],
                    "formal_benchmark": report["formal_benchmark"],
                    "case_count": report["case_count"],
                    "diagnostic_scored_case_count": report[
                        "diagnostic_scored_case_count"
                    ],
                    "diagnostic_all_required_sources_present_count": report[
                        "diagnostic_all_required_sources_present_count"
                    ],
                }
            )
        )
        return 0

    if args.command == "prepare-context":
        manifest = load_manifest(args.manifest)
        index = (
            BM25Index.load(args.index)
            if args.retriever == "bm25"
            else DenseIndex.load(args.index, device=args.device)
        )
        context = prepare_retrieval_context(
            index,
            manifest,
            args.query,
            top_k=args.top_k,
            max_per_source=args.max_per_source,
            retriever_kind=args.retriever,
        )
        write_context(args.output, context)
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "output": str(args.output),
                    "evidence_count": len(context["evidence"]),
                    "generation_status": context["generation_status"],
                }
            )
        )
        return 0

    if args.command == "build-dense":
        index = DenseIndex.build(
            args.chunks,
            model_name=args.model,
            revision=args.revision,
            device=args.device,
            batch_size=args.batch_size,
        )
        index.save(args.output, snapshot_id=args.snapshot_id)
        print(
            json.dumps(
                {
                    "status": "built",
                    "index": str(args.output),
                    "model": index.metadata["model_name"],
                    "model_revision": index.metadata["model_revision"],
                    "input_chunk_count": index.metadata["input_chunk_count"],
                    "chunk_count": index.metadata["chunk_count"],
                    "embedding_dimension": index.metadata["embedding_dimension"],
                    "max_sequence_length": index.metadata["max_sequence_length"],
                    "truncated_chunk_count": index.metadata[
                        "truncated_chunk_count"
                    ],
                }
            )
        )
        return 0

    if args.command == "search-dense":
        index = DenseIndex.load(args.index, device=args.device)
        results = index.search(
            args.query,
            top_k=args.top_k,
            max_per_source=args.max_per_source,
        )
        print(
            json.dumps(
                [result_record(result) for result in results],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "evaluate-pilot-dense":
        index = DenseIndex.load(args.index, device=args.device)
        report = evaluate_development_pilot(
            index,
            args.pilot,
            args.output,
            top_k=args.top_k,
            max_per_source=args.max_per_source,
        )
        report["retriever"] = {
            "kind": "dense_cosine",
            "model_name": index.metadata["model_name"],
            "model_revision": index.metadata.get("model_revision"),
            "max_sequence_length": index.metadata["max_sequence_length"],
            "truncated_chunk_count": index.metadata["truncated_chunk_count"],
        }
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "evaluation_kind": report["evaluation_kind"],
                    "formal_benchmark": report["formal_benchmark"],
                    "model_name": index.metadata["model_name"],
                    "diagnostic_scored_case_count": report[
                        "diagnostic_scored_case_count"
                    ],
                    "diagnostic_all_required_sources_present_count": report[
                        "diagnostic_all_required_sources_present_count"
                    ],
                }
            )
        )
        return 0

    if args.command == "generate-openai":
        context = json.loads(args.context.read_text(encoding="utf-8"))
        result = OpenAIResponsesGenerator(args.model).generate(context)
        write_generation_result(args.output, result)
        print(
            json.dumps(
                {
                    "status": result.status,
                    "model": result.model,
                    "output": str(args.output),
                    "operational_error": result.operational_error,
                },
                ensure_ascii=False,
            )
        )
        return 0 if result.status == "completed" else 1

    if args.command == "serve-api":
        import uvicorn

        from payrag.api import ApiSettings, create_app

        app = create_app(
            ApiSettings(
                manifest_path=args.manifest,
                bm25_index_path=args.bm25_index,
                dense_index_path=args.dense_index,
                dense_device=args.dense_device,
                generation_model=args.generation_model,
            )
        )
        uvicorn.run(app, host=args.host, port=args.port)
        return 0


    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}), file=sys.stderr)
        return 2

    if args.command == "validate-manifest":
        print(
            json.dumps(
                {
                    "status": "valid",
                    "project": manifest["project"],
                    "version": manifest["version"],
                    "source_count": len(manifest["sources"]),
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "probe-expansions":
        report = probe_expansions(
            manifest,
            args.snapshot,
            timeout_seconds=args.timeout,
            attempts=max(1, args.attempts),
        )
        print(
            json.dumps(
                {
                    "snapshot_id": report["snapshot_id"],
                    "probe_count": report["probe_count"],
                    "acquired_count": report["acquired_count"],
                    "failed_count": report["failed_count"],
                }
            )
        )
        return 1 if report["failed_count"] else 0

    known_ids = {source["source_id"] for source in manifest["sources"]}
    requested_ids = set(args.source_ids) if args.source_ids else None
    unknown_ids = requested_ids - known_ids if requested_ids else set()
    if unknown_ids:
        print(
            json.dumps({"status": "invalid", "unknown_source_ids": sorted(unknown_ids)}),
            file=sys.stderr,
        )
        return 2

    run_dir, results = acquire_sources(
        manifest,
        args.output,
        source_ids=requested_ids,
        timeout_seconds=args.timeout,
        attempts=max(1, args.attempts),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "snapshot_dir": str(run_dir),
                "summary": summarize_results(results),
            },
            ensure_ascii=False,
        )
    )
    return 1 if any(result.status == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
