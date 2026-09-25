"""Benchmark the local book index with deterministic, auditable metrics."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_app.runtime import (  # noqa: E402
    LightRAG,
    QueryParam,
    configure_console,
    default_llm_model,
    make_embedding_func,
    make_embedding_rerank_func,
    make_llm_func,
    make_local_tokenizer,
)
from lightrag.utils import TokenTracker  # noqa: E402
from rag_app.agent import RetrievalAgent  # noqa: E402
from rag_app.agent_types import AgentSettings  # noqa: E402
from rag_app.telemetry import ACTIVE_USAGE, RequestUsage  # noqa: E402
from rag_app.lifecycle import finalize_rag  # noqa: E402


from rag_app.config import INDEX_DIR, OUTPUT_DIR, WORKSPACE, project_path  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "data" / "book_eval_dataset.json"
DEFAULT_INDEX = INDEX_DIR
DEFAULT_OUTPUT = OUTPUT_DIR
ALL_STRATEGIES = (
    "agent",
    "naive",
    "local",
    "global",
    "hybrid",
    "mix",
    "bypass",
    "hybrid+rerank",
)


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_keywords: tuple[str, ...]
    expected_source: str


@dataclass(frozen=True)
class EvaluationStrategy:
    name: str
    mode: str
    enable_rerank: bool


def load_cases(path: Path) -> list[EvaluationCase]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    raw_cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"Evaluation dataset must contain a non-empty 'cases' list: {path}")

    cases = []
    seen_ids = set()
    for index, item in enumerate(raw_cases, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Case {index} must be a JSON object")
        case_id = str(item.get("id", "")).strip()
        question = str(item.get("question", "")).strip()
        expected_source = str(item.get("expected_source", "")).strip()
        raw_keywords = item.get("expected_keywords")
        if not case_id or case_id in seen_ids:
            raise ValueError(f"Case {index} has a missing or duplicate id")
        if not question or not expected_source:
            raise ValueError(f"Case {case_id} must define question and expected_source")
        if not isinstance(raw_keywords, list) or not raw_keywords:
            raise ValueError(f"Case {case_id} must define expected_keywords")
        keywords = tuple(str(keyword).strip() for keyword in raw_keywords)
        if any(not keyword for keyword in keywords):
            raise ValueError(f"Case {case_id} contains a blank expected keyword")
        seen_ids.add(case_id)
        cases.append(
            EvaluationCase(
                case_id=case_id,
                question=question,
                expected_keywords=keywords,
                expected_source=expected_source,
            )
        )
    return cases


def parse_strategies(value: str) -> list[EvaluationStrategy]:
    names = list(ALL_STRATEGIES) if value.strip().lower() == "all" else [
        item.strip().lower() for item in value.split(",") if item.strip()
    ]
    if not names:
        raise ValueError("At least one evaluation strategy is required")

    strategies = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        enable_rerank = name.endswith("+rerank")
        mode = name.removesuffix("+rerank")
        if mode not in {"agent", "naive", "local", "global", "hybrid", "mix", "bypass"}:
            raise ValueError(f"Unknown evaluation strategy: {name}")
        if mode == "bypass" and enable_rerank:
            raise ValueError("bypass+rerank is not meaningful because bypass skips retrieval")
        if mode == "agent" and enable_rerank:
            raise ValueError("Use agent; its tools already enable reranking")
        seen.add(name)
        strategies.append(EvaluationStrategy(name, mode, enable_rerank))
    return strategies


def keyword_coverage(text: str, keywords: tuple[str, ...]) -> tuple[float, list[str]]:
    normalized_text = re.sub(r"[\s*_`~]+", "", text.casefold())
    hits = [
        keyword
        for keyword in keywords
        if re.sub(r"[\s*_`~]+", "", keyword.casefold()) in normalized_text
    ]
    return len(hits) / len(keywords), hits


def _source_basename(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1].casefold()


def score_result(
    case: EvaluationCase,
    strategy: EvaluationStrategy,
    result: dict[str, Any],
    latency_seconds: float,
    usage: dict[str, int],
) -> dict[str, Any]:
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    chunks = data.get("chunks") if isinstance(data.get("chunks"), list) else []
    references = (
        data.get("references") if isinstance(data.get("references"), list) else []
    )
    entities = data.get("entities") if isinstance(data.get("entities"), list) else []
    relationships = (
        data.get("relationships")
        if isinstance(data.get("relationships"), list)
        else []
    )
    llm_response = result.get("llm_response")
    answer = (
        llm_response.get("content", "")
        if isinstance(llm_response, dict)
        else ""
    ) or ""
    context = "\n".join(str(chunk.get("content", "")) for chunk in chunks)

    answer_coverage, answer_hits = keyword_coverage(answer, case.expected_keywords)
    evidence_coverage, evidence_hits = keyword_coverage(
        context, case.expected_keywords
    )
    source_names = {
        _source_basename(str(reference.get("file_path", "")))
        for reference in references
        if isinstance(reference, dict)
    }
    source_names.update(
        _source_basename(str(chunk.get("file_path", "")))
        for chunk in chunks
        if isinstance(chunk, dict)
    )
    source_hit = _source_basename(case.expected_source) in source_names
    # Evidence is retained internally even though the web hides the reference section.
    citation_present = bool(chunks) and bool(references)
    status = str(result.get("status", "failure"))

    return {
        "case_id": case.case_id,
        "strategy": strategy.name,
        "mode": strategy.mode,
        "rerank": strategy.enable_rerank,
        "question": case.question,
        "status": status,
        "error": "" if status == "success" else str(result.get("message", "")),
        "expected_keywords": list(case.expected_keywords),
        "answer_keyword_hits": answer_hits,
        "evidence_keyword_hits": evidence_hits,
        "answer_keyword_coverage": answer_coverage,
        "evidence_keyword_coverage": evidence_coverage,
        "source_hit": source_hit,
        "citation_present": citation_present,
        "entity_count": len(entities),
        "relationship_count": len(relationships),
        "chunk_count": len(chunks),
        "latency_seconds": latency_seconds,
        "llm_calls": usage.get("call_count", 0),
        "llm_invocations": usage.get("llm_invocations", 0),
        "embedding_invocations": usage.get("embedding_invocations", 0),
        "embedding_tokens": usage.get("embedding_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "zero_llm_call_cache_hit": usage.get("call_count", 0) == 0 and status == "success",
        "answer": answer,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for strategy in dict.fromkeys(row["strategy"] for row in rows):
        selected = [row for row in rows if row["strategy"] == strategy]
        summaries.append(
            {
                "strategy": strategy,
                "cases": len(selected),
                "success_rate": sum(row["status"] == "success" for row in selected)
                / len(selected),
                "answer_keyword_coverage": statistics.fmean(
                    row["answer_keyword_coverage"] for row in selected
                ),
                "evidence_keyword_coverage": statistics.fmean(
                    row["evidence_keyword_coverage"] for row in selected
                ),
                "source_hit_rate": statistics.fmean(
                    float(row["source_hit"]) for row in selected
                ),
                "citation_rate": statistics.fmean(
                    float(row["citation_present"]) for row in selected
                ),
                "average_latency_seconds": statistics.fmean(
                    row["latency_seconds"] for row in selected
                ),
                "llm_calls": sum(row["llm_calls"] for row in selected),
                "total_tokens": sum(row["total_tokens"] for row in selected),
                "zero_llm_call_cache_rate": statistics.fmean(
                    float(row["zero_llm_call_cache_hit"]) for row in selected
                ),
            }
        )
    return summaries


def write_reports(
    output_dir: Path,
    dataset_path: Path,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"book_rag_eval_{timestamp}.json"
    csv_path = output_dir / f"book_rag_eval_{timestamp}.csv"
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "dataset": str(dataset_path.resolve()),
        "summary": summaries,
        "results": rows,
    }
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_fields = [
        "case_id",
        "strategy",
        "mode",
        "rerank",
        "status",
        "answer_keyword_coverage",
        "evidence_keyword_coverage",
        "source_hit",
        "citation_present",
        "entity_count",
        "relationship_count",
        "chunk_count",
        "latency_seconds",
        "llm_calls",
        "llm_invocations",
        "embedding_invocations",
        "embedding_tokens",
        "tool_calls",
        "stop_reason",
        "cache_policy",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "zero_llm_call_cache_hit",
        "question",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def print_summary(summaries: list[dict[str, Any]]) -> None:
    print("\nEvaluation summary")
    print(
        f"{'Strategy':<18} {'Success':>8} {'Answer':>8} {'Evidence':>9} "
        f"{'Source':>8} {'Cite':>7} {'Latency':>9} {'Tokens':>9}"
    )
    for item in summaries:
        print(
            f"{item['strategy']:<18} "
            f"{item['success_rate']:>7.1%} "
            f"{item['answer_keyword_coverage']:>7.1%} "
            f"{item['evidence_keyword_coverage']:>8.1%} "
            f"{item['source_hit_rate']:>7.1%} "
            f"{item['citation_rate']:>6.1%} "
            f"{item['average_latency_seconds']:>8.2f}s "
            f"{item['total_tokens']:>9}"
        )


def build_rag(args: argparse.Namespace, tracker: TokenTracker) -> LightRAG:
    graph = args.index_dir / WORKSPACE / "graph_chunk_entity_relation.graphml"
    if not graph.is_file():
        raise RuntimeError(f"Build the book index first: {graph}")
    llm_model_func = make_llm_func(args.llm_model, token_tracker=tracker)
    embedding_func = make_embedding_func(args.embedding)
    return LightRAG(
        working_dir=str(args.index_dir),
        workspace=WORKSPACE,
        tokenizer=make_local_tokenizer(),
        llm_model_func=llm_model_func,
        llm_model_name=args.llm_model,
        llm_model_max_async=2,
        embedding_func=embedding_func,
        embedding_func_max_async=4,
        rerank_model_func=make_embedding_rerank_func(embedding_func),
        min_rerank_score=-1.0,
    )


async def run(args: argparse.Namespace) -> tuple[list[dict], list[dict], Path, Path]:
    cases = load_cases(args.dataset)
    if args.limit is not None:
        cases = cases[: args.limit]
    strategies = parse_strategies(args.strategies)
    tracker = TokenTracker()
    rag = build_rag(args, tracker)
    cache_policy = getattr(args, "cache_policy", "cold")
    rag.enable_llm_cache = cache_policy == "warm"
    rag.llm_response_cache.global_config["enable_llm_cache"] = cache_policy == "warm"
    settings = AgentSettings.from_env()
    agent = RetrievalAgent(rag, settings=settings)
    rows = []
    await rag.initialize_storages()
    try:
        total = len(cases) * len(strategies)
        current = 0
        for strategy in strategies:
            for case in cases:
                current += 1
                tracker.reset()
                print(f"[{current}/{total}] {strategy.name}: {case.question}")
                started = time.perf_counter()
                usage = RequestUsage()
                usage_token = ACTIVE_USAGE.set(usage)
                agent_result = None
                try:
                    if strategy.mode == "agent":
                        agent_result = await agent.run(case.question)
                        result = agent_result.evaluation_payload()
                    else:
                        query_param = QueryParam(
                            mode=strategy.mode, stream=False,
                            response_type="Chinese concise answer with evidence",
                            chunk_top_k=args.rerank_top_k, max_total_tokens=16000,
                            max_entity_tokens=1500, max_relation_tokens=1500,
                            enable_rerank=strategy.enable_rerank,
                        )
                        result = await asyncio.wait_for(
                            rag.aquery_llm(case.question, query_param),
                            timeout=settings.request_timeout)
                except Exception as exc:
                    result = {"status": "failure", "message": type(exc).__name__}
                finally:
                    ACTIVE_USAGE.reset(usage_token)
                elapsed = time.perf_counter() - started
                row = score_result(
                    case,
                    strategy,
                    result,
                    elapsed,
                    usage.get_usage(),
                )
                row.update({"cache_policy": cache_policy,
                            "stop_reason": agent_result.stop_reason if agent_result else result.get("status"),
                            "tool_calls": sum(item["action"] in {"search_knowledge", "read_source", "query_graph"}
                                              and item["status"] not in {"duplicate", "budget_blocked"}
                                              for item in agent_result.trace) if agent_result else 0,
                            "trace": agent_result.trace if agent_result else [],
                            "usage_reported": usage.get_usage()["usage_reported"]})
                rows.append(row)
                print(
                    f"  status={row['status']} answer={row['answer_keyword_coverage']:.0%} "
                    f"evidence={row['evidence_keyword_coverage']:.0%} "
                    f"source={row['source_hit']} latency={elapsed:.2f}s"
                )
    finally:
        await finalize_rag(rag)

    summaries = summarize(rows)
    json_path, csv_path = write_reports(
        args.output_dir, args.dataset, rows, summaries
    )
    return rows, summaries, json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=project_path, default=DEFAULT_DATASET)
    parser.add_argument("--index-dir", type=project_path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=project_path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strategies",
        default="agent,hybrid+rerank",
        help="Comma-separated strategies or 'all' (default: agent,hybrid+rerank)",
    )
    parser.add_argument("--cache-policy", choices=("cold", "warm"), default="cold",
                        help="cold bypasses LightRAG answer/keyword caches without deleting them")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N cases")
    parser.add_argument(
        "--embedding", choices=("local", "remote"), default="remote"
    )
    parser.add_argument("--llm-model", default=default_llm_model())
    parser.add_argument("--rerank-top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    configure_console()
    args = parse_args()
    args.dataset = args.dataset.expanduser().resolve()
    args.index_dir = args.index_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")
    if args.rerank_top_k <= 0:
        raise SystemExit("--rerank-top-k must be greater than zero")
    try:
        rows, summaries, json_path, csv_path = asyncio.run(run(args))
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print_summary(summaries)
    print(f"\nJSON report: {json_path}")
    print(f"CSV report:  {csv_path}")
    if any(row["status"] != "success" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
