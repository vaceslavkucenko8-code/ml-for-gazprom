"""
src/continuous_ingestion.py

Continuous, self-collecting corpus builder. Answers a concrete request from
the team: the working dataset should not stay pinned to the 100 organizer
rows — the system should keep searching the open sources already wired up
in `src/retrieval/connectors/`, screen every new candidate through the SAME
rule-based algorithms already used elsewhere in the project, and accumulate
the result into its own growing, persisted dataset (`src/corpus_store.py`).

What this deliberately does NOT do (see AGENTS.md / HANDOFF_STATUS.md):
  * It does not touch data/raw/signals.xlsx or data/self_review/labels.json.
  * It does not invent a binary weak-signal ground-truth label. Every stored
    candidate keeps two clearly separate, already-existing assessments:
      - extraction.relevance.assess_relevance   (search-time triage rule)
      - automatic.assess_automatically           (evidence-pattern rule,
        the same engine used on the 100-row self-review)
    Neither is treated as a verified label; both are stored explicitly
    tagged with their own `model_version`/`score_kind`.
  * It does not claim to have collected any specific row count "already" —
    growth happens across many cycles (see `run_loop`), each one bounded
    and connector-friendly, not in a single unrealistic sweep of the whole
    internet.
  * It does not start itself. Running it (`--once`, or `--loop` for the
    opt-in hourly-by-default scheduler) is something a person invokes
    explicitly — this module never launches a background service on import.

How "adaptive" is implemented, concretely (no hidden magic):
  1. Seed rotation: `DEFAULT_SEED_TOPICS` (8 broad areas, see discovery.py)
     are queried in a round-robin — least-recently-queried areas first
     (`CorpusStore.least_recently_queried_labels`) — so one hourly cycle
     stays fast and no area is starved over many cycles.
  2. Adaptive follow-up topics: the best-scoring, not-yet-followed-up
     candidates already in the corpus each spawn one new, explicit search
     query ("<technology name> развитие внедрение применение") to look for
     newer evidence about the SAME technology. This is the whole "adaptive"
     mechanism — an explicit, inspectable query string, not a learned
     trend-detector.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from .automatic import assess_automatically
from .extraction.candidates import extract_candidates
from .extraction.relevance import RelevanceVerdict, assess_candidates
from .participant2_adapter import adapt_export
from .retrieval.connectors import DEFAULT_CONNECTORS, OPTIONAL_CONNECTORS
from .retrieval.deduplicate import deduplicate_and_score
from .retrieval.discovery import DEFAULT_SEED_TOPICS
from .retrieval.fetch import Fetcher
from .retrieval.search import SearchOrchestrator
from .corpus_store import CorpusStore, DEFAULT_DB_PATH, open_store

logger = logging.getLogger(__name__)

CONTINUOUS_QUERY_LABEL = "непрерывный автосбор (continuous ingestion)"
DEFAULT_INTERVAL_SECONDS = 3600  # hourly by default, per team request — not daily
ADAPTIVE_TOPIC_TEMPLATE = "{name} развитие внедрение применение"


def _new_cycle_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def fast_orchestrator(max_results_per_connector: int = 6) -> SearchOrchestrator:
    """Default orchestrator for scheduled/unattended cycles.

    Found by running real cycles against live connectors (see
    docs/CONTINUOUS_INGESTION.md): the default `Fetcher` (2 retries,
    15s read timeout — tuned for a single interactive query) makes one
    unreachable connector cost 30-60+ seconds *per topic* when it is
    down (observed repeatedly with GDELT in this environment), which
    stacks up across several topics per cycle and defeats the point of
    an hourly, fast-turnaround cycle. `src/pipeline.py`'s CLI already
    solves this the same way for live queries
    (`Fetcher(max_retries=0, connect_timeout=5, read_timeout=12,
    min_delay_per_domain=3)`) — reused here verbatim rather than
    inventing a second tuning.
    """
    fetcher = Fetcher(max_retries=0, connect_timeout=5, read_timeout=12, min_delay_per_domain=3)
    connectors = [cls(fetcher=fetcher) for cls in DEFAULT_CONNECTORS]
    connectors += [cls(fetcher=fetcher) for cls in OPTIONAL_CONNECTORS]
    return SearchOrchestrator(connectors=connectors, fetcher=fetcher, max_results_per_connector=max_results_per_connector)


def _adaptive_topics(store: CorpusStore, limit: int) -> dict[str, str]:
    """Derives up to `limit` new follow-up search topics from the
    best-scoring candidates already in the corpus that have not yet been
    used as an adaptive seed. Explicit and bounded — see module docstring.
    """
    if limit <= 0:
        return {}
    topics: dict[str, str] = {}
    for row in store.high_score_candidates_for_adaptive_seeding(limit=limit):
        name = (row["technology_name"] or "").strip()
        if not name:
            continue
        label = f"Адаптивно: {name}"[:120]
        topic = ADAPTIVE_TOPIC_TEMPLATE.format(name=name)
        topics[label] = topic
        store.mark_adaptive_topic_used(row["candidate_id"], topic, label)
    return topics


def _candidate_record(candidate, assessment, automatic_result: Optional[dict], collected_at: str) -> dict:
    included = assessment.verdict != RelevanceVerdict.LIKELY_NOISE
    return {
        "candidate_id": candidate.candidate_id,
        "technology_name": candidate.technology_name,
        "short_description": candidate.short_description,
        "domain": candidate.domain,
        "query": candidate.query,
        "companies": candidate.companies,
        "source_doc_ids": candidate.source_doc_ids,
        "independent_confirmations": candidate.independent_confirmations,
        "confirmation_explanation": candidate.confirmation_explanation,
        "evidence_count": len(candidate.evidence),
        "relevance_score": assessment.score,
        "relevance_verdict": assessment.verdict.value,
        "relevance_score_kind": "rule_based_triage_0_1_not_probability",
        "is_actual": assessment.is_actual,
        "actuality_reason": assessment.actuality_reason,
        "automatic_decision": (automatic_result or {}).get("decision"),
        "automatic_reason": (automatic_result or {}).get("reason"),
        "automatic_score": (automatic_result or {}).get("score"),
        "automatic_score_kind": (automatic_result or {}).get("score_kind"),
        "automatic_model_version": (automatic_result or {}).get("model_version"),
        "automatic_requires_expert_validation": (automatic_result or {}).get("requires_expert_validation", True),
        "included": included,
        "exclude_reason": None if included else "likely_noise_relevance_rule",
        "collected_at": collected_at,
        "label_kind": "automatic_hypothesis_not_expert_ground_truth",
    }


def run_cycle(
    store: CorpusStore,
    orchestrator: Optional[SearchOrchestrator] = None,
    *,
    seed_topics: Optional[dict[str, str]] = None,
    max_seed_topics: int = 4,
    max_adaptive_topics: int = 3,
    max_queries_per_topic: int = 2,
    as_of: Optional[str] = None,
) -> dict:
    """Runs exactly one bounded ingestion cycle and persists everything new
    to `store`. Never raises for expected failure modes (connector errors,
    unexpected exceptions during scoring) — always returns a report dict
    with `status` in {"ok", "error"}, so a scheduler loop can keep going.
    """
    seed_topics = dict(seed_topics or DEFAULT_SEED_TOPICS)
    orchestrator = orchestrator or fast_orchestrator()
    as_of = as_of or date.today().isoformat()
    cycle_id = _new_cycle_id()

    chosen_labels = store.least_recently_queried_labels(seed_topics.keys(), max_seed_topics)
    topics_to_query: dict[str, str] = {label: seed_topics[label] for label in chosen_labels}
    adaptive = _adaptive_topics(store, max_adaptive_topics)
    topics_to_query.update(adaptive)

    store.start_cycle(cycle_id, seed_topics, list(topics_to_query))

    errors: list[dict] = []
    connectors_used: list[str] = []
    connectors_skipped: list[str] = []
    counts = {"new_documents": 0, "new_candidates": 0, "included_candidates": 0, "excluded_candidates": 0}

    try:
        known_urls = store.known_normalized_urls()
        seen_this_cycle: set[str] = set()
        all_documents = []

        for label, topic in topics_to_query.items():
            run = orchestrator.run(topic, max_queries=max_queries_per_topic)
            errors.extend(dataclasses.asdict(e) for e in run.errors)
            connectors_used.extend(run.connectors_used)
            connectors_skipped.extend(run.connectors_skipped)
            for doc in run.documents:
                norm = doc.normalized_url or doc.url
                if norm in seen_this_cycle:
                    continue
                seen_this_cycle.add(norm)
                doc.raw_metadata = {**doc.raw_metadata, "seed_label": label, "seed_query": topic}
                all_documents.append(doc)

        documents = deduplicate_and_score(all_documents)
        for doc in documents:
            if store.upsert_document(cycle_id, json.loads(doc.model_dump_json())):
                counts["new_documents"] += 1

        candidates = []
        by_topic = {}
        for doc in documents:
            topic = doc.raw_metadata.get("seed_query") or doc.query
            by_topic.setdefault(topic, []).append(doc)
        for topic, topic_documents in by_topic.items():
            candidates.extend(extract_candidates(topic, topic_documents))
        import hashlib
        for candidate in candidates:
            identity=json.dumps([candidate.query, sorted(candidate.source_doc_ids)],ensure_ascii=False)
            candidate.candidate_id='corpus_'+hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]
        relevance_by_id = {a.candidate_id: a for a in assess_candidates(candidates, documents)}

        automatic_by_id: dict[str, dict] = {}
        if candidates:
            export_payload = {
                "query": CONTINUOUS_QUERY_LABEL,
                "documents": [json.loads(d.model_dump_json()) for d in documents],
                "candidates": [json.loads(c.model_dump_json()) for c in candidates],
            }
            adapted = adapt_export(export_payload)
            automatic = assess_automatically(adapted["candidates"], as_of)
            automatic_by_id = {r["candidate_id"]: r for r in automatic["assessments"]}

        collected_at = datetime.now(timezone.utc).isoformat()
        known_candidate_ids = store.known_candidate_ids()
        for candidate in candidates:
            assessment = relevance_by_id[candidate.candidate_id]
            record = _candidate_record(candidate, assessment, automatic_by_id.get(candidate.candidate_id), collected_at)
            is_new = candidate.candidate_id not in known_candidate_ids
            store.upsert_candidate(cycle_id, record)
            if is_new:
                counts["new_candidates"] += 1
                if record["included"]:
                    counts["included_candidates"] += 1
                else:
                    counts["excluded_candidates"] += 1

        store.mark_labels_queried(chosen_labels, cycle_id)
        store.finish_cycle(
            cycle_id, status="ok", counts=counts,
            connectors_used=connectors_used, connectors_skipped=connectors_skipped, errors=errors,
        )
        return {
            "cycle_id": cycle_id, "status": "ok", "as_of": as_of,
            "topics_queried": topics_to_query, "counts": counts,
            "connectors_used": sorted(set(connectors_used)), "connectors_skipped": sorted(set(connectors_skipped)),
            "errors": errors,
        }
    except Exception as exc:  # noqa: BLE001 — one bad cycle must not kill a scheduler loop
        logger.exception("Цикл автосбора %s завершился ошибкой", cycle_id)
        store.finish_cycle(
            cycle_id, status="error", counts=counts,
            connectors_used=connectors_used, connectors_skipped=connectors_skipped, errors=errors,
            error_message=str(exc),
        )
        return {
            "cycle_id": cycle_id, "status": "error", "as_of": as_of,
            "topics_queried": topics_to_query, "counts": counts, "error": str(exc),
            "connectors_used": sorted(set(connectors_used)), "connectors_skipped": sorted(set(connectors_skipped)),
            "errors": errors,
        }


def run_loop(
    store: CorpusStore,
    *,
    orchestrator_factory: Callable[[], SearchOrchestrator] = fast_orchestrator,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    max_cycles: Optional[int] = None,
    seed_topics: Optional[dict[str, str]] = None,
    max_seed_topics: int = 4,
    max_adaptive_topics: int = 3,
    max_queries_per_topic: int = 2,
    sleep_fn: Callable[[float], None] = time.sleep,
    heartbeat_path: Optional[Path] = None,
    on_cycle: Optional[Callable[[dict], None]] = None,
) -> list[dict]:
    """Opt-in scheduler: repeatedly calls `run_cycle`, sleeping
    `interval_seconds` between cycles (3600s / hourly by default, per the
    team's request — explicitly not once a day). Must be started explicitly
    by a person (CLI `--loop`, Task Scheduler entry, etc.) — never launched
    implicitly on import.
    """
    heartbeat_path = heartbeat_path or (Path(__file__).resolve().parents[1] / "artifacts" / "continuous_ingestion" / "heartbeat.json")
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    cycle_n = 0
    while max_cycles is None or cycle_n < max_cycles:
        cycle_n += 1
        report = run_cycle(
            store, orchestrator_factory(),
            seed_topics=seed_topics, max_seed_topics=max_seed_topics,
            max_adaptive_topics=max_adaptive_topics, max_queries_per_topic=max_queries_per_topic,
        )
        reports.append(report)
        loop_finished = max_cycles is not None and cycle_n >= max_cycles
        heartbeat = {
            "last_cycle": report, "cycle_number": cycle_n,
            "interval_seconds": interval_seconds,
            # None means "loop stopped, no further cycle is scheduled" —
            # this used to be inverted (set only when the loop had already
            # finished), which made a live scheduler's heartbeat claim it
            # had no next run while still running normally.
            "next_run_at": None if loop_finished else (datetime.now(timezone.utc) + timedelta(seconds=interval_seconds)).isoformat(),
            "stats": store.stats(),
        }
        heartbeat_path.write_text(json.dumps(heartbeat, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        if on_cycle:
            on_cycle(report)
        if loop_finished:
            break
        sleep_fn(interval_seconds)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Непрерывный самостоятельный сбор кандидатов в открытых источниках")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Путь к файлу corpus.db")
    parser.add_argument("--loop", action="store_true", help="Запустить планировщик (по умолчанию — ежечасно), а не один цикл")
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--max-cycles", type=int, default=None, help="Ограничить число циклов (для теста/разового запуска планировщика)")
    parser.add_argument("--max-seed-topics", type=int, default=4)
    parser.add_argument("--max-adaptive-topics", type=int, default=3)
    parser.add_argument("--max-queries-per-topic", type=int, default=2)
    parser.add_argument("--export", action="store_true", help="После завершения выгрузить data/discovered/dataset.jsonl")
    args = parser.parse_args()

    for _stream in (sys.stdout, sys.stderr):
        if _stream.encoding and _stream.encoding.lower() != "utf-8":
            _stream.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open_store(args.db) as store:
        if args.loop:
            reports = run_loop(
                store, interval_seconds=args.interval_seconds, max_cycles=args.max_cycles,
                max_seed_topics=args.max_seed_topics, max_adaptive_topics=args.max_adaptive_topics,
                max_queries_per_topic=args.max_queries_per_topic,
            )
            print(json.dumps({"cycles": len(reports), "last": reports[-1] if reports else None}, ensure_ascii=False, indent=2, default=str), flush=True)
        else:
            report = run_cycle(
                store, fast_orchestrator(),
                max_seed_topics=args.max_seed_topics, max_adaptive_topics=args.max_adaptive_topics,
                max_queries_per_topic=args.max_queries_per_topic,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
        print(json.dumps(store.stats(), ensure_ascii=False, indent=2, default=str), flush=True)
        if args.export:
            summary = store.export_dataset()
            print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
