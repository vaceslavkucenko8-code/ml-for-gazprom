"""
src/corpus_store.py

Persistent, append-only, self-growing corpus of documents/candidates found by
`src/continuous_ingestion.py`. This is deliberately a SEPARATE dataset from:

  * data/raw/signals.xlsx           — original organizer file, never modified.
  * data/self_review/labels.json    — human-entered labels, never overwritten.

Nothing here invents a binary weak-signal label. Every stored candidate keeps
the SAME two independent, already-existing, rule-based assessments the rest
of the project uses (no parallel/duplicate scoring logic):

  * `extraction.relevance.assess_relevance`  -> search-time triage score/verdict
  * `automatic.assess_automatically`         -> evidence-pattern rule engine
    (the same one used on the 100-row self-review, see docs/SELF_REVIEW.md)

Storage is plain SQLite (stdlib, no new dependency) so it scales to a large,
continuously growing number of rows without needing PostgreSQL/network.
Growth happens across many ingestion cycles (see continuous_ingestion.py),
not in one shot — a single run never claims to have "collected 100000 rows".
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "corpus" / "corpus.db"

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cycles (
  cycle_id TEXT PRIMARY KEY,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL,
  seed_topics_json TEXT NOT NULL,
  topics_queried_json TEXT NOT NULL,
  new_documents INTEGER NOT NULL DEFAULT 0,
  new_candidates INTEGER NOT NULL DEFAULT 0,
  included_candidates INTEGER NOT NULL DEFAULT 0,
  excluded_candidates INTEGER NOT NULL DEFAULT 0,
  connectors_used_json TEXT NOT NULL DEFAULT '[]',
  connectors_skipped_json TEXT NOT NULL DEFAULT '[]',
  errors_json TEXT NOT NULL DEFAULT '[]',
  error_message TEXT
);

CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY,
  normalized_url TEXT NOT NULL,
  url TEXT NOT NULL,
  title TEXT,
  domain TEXT,
  source_type TEXT,
  connector TEXT,
  published_at TEXT,
  fetch_status TEXT,
  trust_level TEXT,
  first_seen_cycle TEXT NOT NULL,
  last_seen_cycle TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_normalized_url ON documents(normalized_url);
CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents(domain);

CREATE TABLE IF NOT EXISTS candidates (
  candidate_id TEXT PRIMARY KEY,
  technology_name TEXT NOT NULL,
  domain TEXT,
  seed_topic TEXT,
  query TEXT,
  relevance_score REAL,
  relevance_verdict TEXT,
  is_actual TEXT,
  automatic_decision TEXT,
  automatic_reason TEXT,
  automatic_score REAL,
  included INTEGER NOT NULL DEFAULT 0,
  exclude_reason TEXT,
  first_seen_cycle TEXT NOT NULL,
  last_seen_cycle TEXT NOT NULL,
  times_seen INTEGER NOT NULL DEFAULT 1,
  payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_candidates_domain ON candidates(domain);
CREATE INDEX IF NOT EXISTS idx_candidates_included ON candidates(included);
CREATE INDEX IF NOT EXISTS idx_candidates_automatic_decision ON candidates(automatic_decision);

CREATE TABLE IF NOT EXISTS seed_topic_rotation (
  label TEXT PRIMARY KEY,
  last_queried_cycle TEXT,
  times_queried INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS adaptive_topic_seeds (
  source_candidate_id TEXT PRIMARY KEY,
  topic TEXT NOT NULL,
  label TEXT NOT NULL,
  created_at TEXT NOT NULL,
  used INTEGER NOT NULL DEFAULT 0
);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CorpusStore:
    """Thin, explicit wrapper around a SQLite file. No ORM: every query is
    written out so it stays auditable, matching the rest of this codebase.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(DDL)
        self.conn.execute("CREATE TABLE IF NOT EXISTS document_versions (normalized_url TEXT, cycle_id TEXT, payload_json TEXT, PRIMARY KEY(normalized_url,cycle_id))")
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "CorpusStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Cycles
    # ------------------------------------------------------------------ #

    def start_cycle(self, cycle_id: str, seed_topics: dict, topics_queried: list[str]) -> None:
        self.conn.execute(
            "INSERT INTO cycles(cycle_id, started_at, status, seed_topics_json, topics_queried_json) "
            "VALUES (?, ?, 'running', ?, ?)",
            (cycle_id, _utc_now_iso(), json.dumps(seed_topics, ensure_ascii=False), json.dumps(topics_queried, ensure_ascii=False)),
        )
        self.conn.commit()

    def finish_cycle(self, cycle_id: str, *, status: str, counts: dict, connectors_used: list[str],
                      connectors_skipped: list[str], errors: list[Any], error_message: Optional[str] = None) -> None:
        self.conn.execute(
            "UPDATE cycles SET finished_at=?, status=?, new_documents=?, new_candidates=?, "
            "included_candidates=?, excluded_candidates=?, connectors_used_json=?, "
            "connectors_skipped_json=?, errors_json=?, error_message=? WHERE cycle_id=?",
            (
                _utc_now_iso(), status,
                counts.get("new_documents", 0), counts.get("new_candidates", 0),
                counts.get("included_candidates", 0), counts.get("excluded_candidates", 0),
                json.dumps(sorted(set(connectors_used)), ensure_ascii=False),
                json.dumps(sorted(set(connectors_skipped)), ensure_ascii=False),
                json.dumps(errors, ensure_ascii=False, default=str),
                error_message,
                cycle_id,
            ),
        )
        self.conn.commit()

    def recent_cycles(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM cycles ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def last_cycle_id(self) -> Optional[str]:
        row = self.conn.execute("SELECT cycle_id FROM cycles ORDER BY started_at DESC LIMIT 1").fetchone()
        return row["cycle_id"] if row else None

    # ------------------------------------------------------------------ #
    # Dedup helpers
    # ------------------------------------------------------------------ #

    def known_normalized_urls(self) -> set[str]:
        return {r["normalized_url"] for r in self.conn.execute("SELECT normalized_url FROM documents")}

    def known_candidate_ids(self) -> set[str]:
        return {r["candidate_id"] for r in self.conn.execute("SELECT candidate_id FROM candidates")}

    # ------------------------------------------------------------------ #
    # Documents
    # ------------------------------------------------------------------ #

    def upsert_document(self, cycle_id: str, document: dict) -> bool:
        """Returns True if this is a newly-seen document (by normalized_url)."""
        normalized_url = document.get("normalized_url") or document["url"]
        existing = self.conn.execute(
            "SELECT doc_id, payload_json, last_seen_cycle FROM documents WHERE normalized_url=?", (normalized_url,)
        ).fetchone()
        payload_json = json.dumps(document, ensure_ascii=False, default=str)
        if existing:
            self.conn.execute("INSERT OR IGNORE INTO document_versions VALUES (?,?,?)",
                              (normalized_url, existing["last_seen_cycle"], existing["payload_json"]))
            self.conn.execute(
                "UPDATE documents SET last_seen_cycle=?, payload_json=? WHERE normalized_url=?",
                (cycle_id, payload_json, normalized_url),
            )
            return False
        self.conn.execute(
            "INSERT INTO documents(doc_id, normalized_url, url, title, domain, source_type, connector, "
            "published_at, fetch_status, trust_level, first_seen_cycle, last_seen_cycle, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                document["doc_id"], normalized_url, document["url"], document.get("title"),
                document.get("domain"), document.get("source_type"), document.get("connector"),
                document.get("published_at"), document.get("fetch_status"), document.get("trust_level"),
                cycle_id, cycle_id, payload_json,
            ),
        )
        return True

    # ------------------------------------------------------------------ #
    # Candidates
    # ------------------------------------------------------------------ #

    def upsert_candidate(self, cycle_id: str, candidate_record: dict) -> bool:
        """`candidate_record` is the flat dict produced by
        `continuous_ingestion._candidate_record`. Returns True if new.
        """
        cid = candidate_record["candidate_id"]
        existing = self.conn.execute("SELECT candidate_id, times_seen FROM candidates WHERE candidate_id=?", (cid,)).fetchone()
        payload_json = json.dumps(candidate_record, ensure_ascii=False, default=str)
        if existing:
            self.conn.execute(
                "UPDATE candidates SET last_seen_cycle=?, times_seen=?, relevance_score=?, relevance_verdict=?, "
                "is_actual=?, automatic_decision=?, automatic_reason=?, automatic_score=?, included=?, "
                "exclude_reason=?, payload_json=? WHERE candidate_id=?",
                (
                    cycle_id, existing["times_seen"] + 1,
                    candidate_record.get("relevance_score"), candidate_record.get("relevance_verdict"),
                    str(candidate_record.get("is_actual")), candidate_record.get("automatic_decision"),
                    candidate_record.get("automatic_reason"), candidate_record.get("automatic_score"),
                    int(candidate_record.get("included", False)), candidate_record.get("exclude_reason"),
                    payload_json, cid,
                ),
            )
            return False
        self.conn.execute(
            "INSERT INTO candidates(candidate_id, technology_name, domain, seed_topic, query, relevance_score, "
            "relevance_verdict, is_actual, automatic_decision, automatic_reason, automatic_score, included, "
            "exclude_reason, first_seen_cycle, last_seen_cycle, times_seen, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
            (
                cid, candidate_record["technology_name"], candidate_record.get("domain"),
                candidate_record.get("seed_topic"), candidate_record.get("query"),
                candidate_record.get("relevance_score"), candidate_record.get("relevance_verdict"),
                str(candidate_record.get("is_actual")), candidate_record.get("automatic_decision"),
                candidate_record.get("automatic_reason"), candidate_record.get("automatic_score"),
                int(candidate_record.get("included", False)), candidate_record.get("exclude_reason"),
                cycle_id, cycle_id, payload_json,
            ),
        )
        return True

    def candidates(self, *, included_only: bool = False, limit: Optional[int] = None) -> list[dict]:
        query = "SELECT * FROM candidates"
        if included_only:
            query += " WHERE included=1"
        query += " ORDER BY first_seen_cycle DESC"
        if limit:
            query += f" LIMIT {int(limit)}"
        return [dict(r) for r in self.conn.execute(query)]

    def high_score_candidates_for_adaptive_seeding(self, limit: int = 8) -> list[dict]:
        """Best-scoring included candidates not yet turned into an adaptive
        follow-up search topic — feeds `continuous_ingestion._adaptive_topics`.
        """
        rows = self.conn.execute(
            "SELECT c.* FROM candidates c LEFT JOIN adaptive_topic_seeds a "
            "ON a.source_candidate_id = c.candidate_id "
            "WHERE c.included=1 AND a.source_candidate_id IS NULL "
            "ORDER BY c.relevance_score DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_adaptive_topic_used(self, source_candidate_id: str, topic: str, label: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO adaptive_topic_seeds(source_candidate_id, topic, label, created_at, used) "
            "VALUES (?, ?, ?, ?, 1)",
            (source_candidate_id, topic, label, _utc_now_iso()),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Seed topic rotation (so an hourly cycle does not have to query every
    # base area every time — spreads load across cycles, connector-friendly)
    # ------------------------------------------------------------------ #

    def least_recently_queried_labels(self, all_labels: Iterable[str], limit: int) -> list[str]:
        labels = list(all_labels)
        for label in labels:
            self.conn.execute(
                "INSERT OR IGNORE INTO seed_topic_rotation(label, last_queried_cycle, times_queried) VALUES (?, NULL, 0)",
                (label,),
            )
        self.conn.commit()
        rows = self.conn.execute(
            "SELECT label FROM seed_topic_rotation WHERE label IN (%s) "
            "ORDER BY (last_queried_cycle IS NOT NULL), last_queried_cycle ASC, times_queried ASC "
            % ",".join("?" * len(labels)),
            labels,
        ).fetchall()
        return [r["label"] for r in rows][:limit]

    def mark_labels_queried(self, labels: Iterable[str], cycle_id: str) -> None:
        for label in labels:
            self.conn.execute(
                "UPDATE seed_topic_rotation SET last_queried_cycle=?, times_queried=times_queried+1 WHERE label=?",
                (cycle_id, label),
            )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Stats / export
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        doc_count = self.conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
        cand_count = self.conn.execute("SELECT COUNT(*) c FROM candidates").fetchone()["c"]
        included = self.conn.execute("SELECT COUNT(*) c FROM candidates WHERE included=1").fetchone()["c"]
        by_automatic = {
            r["automatic_decision"] or "not_evaluated": r["c"]
            for r in self.conn.execute("SELECT automatic_decision, COUNT(*) c FROM candidates GROUP BY automatic_decision")
        }
        by_relevance = {
            r["relevance_verdict"] or "not_evaluated": r["c"]
            for r in self.conn.execute("SELECT relevance_verdict, COUNT(*) c FROM candidates GROUP BY relevance_verdict")
        }
        cycles = self.conn.execute("SELECT COUNT(*) c FROM cycles WHERE status='ok'").fetchone()["c"]
        cycles_failed = self.conn.execute("SELECT COUNT(*) c FROM cycles WHERE status='error'").fetchone()["c"]
        return {
            "documents": doc_count,
            "candidates": cand_count,
            "included_candidates": included,
            "excluded_candidates": cand_count - included,
            "by_automatic_decision": by_automatic,
            "by_relevance_verdict": by_relevance,
            "cycles_ok": cycles,
            "cycles_failed": cycles_failed,
            "db_path": str(self.db_path),
            "note": "Автособранный корпус; rule-based скрининг, не подтверждённая человеком истина. "
                    "Не заменяет data/raw/signals.xlsx и data/self_review/labels.json.",
        }

    def export_dataset(self, out_dir: Optional[Path] = None, *, included_only: bool = True) -> dict:
        """Writes the growing self-collected dataset out as JSONL + a summary,
        separate from the organizer XLSX and from human self-review labels.
        Returns the summary dict that was written.
        """
        out_dir = Path(out_dir) if out_dir else PROJECT_ROOT / "data" / "discovered"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = self.candidates(included_only=included_only)
        dataset_path = out_dir / "dataset.jsonl"
        with dataset_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                record = json.loads(row["payload_json"])
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        summary = {
            "generated_at": _utc_now_iso(),
            "row_count": len(rows),
            "included_only": included_only,
            **self.stats(),
            "provenance": (
                "Собрано автоматически из открытых источников через src/continuous_ingestion.py; "
                "оценки — те же rule-based алгоритмы, что и в src/extraction/relevance.py и src/automatic.py. "
                "Это не независимая экспертная разметка и не замена скрытых организаторских меток."
            ),
        }
        (out_dir / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        return summary


@contextlib.contextmanager
def open_store(db_path: Optional[Path] = None):
    store = CorpusStore(db_path)
    try:
        yield store
    finally:
        store.close()
