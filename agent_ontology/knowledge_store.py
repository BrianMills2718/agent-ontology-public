#!/usr/bin/env python3
"""
Persistent Knowledge Store for Agent Architecture Evolution

SQLite-backed store that accumulates structured facts about agent architectures:
- Genotype: spec structure, patterns, topology
- Phenotype: benchmark scores, LLM calls, duration
- Lineage: parents, generation, mutations
- Analysis: diagnosis from deep analysis model (Step 2)

Used by evolve.py to persist results after every candidate evaluation,
and queried by recommend.py for evidence-backed architecture suggestions.

Usage:
    from agent_ontology.knowledge_store import KnowledgeStore

    store = KnowledgeStore()  # defaults to ~/.agent_ontology/evolution.db
    store.record_candidate(...)
    best = store.best_genotypes("gsm8k", limit=5)
"""

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone


DEFAULT_DB_PATH = os.path.join(
    os.path.expanduser("~"), ".agent_ontology", "evolution.db"
)


def _topology_hash(spec: dict) -> str:
    """Compute a hash of the graph topology (processes + edges), ignoring names/labels.

    Two specs with the same process types, edge types, and connectivity pattern
    will get the same hash, even if IDs differ.
    """
    # Collect process types in order
    proc_types = []
    for p in spec.get("processes", []):
        proc_types.append(p.get("type", ""))

    # Collect edge signatures (type, from_type, to_type)
    proc_type_map = {p["id"]: p.get("type", "") for p in spec.get("processes", [])}
    entity_type_map = {e["id"]: e.get("type", "") for e in spec.get("entities", [])}
    all_types = {**proc_type_map, **entity_type_map}

    edge_sigs = []
    for e in spec.get("edges", []):
        from_type = all_types.get(e.get("from", ""), "?")
        to_type = all_types.get(e.get("to", ""), "?")
        edge_sigs.append(f"{e.get('type', '')}:{from_type}->{to_type}")
    edge_sigs.sort()

    canonical = json.dumps({"procs": proc_types, "edges": edge_sigs}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class KnowledgeStore:
    """SQLite-backed persistent store for evolution results."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS evolution_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                -- Genotype (what the agent IS)
                spec_name TEXT NOT NULL,
                spec_yaml TEXT,
                base_spec TEXT,
                detected_patterns TEXT,
                topology_hash TEXT,
                entity_count INTEGER,
                process_count INTEGER,
                edge_count INTEGER,
                -- Lineage (where it came from)
                generation INTEGER,
                parents TEXT,
                mutation_description TEXT,
                -- Phenotype (how it performed)
                benchmark TEXT,
                score_em REAL,
                score_f1 REAL,
                fitness REAL,
                llm_calls INTEGER,
                duration_ms INTEGER,
                status TEXT,
                error_details TEXT,
                -- Analysis (from deep analysis model, Step 2)
                analysis TEXT,
                lessons TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_benchmark
                ON evolution_results(benchmark);
            CREATE INDEX IF NOT EXISTS idx_fitness
                ON evolution_results(fitness DESC);
            CREATE INDEX IF NOT EXISTS idx_base_spec
                ON evolution_results(base_spec);
            CREATE INDEX IF NOT EXISTS idx_status
                ON evolution_results(status);
            CREATE INDEX IF NOT EXISTS idx_topology_hash
                ON evolution_results(topology_hash);
        """)
        self.conn.commit()

    def record_candidate(
        self,
        spec_name: str,
        spec: dict | None = None,
        spec_yaml: str | None = None,
        base_spec: str | None = None,
        generation: int = 0,
        parents: list[str] | None = None,
        mutation_description: str | None = None,
        benchmark: str | None = None,
        score_em: float | None = None,
        score_f1: float | None = None,
        fitness: float = 0.0,
        llm_calls: int = 0,
        duration_ms: int = 0,
        status: str = "UNKNOWN",
        error_details: str | None = None,
        analysis: str | None = None,
        lessons: str | None = None,
    ) -> int:
        """Record a single evolution candidate result. Returns the row ID."""
        # Compute derived fields from spec if provided
        detected_patterns = None
        topology_hash = None
        entity_count = None
        process_count = None
        edge_count = None

        if spec:
            entity_count = len(spec.get("entities", []))
            process_count = len(spec.get("processes", []))
            edge_count = len(spec.get("edges", []))
            topology_hash = _topology_hash(spec)

            # Detect patterns
            try:
                from . import patterns as pat_mod
                detected = pat_mod.detect_patterns(spec)
                detected_patterns = json.dumps([p[0] for p in detected])
            except Exception:
                detected_patterns = "[]"

        cursor = self.conn.execute(
            """INSERT INTO evolution_results (
                timestamp, spec_name, spec_yaml, base_spec,
                detected_patterns, topology_hash,
                entity_count, process_count, edge_count,
                generation, parents, mutation_description,
                benchmark, score_em, score_f1, fitness,
                llm_calls, duration_ms, status, error_details,
                analysis, lessons
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                spec_name,
                spec_yaml,
                base_spec,
                detected_patterns,
                topology_hash,
                entity_count,
                process_count,
                edge_count,
                generation,
                json.dumps(parents or []),
                mutation_description,
                benchmark,
                score_em,
                score_f1,
                fitness,
                llm_calls,
                duration_ms,
                status,
                error_details,
                analysis,
                lessons,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    # ── Query helpers ───────────────────────────────────────────

    def best_genotypes(self, benchmark: str, limit: int = 10) -> list[dict]:
        """Best-performing genotypes for a given benchmark, by fitness."""
        rows = self.conn.execute(
            """SELECT spec_name, detected_patterns, fitness, score_em, score_f1,
                      llm_calls, duration_ms, generation, parents, mutation_description
               FROM evolution_results
               WHERE benchmark = ? AND status = 'PASS'
               ORDER BY fitness DESC
               LIMIT ?""",
            (benchmark, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def pattern_performance(self, benchmark: str) -> list[dict]:
        """Average fitness by detected pattern for a benchmark."""
        rows = self.conn.execute(
            """SELECT detected_patterns, AVG(fitness) as avg_fitness,
                      COUNT(*) as count, MAX(fitness) as max_fitness
               FROM evolution_results
               WHERE benchmark = ? AND status = 'PASS' AND detected_patterns IS NOT NULL
               GROUP BY detected_patterns
               ORDER BY avg_fitness DESC""",
            (benchmark,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mutation_effectiveness(self, base_spec: str | None = None) -> list[dict]:
        """Which mutations improved fitness? Optionally filter by base spec."""
        query = """SELECT mutation_description, AVG(fitness) as avg_fitness,
                          COUNT(*) as count, MAX(fitness) as max_fitness,
                          MIN(fitness) as min_fitness
                   FROM evolution_results
                   WHERE mutation_description IS NOT NULL AND fitness > 0"""
        params = []
        if base_spec:
            query += " AND base_spec = ?"
            params.append(base_spec)
        query += " GROUP BY mutation_description ORDER BY avg_fitness DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def failure_lessons(self, benchmark: str | None = None, limit: int = 20) -> list[dict]:
        """Recent failure lessons for learning from mistakes."""
        query = """SELECT spec_name, status, error_details, lessons,
                          mutation_description, detected_patterns, generation
                   FROM evolution_results
                   WHERE status != 'PASS'"""
        params = []
        if benchmark:
            query += " AND benchmark = ?"
            params.append(benchmark)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def topology_duplicates(self) -> list[dict]:
        """Find topology hashes that appear multiple times (potential dedup)."""
        rows = self.conn.execute(
            """SELECT topology_hash, COUNT(*) as count,
                      GROUP_CONCAT(DISTINCT spec_name) as specs,
                      AVG(fitness) as avg_fitness
               FROM evolution_results
               WHERE topology_hash IS NOT NULL
               GROUP BY topology_hash
               HAVING count > 1
               ORDER BY count DESC""",
        ).fetchall()
        return [dict(r) for r in rows]

    def generation_summary(self, base_spec: str | None = None) -> list[dict]:
        """Summary stats per generation."""
        query = """SELECT generation,
                          COUNT(*) as candidates,
                          SUM(CASE WHEN status = 'PASS' THEN 1 ELSE 0 END) as passed,
                          SUM(CASE WHEN status = 'INVALID' THEN 1 ELSE 0 END) as invalid,
                          AVG(fitness) as avg_fitness,
                          MAX(fitness) as max_fitness,
                          AVG(llm_calls) as avg_llm_calls
                   FROM evolution_results"""
        params = []
        if base_spec:
            query += " WHERE base_spec = ?"
            params.append(base_spec)
        query += " GROUP BY generation ORDER BY generation"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        """Total number of recorded candidates."""
        return self.conn.execute("SELECT COUNT(*) FROM evolution_results").fetchone()[0]

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── CLI ─────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Query the evolution knowledge store")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Database path")
    sub = parser.add_subparsers(dest="command")

    # stats
    sub.add_parser("stats", help="Show store statistics")

    # best
    best_p = sub.add_parser("best", help="Best genotypes for a benchmark")
    best_p.add_argument("benchmark", help="Benchmark name (e.g., gsm8k)")
    best_p.add_argument("--limit", type=int, default=10)

    # patterns
    pat_p = sub.add_parser("patterns", help="Pattern performance on a benchmark")
    pat_p.add_argument("benchmark", help="Benchmark name")

    # mutations
    mut_p = sub.add_parser("mutations", help="Mutation effectiveness")
    mut_p.add_argument("--base-spec", help="Filter by base spec")

    # failures
    fail_p = sub.add_parser("failures", help="Recent failure lessons")
    fail_p.add_argument("--benchmark", help="Filter by benchmark")
    fail_p.add_argument("--limit", type=int, default=20)

    # generations
    gen_p = sub.add_parser("generations", help="Generation summary")
    gen_p.add_argument("--base-spec", help="Filter by base spec")

    args = parser.parse_args()

    store = KnowledgeStore(args.db)

    if args.command == "stats":
        n = store.count()
        print(f"Total candidates: {n}")
        if n > 0:
            row = store.conn.execute(
                "SELECT MIN(timestamp) as first, MAX(timestamp) as last FROM evolution_results"
            ).fetchone()
            print(f"First: {row['first']}")
            print(f"Last:  {row['last']}")
            benchmarks = store.conn.execute(
                "SELECT DISTINCT benchmark FROM evolution_results WHERE benchmark IS NOT NULL"
            ).fetchall()
            print(f"Benchmarks: {', '.join(r['benchmark'] for r in benchmarks)}")

    elif args.command == "best":
        results = store.best_genotypes(args.benchmark, args.limit)
        if not results:
            print(f"No results for benchmark '{args.benchmark}'")
        for r in results:
            print(f"  {r['spec_name']:40s}  fitness={r['fitness']:6.1f}  "
                  f"em={r['score_em'] or 0:.2f}  calls={r['llm_calls']}")

    elif args.command == "patterns":
        results = store.pattern_performance(args.benchmark)
        if not results:
            print(f"No pattern data for benchmark '{args.benchmark}'")
        for r in results:
            print(f"  {r['detected_patterns']:50s}  avg={r['avg_fitness']:6.1f}  "
                  f"max={r['max_fitness']:6.1f}  n={r['count']}")

    elif args.command == "mutations":
        results = store.mutation_effectiveness(args.base_spec)
        if not results:
            print("No mutation data")
        for r in results:
            desc = (r['mutation_description'] or "")[:60]
            print(f"  {desc:60s}  avg={r['avg_fitness']:6.1f}  "
                  f"max={r['max_fitness']:6.1f}  n={r['count']}")

    elif args.command == "failures":
        results = store.failure_lessons(args.benchmark, args.limit)
        if not results:
            print("No failures recorded")
        for r in results:
            print(f"  [{r['status']}] {r['spec_name']}: {(r['error_details'] or '')[:80]}")
            if r['lessons']:
                print(f"    Lesson: {r['lessons'][:100]}")

    elif args.command == "generations":
        results = store.generation_summary(args.base_spec)
        if not results:
            print("No generation data")
        for r in results:
            print(f"  Gen {r['generation']}: {r['candidates']} candidates, "
                  f"{r['passed']} passed, {r['invalid']} invalid, "
                  f"avg_fitness={r['avg_fitness']:6.1f}, max={r['max_fitness']:6.1f}")

    else:
        parser.print_help()

    store.close()


if __name__ == "__main__":
    main()
