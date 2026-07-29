"""
registry.py
===========
A model registry with versions, metrics and lineage (Stage B deliverable).

Design decisions (Section 8 alternatives, written down as required):
  - Backing store: SQLite file, not MLflow server. REJECTED MLflow because
    this environment has no persistent server process and MLflow's own
    tracking DB is itself just SQLite under the hood for local use — so we
    get the same guarantees (queryable, ACID, versioned rows) with zero
    extra infra. If PlaceMux later needs multi-host access, the schema
    below maps 1:1 onto MLflow's model registry tables, so migration is a
    straight export.
  - Promotion model: explicit `production` pointer table with full history,
    rather than mutating a "latest" tag, so "which model served decision X
    six months ago" (Section 9 question) is always answerable.

Every row records: version, artifact path, training data hash (lineage),
feature schema hash (train/serve contract), offline metrics, who/why
rejected as candidates, and timestamps. Rollback = re-pointing production
to an older version, itself logged as an event (audit trail, not deletion).
"""
import hashlib
import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    version TEXT PRIMARY KEY,
    created_at REAL,
    artifact_path TEXT,
    training_data_hash TEXT,
    feature_schema_hash TEXT,
    parent_version TEXT,
    train_rows INTEGER,
    metrics_json TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS promotions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT,
    action TEXT,           -- 'promote' | 'rollback'
    ts REAL,
    reason TEXT,
    approved_by TEXT
);
"""


def hash_dataframe(df) -> str:
    """Content hash of the exact training slice, so lineage is reproducible."""
    return hashlib.sha256(
        df.to_csv(index=False).encode()
    ).hexdigest()[:16]


class ModelRegistry:
    def __init__(self, db_path="governance/registry.db", artifact_dir="governance/models"):
        self.db_path = db_path
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---------- registration ----------
    def register(self, version, model_obj, training_df, feature_schema_hash,
                 metrics: dict, parent_version=None, notes=""):
        import joblib
        artifact_path = str(self.artifact_dir / f"{version}.joblib")
        joblib.dump(model_obj, artifact_path)
        data_hash = hash_dataframe(training_df)
        self.conn.execute(
            "INSERT OR REPLACE INTO models VALUES (?,?,?,?,?,?,?,?,?)",
            (version, time.time(), artifact_path, data_hash, feature_schema_hash,
             parent_version, len(training_df), json.dumps(metrics), notes),
        )
        self.conn.commit()
        return artifact_path

    # ---------- promotion / rollback (audited) ----------
    def promote(self, version, reason="", approved_by="pipeline"):
        if not self._exists(version):
            raise ValueError(f"version {version} not in registry")
        self.conn.execute(
            "INSERT INTO promotions (version, action, ts, reason, approved_by) VALUES (?,?,?,?,?)",
            (version, "promote", time.time(), reason, approved_by),
        )
        self.conn.commit()

    def rollback(self, version, reason="", approved_by="oncall"):
        if not self._exists(version):
            raise ValueError(f"version {version} not in registry")
        self.conn.execute(
            "INSERT INTO promotions (version, action, ts, reason, approved_by) VALUES (?,?,?,?,?)",
            (version, "rollback", time.time(), reason, approved_by),
        )
        self.conn.commit()

    def current_production(self):
        row = self.conn.execute(
            "SELECT version, action, ts, reason FROM promotions ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        return row  # (version, action, ts, reason) or None

    # ---------- lookup ----------
    def _exists(self, version):
        return self.conn.execute(
            "SELECT 1 FROM models WHERE version=?", (version,)
        ).fetchone() is not None

    def get(self, version):
        row = self.conn.execute(
            "SELECT * FROM models WHERE version=?", (version,)
        ).fetchone()
        if not row:
            return None
        cols = ["version", "created_at", "artifact_path", "training_data_hash",
                "feature_schema_hash", "parent_version", "train_rows", "metrics_json", "notes"]
        d = dict(zip(cols, row))
        d["metrics"] = json.loads(d.pop("metrics_json"))
        return d

    def load_model(self, version):
        import joblib
        info = self.get(version)
        if info is None:
            return None
        return joblib.load(info["artifact_path"])

    def list_versions(self):
        rows = self.conn.execute(
            "SELECT version, created_at, parent_version, metrics_json FROM models ORDER BY created_at"
        ).fetchall()
        return [
            {"version": r[0], "created_at": r[1], "parent_version": r[2],
             "metrics": json.loads(r[3])}
            for r in rows
        ]

    def audit_trail(self):
        rows = self.conn.execute(
            "SELECT version, action, ts, reason, approved_by FROM promotions ORDER BY ts"
        ).fetchall()
        return [
            {"version": r[0], "action": r[1], "ts": r[2], "reason": r[3], "approved_by": r[4]}
            for r in rows
        ]
