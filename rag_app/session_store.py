"""Local SQLite conversations and execution records, committed atomically."""

import json
import sqlite3
from pathlib import Path
from uuid import uuid4


class SessionStore:
    def __init__(self, path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA secure_delete=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL, content TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, id);
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                engine TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE INDEX IF NOT EXISTS runs_session ON runs(session_id, created_at);
        """)
        # This service uses one process. A running record at startup was interrupted.
        with self.db:
            self.db.execute("UPDATE runs SET status='interrupted' WHERE status='running'")

    def close(self):
        self.db.close()

    def exists(self, session_id):
        return self.db.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is not None

    def history(self, session_id):
        return [dict(row) for row in self.db.execute(
            "SELECT role,content FROM messages WHERE session_id=? ORDER BY id", (session_id,))]

    def start_run(self, session_id, question, engine):
        run_id = str(uuid4())
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO sessions(id) VALUES (?)", (session_id,))
            self.db.execute("INSERT INTO runs(id,session_id,engine,status,payload) VALUES (?,?,?,?,?)", (
                run_id, session_id, engine, "running",
                json.dumps({"question": question, "trace": []}, ensure_ascii=False)))
        return run_id

    def progress(self, session_id, run_id, question, trace, usage):
        with self.db:
            self.db.execute("UPDATE runs SET payload=? WHERE id=? AND session_id=? AND status='running'", (
                json.dumps({"question": question, "trace": trace, "usage": usage}, ensure_ascii=False),
                run_id, session_id))

    def save_turn(self, session_id, question, result, engine, commit_messages=True, run_id=None):
        run_id = run_id or str(uuid4())
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO sessions(id) VALUES (?)", (session_id,))
            if commit_messages:
                self.db.executemany("INSERT INTO messages(session_id,role,content) VALUES (?,?,?)", [
                    (session_id, "user", question), (session_id, "assistant", result.answer)])
            self.db.execute("""INSERT INTO runs(id,session_id,engine,status,payload) VALUES (?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET status=excluded.status,payload=excluded.payload""", (
                run_id, session_id, engine, result.stop_reason,
                json.dumps({"question": question, "answer": result.answer,
                            "evidence": result.evidence, "trace": result.trace,
                            "usage": result.usage}, ensure_ascii=False)))
        return run_id

    def recent_runs(self, session_id):
        return [dict(row) for row in self.db.execute(
            "SELECT id,engine,status,created_at FROM runs WHERE session_id=? ORDER BY rowid DESC LIMIT 20",
            (session_id,))]

    def run(self, session_id, run_id):
        row = self.db.execute("SELECT * FROM runs WHERE id=? AND session_id=?", (run_id, session_id)).fetchone()
        if row is None:
            return None
        return {"id": row["id"], "engine": row["engine"], "status": row["status"],
                "created_at": row["created_at"], **json.loads(row["payload"])}

    def delete(self, session_id):
        with self.db:
            cursor = self.db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        return cursor.rowcount > 0
