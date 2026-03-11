import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


DB_PATH = Path(__file__).with_name("interview.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                location TEXT NOT NULL DEFAULT '',
                required_skills TEXT NOT NULL DEFAULT '',
                jd_text TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS interviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_name TEXT NOT NULL DEFAULT '',
                candidate_email TEXT NOT NULL DEFAULT '',
                job_id INTEGER NOT NULL,
                input_mode TEXT NOT NULL DEFAULT 'both',
                resume_json TEXT NOT NULL DEFAULT '{}',
                transcript_json TEXT NOT NULL DEFAULT '[]',
                violations_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'active',
                started_at TEXT NOT NULL,
                ended_at TEXT,
                result_json TEXT,
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            );
            """
        )


def seed_default_jobs_if_empty() -> None:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(1) AS c FROM jobs").fetchone()
        if row and int(row["c"]) > 0:
            return
        now = datetime.utcnow().isoformat()
        conn.execute(
            """
            INSERT INTO jobs (title, location, required_skills, jd_text, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "Python Developer",
                "Remote",
                "Python, APIs, SQL",
                "Build backend services in Python, integrate APIs, and work with SQL databases.",
                now,
            ),
        )


def list_jobs() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, location, required_skills, jd_text FROM jobs ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_job(job_id: int) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, title, location, required_skills, jd_text FROM jobs WHERE id=?",
            (job_id,),
        ).fetchone()
        return dict(row) if row else None


def upsert_job(job_id: Optional[int], title: str, location: str, required_skills: str, jd_text: str) -> int:
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        if job_id:
            conn.execute(
                """
                UPDATE jobs
                SET title=?, location=?, required_skills=?, jd_text=?
                WHERE id=?
                """,
                (title, location, required_skills, jd_text, job_id),
            )
            return int(job_id)
        cur = conn.execute(
            """
            INSERT INTO jobs (title, location, required_skills, jd_text, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (title, location, required_skills, jd_text, now),
        )
        return int(cur.lastrowid)


def create_interview(
    *,
    candidate_name: str,
    candidate_email: str,
    job_id: int,
    input_mode: str,
    resume_data: dict[str, Any],
    transcript: list[dict[str, Any]],
    violations: dict[str, Any],
) -> int:
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO interviews (
                candidate_name, candidate_email, job_id, input_mode,
                resume_json, transcript_json, violations_json,
                status, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)
            """,
            (
                candidate_name,
                candidate_email,
                job_id,
                input_mode,
                json.dumps(resume_data, ensure_ascii=False),
                json.dumps(transcript, ensure_ascii=False),
                json.dumps(violations, ensure_ascii=False),
                now,
            ),
        )
        return int(cur.lastrowid)


def update_interview_transcript(interview_id: int, transcript: list[dict[str, Any]], violations: dict[str, Any]) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE interviews
            SET transcript_json=?, violations_json=?
            WHERE id=?
            """,
            (
                json.dumps(transcript, ensure_ascii=False),
                json.dumps(violations, ensure_ascii=False),
                interview_id,
            ),
        )


def complete_interview(interview_id: int, result: dict[str, Any]) -> None:
    ended = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE interviews
            SET status='completed', ended_at=?, result_json=?
            WHERE id=?
            """,
            (ended, json.dumps(result, ensure_ascii=False), interview_id),
        )


def list_completed_interviews_for_job(job_id: Optional[int] = None) -> list[dict[str, Any]]:
    with _connect() as conn:
        if job_id:
            rows = conn.execute(
                """
                SELECT i.id, i.candidate_name, i.candidate_email, i.job_id, i.ended_at, i.result_json,
                       j.title AS job_title
                FROM interviews i
                JOIN jobs j ON j.id=i.job_id
                WHERE i.status='completed' AND i.job_id=?
                ORDER BY i.ended_at DESC
                """,
                (job_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT i.id, i.candidate_name, i.candidate_email, i.job_id, i.ended_at, i.result_json,
                       j.title AS job_title
                FROM interviews i
                JOIN jobs j ON j.id=i.job_id
                WHERE i.status='completed'
                ORDER BY i.ended_at DESC
                """
            ).fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        try:
            d["result"] = json.loads(d.get("result_json") or "{}")
        except Exception:
            d["result"] = {}
        out.append(d)
    return out

