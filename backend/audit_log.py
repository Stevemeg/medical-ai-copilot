"""
Tamper-evident audit logging.

Every answered question is logged to a local SQLite database as a hash
chain: each row's hash is computed over its own content PLUS the previous
row's hash. This means:

  - Modifying any historical row breaks that row's hash AND every
    subsequent row's hash (since they all incorporated the original,
    now-incorrect prev_hash transitively). Tampering with old entries is
    detectable, not just tampering with the most recent one.
  - Deleting a row breaks the chain at that point too, since the next
    remaining row's prev_hash will no longer match any row actually
    present in the table.

This is the same fundamental integrity mechanism a blockchain uses (hash-
linking), without any of the distributed consensus machinery that's
irrelevant for a single local audit log. verify_chain() is the actual
proof this works -- it walks every row and recomputes hashes fresh,
rather than just trusting what's stored.

Deliberately logs NO user identity (there's no auth in this app) and NO
real patient data (this tool is designed for hypothetical/educational
clinical questions, not real patient records) -- this keeps the log
genuinely safe to demonstrate without any privacy concerns of its own.
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "audit_log.db"

GENESIS_HASH = "0" * 64  # the prev_hash value for the very first row


def _get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            entry_hash TEXT NOT NULL
        )
        """
    )
    return conn


def _compute_hash(timestamp: str, question: str, answer: str, sources_json: str, prev_hash: str) -> str:
    """
    Hashes the concatenation of all of this row's content plus the
    previous row's hash. Using a clear separator between fields avoids a
    (very unlikely but real) ambiguity where two different sets of field
    values could concatenate to the same string -- e.g. without a
    separator, question="ab" + answer="c" is indistinguishable from
    question="a" + answer="bc" once concatenated.
    """
    payload = "\x1f".join([timestamp, question, answer, sources_json, prev_hash])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def log_interaction(question: str, answer: str, sources: list) -> None:
    """
    Appends one entry to the audit log. Failures here are caught and
    printed rather than raised -- a logging failure should never prevent
    the user from getting their actual answer.
    """
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        sources_json = json.dumps(sources, ensure_ascii=False)

        conn = _get_connection()
        cur = conn.execute("SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        prev_hash = row[0] if row else GENESIS_HASH

        entry_hash = _compute_hash(timestamp, question, answer, sources_json, prev_hash)

        conn.execute(
            """
            INSERT INTO audit_log (timestamp, question, answer, sources, prev_hash, entry_hash)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, question, answer, sources_json, prev_hash, entry_hash),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[audit_log] Failed to log interaction (non-fatal): {e}")


def verify_chain() -> dict:
    """
    Walks the entire audit log and recomputes every row's hash from
    scratch, confirming it matches what's stored AND that each row's
    prev_hash correctly matches the actual previous row's entry_hash.

    Returns a dict describing the result -- this is the actual proof the
    tamper-evidence property holds, not just a claim. Run this any time
    you want to confirm the log hasn't been modified outside this module.
    """
    conn = _get_connection()
    rows = conn.execute(
        "SELECT id, timestamp, question, answer, sources, prev_hash, entry_hash FROM audit_log ORDER BY id ASC"
    ).fetchall()
    conn.close()

    if not rows:
        return {"valid": True, "rows_checked": 0, "message": "Log is empty -- nothing to verify."}

    expected_prev_hash = GENESIS_HASH
    for row in rows:
        row_id, timestamp, question, answer, sources_json, stored_prev_hash, stored_entry_hash = row

        if stored_prev_hash != expected_prev_hash:
            return {
                "valid": False,
                "rows_checked": row_id,
                "message": f"Chain broken at row {row_id}: prev_hash doesn't match the previous row's actual hash. The log was likely modified or rows were deleted/reordered.",
            }

        recomputed_hash = _compute_hash(timestamp, question, answer, sources_json, stored_prev_hash)
        if recomputed_hash != stored_entry_hash:
            return {
                "valid": False,
                "rows_checked": row_id,
                "message": f"Chain broken at row {row_id}: stored hash doesn't match recomputed hash. This row's content was likely modified after logging.",
            }

        expected_prev_hash = stored_entry_hash

    return {
        "valid": True,
        "rows_checked": len(rows),
        "message": f"All {len(rows)} entries verified -- hash chain is intact.",
    }


if __name__ == "__main__":
    # Quick CLI check: run this file directly to verify the current log
    result = verify_chain()
    print(result["message"])
    print(f"Valid: {result['valid']}, rows checked: {result['rows_checked']}")