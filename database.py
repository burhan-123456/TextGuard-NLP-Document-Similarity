
import sqlite3
from datetime import datetime, timezone

DB_NAME = "textguard.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                document1_words INTEGER,
                document2_words INTEGER,
                tfidf REAL,
                jaccard REAL,
                bigram REAL,
                similarity_level TEXT,
                similar_sentences INTEGER,
                risk_level TEXT,
                threshold INTEGER,
                min_words INTEGER
            )
        """)


def save_comparison(entry):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO comparisons (
                created_at, document1_words, document2_words,
                tfidf, jaccard, bigram, similarity_level,
                similar_sentences, risk_level, threshold, min_words
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            entry["Document 1 Words"],
            entry["Document 2 Words"],
            entry["TF-IDF (%)"],
            entry["Jaccard (%)"],
            entry["Bigram (%)"],
            entry["Level"],
            entry["Similar Sentences"],
            entry["Risk"],
            entry["Threshold (%)"],
            entry["Minimum Words"]
        ))


def get_history(search="", level="All", risk="All"):
    query = "SELECT * FROM comparisons WHERE 1=1"
    params = []

    if search.strip():
        query += " AND CAST(id AS TEXT) LIKE ?"
        params.append(f"%{search.strip()}%")

    if level != "All":
        query += " AND similarity_level = ?"
        params.append(level)

    if risk != "All":
        query += " AND risk_level = ?"
        params.append(risk)

    query += " ORDER BY id DESC"

    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def clear_history():
    with get_connection() as conn:
        conn.execute("DELETE FROM comparisons")
