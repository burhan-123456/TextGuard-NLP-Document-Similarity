
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


def get_history():
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM comparisons ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def clear_history():
    with get_connection() as conn:
        conn.execute("DELETE FROM comparisons")
