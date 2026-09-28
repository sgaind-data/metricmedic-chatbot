import sqlite3
from datetime import datetime
from pathlib import Path


DB_PATH = Path(__file__).parent / "metricmedic_chats.db"


class ChatStore:
    """Persistent local storage for MetricMedic conversations."""

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._create_tables()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _create_tables(self):
        """Create chat and message tables if they do not already exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES chats(id)
                )
                """
            )

    def create_chat(self, title="New Chat"):
        """Create a new conversation and return its ID."""
        now = datetime.now().isoformat()

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chats (title, created_at, updated_at)
                VALUES (?, ?, ?)
                """,
                (title, now, now),
            )

            return cursor.lastrowid

    def update_chat_title(self, chat_id, title):
        """Update a conversation title."""
        now = datetime.now().isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE chats
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (title, now, chat_id),
            )

    def add_message(self, chat_id, role, content):
        """Store one user or assistant message."""
        now = datetime.now().isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    chat_id,
                    role,
                    content,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, role, content, now),
            )

            conn.execute(
                """
                UPDATE chats
                SET updated_at = ?
                WHERE id = ?
                """,
                (now, chat_id),
            )

    def get_chats(self):
        """Return conversations ordered by most recent activity."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM chats
                ORDER BY updated_at DESC
                """
            )

            return cursor.fetchall()

    def get_messages(self, chat_id):
        """Return all messages for a conversation."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE chat_id = ?
                ORDER BY id ASC
                """,
                (chat_id,),
            )

            return [
                {
                    "role": row[0],
                    "content": row[1],
                }
                for row in cursor.fetchall()
            ]

    def delete_chat(self, chat_id):
        """Delete a conversation and all of its messages."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM messages WHERE chat_id = ?",
                (chat_id,),
            )

            conn.execute(
                "DELETE FROM chats WHERE id = ?",
                (chat_id,),
            )