import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np


class SQLiteDocumentStore:
    """Small persistent store for document text, metadata and float32 embeddings."""

    def __init__(self, database_path: str):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents(updated_at)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def upsert_many(self, documents: list[dict[str, Any]], vectors: np.ndarray) -> int:
        if len(documents) != len(vectors):
            raise ValueError("Each document must have exactly one embedding.")
        rows = []
        for document, vector in zip(documents, vectors):
            normalized = np.asarray(vector, dtype=np.float32).reshape(-1)
            if normalized.size == 0 or not np.isfinite(normalized).all():
                raise ValueError("Embeddings must be non-empty finite vectors.")
            rows.append((
                document["id"],
                document["text"],
                json.dumps(document.get("metadata", {}), ensure_ascii=False),
                normalized.tobytes(),
                int(normalized.size),
            ))
        with self._connect() as connection:
            connection.executemany("""
                INSERT INTO documents (id, text, metadata_json, embedding, dimension)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    text=excluded.text,
                    metadata_json=excluded.metadata_json,
                    embedding=excluded.embedding,
                    dimension=excluded.dimension,
                    updated_at=CURRENT_TIMESTAMP
            """, rows)
        return len(rows)

    def list_documents(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, text, metadata_json FROM documents ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"id": row["id"], "text": row["text"], "metadata": json.loads(row["metadata_json"])}
            for row in rows
        ]

    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0 or not np.isfinite(query).all():
            raise ValueError("Query embedding must be a non-zero finite vector.")
        query = query / query_norm

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, text, metadata_json, embedding, dimension FROM documents"
            ).fetchall()

        scored: list[dict[str, Any]] = []
        for row in rows:
            if row["dimension"] != query.size:
                continue
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if norm == 0:
                continue
            score = float(np.dot(query, vector / norm))
            scored.append({
                "id": row["id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata_json"]),
                "score": score,
            })
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def delete(self, document_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return cursor.rowcount > 0

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM documents").fetchone()
        return int(row["count"])
