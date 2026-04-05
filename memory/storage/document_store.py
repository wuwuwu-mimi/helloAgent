from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List

from memory.base import MemoryItem


class DocumentStore:
    """基于 SQLite 的轻量文档存储，用于记忆持久化。"""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.fallback_path = self.db_path.with_suffix(".json")
        self.backend = "sqlite"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_items (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        expires_at TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memory_session_created
                    ON memory_items(session_id, created_at)
                    """
                )
        except sqlite3.OperationalError:
            # 修改说明：某些受限环境里 SQLite 可能无法正常落盘，
            # 这里自动回退到 JSON 文件存储，保证记忆功能仍然可用。
            self.backend = "json"
            if not self.fallback_path.exists():
                self.fallback_path.write_text("[]", encoding="utf-8")

    def add_item(self, item: MemoryItem) -> None:
        """写入一条持久化记忆。"""
        if self.backend == "json":
            items = self._load_json_items()
            items = [current for current in items if current.id != item.id]
            items.append(item)
            self._save_json_items(items)
            return

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO memory_items (
                    id, session_id, role, content, memory_type, metadata, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.session_id,
                    item.role,
                    item.content,
                    item.memory_type,
                    json.dumps(item.metadata, ensure_ascii=False),
                    item.created_at.isoformat(),
                    item.expires_at.isoformat() if item.expires_at else None,
                ),
            )

    def list_recent(self, session_id: str, limit: int = 10) -> List[MemoryItem]:
        """读取最近若干条记忆。"""
        if self.backend == "json":
            items = [item for item in self._load_json_items() if item.session_id == session_id]
            items.sort(key=lambda item: item.created_at, reverse=True)
            return items[:limit]

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM memory_items
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def search_items(self, session_id: str, query: str, limit: int = 10) -> List[MemoryItem]:
        """通过 SQLite LIKE 做一个最小可用的记忆检索。"""
        if self.backend == "json":
            normalized_query = query.strip().lower()
            items = [
                item
                for item in self._load_json_items()
                if item.session_id == session_id and normalized_query in item.content.lower()
            ]
            items.sort(key=lambda item: item.created_at, reverse=True)
            return items[:limit]

        like_query = f"%{query.strip()}%"
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM memory_items
                WHERE session_id = ?
                  AND content LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, like_query, limit),
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def clear_session(self, session_id: str) -> None:
        """清空指定 session 的所有持久化记忆。"""
        if self.backend == "json":
            items = [item for item in self._load_json_items() if item.session_id != session_id]
            self._save_json_items(items)
            return

        with self._connect() as connection:
            connection.execute(
                "DELETE FROM memory_items WHERE session_id = ?",
                (session_id,),
            )

    def _load_json_items(self) -> List[MemoryItem]:
        raw_text = self.fallback_path.read_text(encoding="utf-8") if self.fallback_path.exists() else "[]"
        payload = json.loads(raw_text or "[]")
        return [MemoryItem.model_validate(item) for item in payload]

    def _save_json_items(self, items: List[MemoryItem]) -> None:
        payload = [item.model_dump(mode="json") for item in items]
        self.fallback_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            memory_type=row["memory_type"],
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )
