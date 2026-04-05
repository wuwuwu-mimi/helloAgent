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
            self._activate_json_fallback()

    def add_item(self, item: MemoryItem) -> None:
        """写入一条持久化记忆。"""
        if self.backend == "json":
            items = self._load_json_items()
            items = [current for current in items if current.id != item.id]
            items.append(item)
            self._save_json_items(items)
            return

        try:
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
        except sqlite3.OperationalError:
            # 修改说明：有些 Windows / 沙箱环境会在运行时突然出现 disk I/O error，
            # 这里即时切回 JSON fallback，避免一次 SQLite 异常直接让整条 Agent 链路中断。
            self._activate_json_fallback()
            self.add_item(item)

    def list_recent(self, session_id: str, limit: int = 10) -> List[MemoryItem]:
        """读取最近若干条记忆。"""
        if self.backend == "json":
            items = [item for item in self._load_json_items() if item.session_id == session_id]
            items.sort(key=lambda item: item.created_at, reverse=True)
            return items[:limit]

        try:
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
        except sqlite3.OperationalError:
            self._activate_json_fallback()
            return self.list_recent(session_id, limit)
        return [self._row_to_item(row) for row in rows]

    def list_session_items(self, session_id: str) -> List[MemoryItem]:
        """读取某个 session 的全部持久化记忆，供保留策略统一裁剪。"""
        if self.backend == "json":
            items = [item for item in self._load_json_items() if item.session_id == session_id]
            items.sort(key=lambda item: item.created_at)
            return items

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT * FROM memory_items
                    WHERE session_id = ?
                    ORDER BY created_at ASC
                    """,
                    (session_id,),
                ).fetchall()
        except sqlite3.OperationalError:
            self._activate_json_fallback()
            return self.list_session_items(session_id)
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
        try:
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
        except sqlite3.OperationalError:
            self._activate_json_fallback()
            return self.search_items(session_id, query, limit)
        return [self._row_to_item(row) for row in rows]

    def clear_session(self, session_id: str) -> None:
        """清空指定 session 的所有持久化记忆。"""
        if self.backend == "json":
            items = [item for item in self._load_json_items() if item.session_id != session_id]
            self._save_json_items(items)
            return

        try:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM memory_items WHERE session_id = ?",
                    (session_id,),
                )
        except sqlite3.OperationalError:
            self._activate_json_fallback()
            self.clear_session(session_id)

    def prune_session(self, session_id: str, keep_ids: List[str]) -> int:
        """只保留指定 id 的情景记忆，返回本次裁剪掉的条数。"""
        keep_id_set = set(keep_ids)
        if self.backend == "json":
            items = self._load_json_items()
            kept_items = [
                item
                for item in items
                if item.session_id != session_id or item.id in keep_id_set
            ]
            pruned_count = len(items) - len(kept_items)
            if pruned_count > 0:
                self._save_json_items(kept_items)
            return pruned_count

        try:
            with self._connect() as connection:
                existing_rows = connection.execute(
                    "SELECT id FROM memory_items WHERE session_id = ?",
                    (session_id,),
                ).fetchall()
                existing_ids = [str(row["id"]) for row in existing_rows]
                drop_ids = [item_id for item_id in existing_ids if item_id not in keep_id_set]
                if not drop_ids:
                    return 0
                placeholders = ", ".join("?" for _ in drop_ids)
                connection.execute(
                    f"DELETE FROM memory_items WHERE session_id = ? AND id IN ({placeholders})",
                    [session_id, *drop_ids],
                )
                return len(drop_ids)
        except sqlite3.OperationalError:
            self._activate_json_fallback()
            return self.prune_session(session_id, keep_ids)

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

    def _activate_json_fallback(self) -> None:
        """切换到 JSON fallback，并确保 fallback 文件存在。"""
        self.backend = "json"
        if not self.fallback_path.exists():
            self.fallback_path.write_text("[]", encoding="utf-8")

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
