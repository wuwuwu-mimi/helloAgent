from __future__ import annotations

from typing import List

from memory.base import BaseMemory, MemoryItem
from memory.storage.document_store import DocumentStore


class EpisodicMemory(BaseMemory):
    """情景记忆：把会话中的关键消息持久化到 SQLite。"""

    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    def add(self, item: MemoryItem) -> None:
        self.store.add_item(item)

    def recent(self, session_id: str, limit: int = 10) -> List[MemoryItem]:
        return list(reversed(self.store.list_recent(session_id, limit)))

    def search(self, session_id: str, query: str, limit: int = 10) -> List[MemoryItem]:
        return list(reversed(self.store.search_items(session_id, query, limit)))

    def list_all(self, session_id: str) -> List[MemoryItem]:
        """返回某个 session 的全部情景记忆，供保留策略统一裁剪。"""
        return self.store.list_session_items(session_id)

    def prune(self, session_id: str, keep_ids: List[str]) -> int:
        """裁剪情景记忆，只保留 keep_ids 中的条目。"""
        return self.store.prune_session(session_id, keep_ids)

    def clear(self, session_id: str) -> None:
        self.store.clear_session(session_id)
