from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from typing import Deque, Dict, List

from memory.base import BaseMemory, MemoryConfig, MemoryItem


class WorkingMemory(BaseMemory):
    """纯内存工作记忆，负责保存最近一段时间的上下文。"""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._items: Dict[str, Deque[MemoryItem]] = defaultdict(deque)

    def add(self, item: MemoryItem) -> None:
        self._cleanup(item.session_id)
        items = self._items[item.session_id]
        items.append(item)
        while len(items) > self.config.working_memory_max_items:
            items.popleft()

    def recent(self, session_id: str, limit: int = 10) -> List[MemoryItem]:
        self._cleanup(session_id)
        return list(self._items[session_id])[-limit:]

    def search(self, session_id: str, query: str, limit: int = 10) -> List[MemoryItem]:
        self._cleanup(session_id)
        normalized_query = query.strip().lower()
        tokens = [token.lower() for token in query.split() if token.strip()]
        if normalized_query and not tokens:
            # 修改说明：中文查询通常没有空格，这里至少保留完整 query 作为一个匹配单元。
            tokens = [normalized_query]
        if not tokens:
            return self.recent(session_id, limit)

        scored: List[tuple[int, MemoryItem]] = []
        for item in self._items[session_id]:
            haystack = item.content.lower()
            score = sum(token in haystack for token in tokens)
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
        return [item for _, item in scored[:limit]]

    def clear(self, session_id: str) -> None:
        self._items.pop(session_id, None)

    def _cleanup(self, session_id: str) -> None:
        now = datetime.now()
        items = self._items.get(session_id)
        if not items:
            return
        self._items[session_id] = deque(item for item in items if not item.is_expired(now))
