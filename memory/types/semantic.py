from __future__ import annotations

from typing import List

from memory.base import BaseMemory, MemoryConfig, MemoryItem
from memory.embedding import BaseEmbeddingService
from memory.storage.qdrant_store import QdrantVectorStore


class SemanticMemory(BaseMemory):
    """基于向量检索的最小语义记忆实现。"""

    def __init__(
        self,
        config: MemoryConfig,
        store: QdrantVectorStore,
        embedding_service: BaseEmbeddingService,
    ) -> None:
        self.config = config
        self.store = store
        self.embedding_service = embedding_service

    def add(self, item: MemoryItem) -> None:
        self.store.upsert(item)

    def recent(self, session_id: str, limit: int = 10) -> List[MemoryItem]:
        return list(reversed(self.store.list_recent(session_id, limit)))

    def search(self, session_id: str, query: str, limit: int = 10) -> List[MemoryItem]:
        if not query.strip():
            return self.recent(session_id, limit)
        return list(
            reversed(
                self.store.search(
                    session_id,
                    query,
                    limit=limit,
                    score_threshold=self.config.semantic_score_threshold,
                )
            )
        )

    def clear(self, session_id: str) -> None:
        self.store.clear_session(session_id)
