from __future__ import annotations

from typing import List

from memory.base import BaseMemory, MemoryConfig, MemoryItem
from memory.embedding import BaseEmbeddingService
from memory.storage.neo4j_store import Neo4jGraphStore
from memory.storage.qdrant_store import QdrantVectorStore


class SemanticMemory(BaseMemory):
    """基于“向量检索 + 图谱检索”的最小语义记忆实现。"""

    def __init__(
        self,
        config: MemoryConfig,
        store: QdrantVectorStore,
        graph_store: Neo4jGraphStore,
        embedding_service: BaseEmbeddingService,
    ) -> None:
        self.config = config
        self.store = store
        self.graph_store = graph_store
        self.embedding_service = embedding_service

    def add(self, item: MemoryItem) -> None:
        self.store.upsert(item)
        entities = self.graph_store.extract_entities(item.content)
        relations = self.graph_store.extract_relations(item.content, entities)
        relation_entities = [
            value
            for relation in relations
            for key, value in relation.items()
            if key in {"source", "target"} and value
        ]
        entities = list(dict.fromkeys(entities + relation_entities))
        self.graph_store.upsert_memory(item, entities, relations)

    def recent(self, session_id: str, limit: int = 10) -> List[MemoryItem]:
        vector_items = self.store.list_recent(session_id, limit)
        graph_items = self.graph_store.list_recent(
            session_id,
            min(limit, self.config.graph_recall_top_k),
        )
        return list(reversed(self._merge_items(vector_items + graph_items, limit)))

    def search(self, session_id: str, query: str, limit: int = 10) -> List[MemoryItem]:
        if not query.strip():
            return self.recent(session_id, limit)
        vector_items = self.store.search(
            session_id,
            query,
            limit=limit,
            score_threshold=self.config.semantic_score_threshold,
        )
        graph_items = self.graph_store.search_related(
            session_id,
            query,
            limit=min(limit, self.config.graph_recall_top_k),
        )
        return list(reversed(self._merge_items(vector_items + graph_items, limit)))

    def clear(self, session_id: str) -> None:
        self.store.clear_session(session_id)
        self.graph_store.clear_session(session_id)

    @staticmethod
    def _merge_items(items: List[MemoryItem], limit: int) -> List[MemoryItem]:
        merged: List[MemoryItem] = []
        seen: set[tuple[str, str, str]] = set()
        for item in sorted(items, key=lambda current: current.created_at):
            key = (item.role, item.content, item.created_at.isoformat())
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged[-limit:]
