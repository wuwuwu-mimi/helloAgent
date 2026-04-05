from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from memory.base import MemoryItem
from memory.embedding import BaseEmbeddingService


class QdrantVectorStore:
    """
    一个面向 Qdrant 接口风格的最小本地向量存储。

    修改说明：当前先用 JSON 文件把“向量入库 / 相似度检索”跑通，
    这样无需额外依赖也能先把语义记忆接进系统里；
    后续如果要替换成真正的 Qdrant 服务，只需要保留这个类的接口。
    """

    def __init__(
        self,
        store_path: str,
        embedding_service: BaseEmbeddingService,
    ) -> None:
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_service = embedding_service
        self.backend = "json"
        self._ensure_store_file()

    def upsert(self, item: MemoryItem) -> None:
        """写入或更新一条带向量的记忆。"""
        vector = self.embedding_service.embed(item.content)
        records = self._load_records()
        payload = {
            "id": item.id,
            "session_id": item.session_id,
            "vector": vector,
            "memory_item": item.model_dump(mode="json"),
        }
        records = [record for record in records if record["id"] != item.id]
        records.append(payload)
        self._save_records(records)

    def search(
        self,
        session_id: str,
        query: str,
        *,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> List[MemoryItem]:
        """根据 query 做向量相似度检索。"""
        query_vector = self.embedding_service.embed(query)
        scored_items: List[tuple[float, MemoryItem]] = []

        for record in self._load_records():
            if record.get("session_id") != session_id:
                continue
            score = self.embedding_service.cosine_similarity(query_vector, record.get("vector", []))
            if score < score_threshold:
                continue
            item = MemoryItem.model_validate(record["memory_item"])
            item.metadata = {**item.metadata, "semantic_score": round(score, 4)}
            scored_items.append((score, item))

        scored_items.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
        return [item for _, item in scored_items[:limit]]

    def list_recent(self, session_id: str, limit: int = 10) -> List[MemoryItem]:
        """按创建时间返回最近入库的语义记忆。"""
        items = [
            MemoryItem.model_validate(record["memory_item"])
            for record in self._load_records()
            if record.get("session_id") == session_id
        ]
        items.sort(key=lambda item: item.created_at, reverse=True)
        return items[:limit]

    def clear_session(self, session_id: str) -> None:
        """清空某个 session 的语义向量记录。"""
        records = [
            record for record in self._load_records() if record.get("session_id") != session_id
        ]
        self._save_records(records)

    def _ensure_store_file(self) -> None:
        if not self.store_path.exists():
            self.store_path.write_text("[]", encoding="utf-8")

    def _load_records(self) -> List[Dict[str, Any]]:
        raw_text = self.store_path.read_text(encoding="utf-8")
        return json.loads(raw_text or "[]")

    def _save_records(self, records: List[Dict[str, Any]]) -> None:
        self.store_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
