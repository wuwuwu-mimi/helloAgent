from __future__ import annotations

import json
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory.base import MemoryConfig, MemoryItem
from memory.embedding import BaseEmbeddingService

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
except ImportError:  # pragma: no cover - 当前环境可能未安装 qdrant-client
    QdrantClient = None
    qdrant_models = None


class QdrantVectorStore:
    """
    一个支持“真实 Qdrant / 本地 JSON fallback”的向量存储适配层。

    修改说明：这一层现在会优先尝试接入真实 Qdrant，
    但如果本机没有安装 `qdrant-client` 或没有配置 Qdrant 连接信息，
    仍然会自动回退到 JSON 文件，保证开发阶段可以继续跑通。
    """

    def __init__(
        self,
        config: MemoryConfig,
        embedding_service: BaseEmbeddingService,
        *,
        collection_name: str,
        store_path: str,
    ) -> None:
        self.config = config
        self.embedding_service = embedding_service
        self.collection_name = collection_name
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = "json"
        self.client: Any = None
        self._initialize_backend()

    def upsert(self, item: MemoryItem) -> None:
        """兼容语义记忆接口：写入一条记忆。"""
        self.upsert_record(
            record_id=item.id,
            vector=self.embedding_service.embed(item.content),
            payload={
                "session_id": item.session_id,
                "content": item.content,
                "memory_item": item.model_dump(mode="json"),
                "created_at": item.created_at.isoformat(),
            },
        )

    def search(
        self,
        session_id: str,
        query: str,
        *,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> List[MemoryItem]:
        """兼容语义记忆接口：按 query 检索记忆。"""
        matches = self.search_records(
            vector=self.embedding_service.embed(query),
            limit=limit,
            score_threshold=score_threshold,
            filters={"session_id": session_id},
        )
        items: List[MemoryItem] = []
        for match in matches:
            payload = match["payload"]
            if "memory_item" not in payload:
                continue
            item = MemoryItem.model_validate(payload["memory_item"])
            item.metadata = {
                **item.metadata,
                "semantic_score": round(match["score"], 4),
                "vector_backend": self.backend,
            }
            items.append(item)
        return items

    def list_recent(self, session_id: str, limit: int = 10) -> List[MemoryItem]:
        """兼容语义记忆接口：获取最近入库的记忆。"""
        records = self.list_recent_records(limit=limit, filters={"session_id": session_id})
        items: List[MemoryItem] = []
        for record in records:
            payload = record["payload"]
            if "memory_item" not in payload:
                continue
            items.append(MemoryItem.model_validate(payload["memory_item"]))
        return items

    def clear_session(self, session_id: str) -> None:
        """兼容语义记忆接口：清空某个 session 的向量记录。"""
        self.clear_records(filters={"session_id": session_id})

    def upsert_record(
        self,
        *,
        record_id: str,
        vector: List[float],
        payload: Dict[str, Any],
    ) -> None:
        """写入一条通用向量记录。"""
        if self.backend == "qdrant":
            self._qdrant_upsert(record_id=record_id, vector=vector, payload=payload)
            return

        records = self._load_json_records()
        new_record = {
            "id": record_id,
            "vector": vector,
            "payload": payload,
        }
        records = [record for record in records if record["id"] != record_id]
        records.append(new_record)
        self._save_json_records(records)

    def search_records(
        self,
        *,
        vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """检索通用向量记录，返回统一格式的命中结果。"""
        if self.backend == "qdrant":
            return self._qdrant_search(
                vector=vector,
                limit=limit,
                score_threshold=score_threshold,
                filters=filters or {},
            )

        scored_items: List[Dict[str, Any]] = []
        for record in self._load_json_records():
            payload = record.get("payload", {})
            if not self._match_filters(payload, filters or {}):
                continue
            score = self.embedding_service.cosine_similarity(vector, record.get("vector", []))
            if score < score_threshold:
                continue
            scored_items.append(
                {
                    "id": record["id"],
                    "score": score,
                    "payload": payload,
                }
            )

        scored_items.sort(
            key=lambda item: (item["score"], item["payload"].get("created_at", "")),
            reverse=True,
        )
        return scored_items[:limit]

    def list_recent_records(
        self,
        *,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """按 payload.created_at 返回最近的通用向量记录。"""
        if self.backend == "qdrant":
            return self._qdrant_list_recent(limit=limit, filters=filters or {})

        records = [
            record
            for record in self._load_json_records()
            if self._match_filters(record.get("payload", {}), filters or {})
        ]
        records.sort(key=lambda item: item.get("payload", {}).get("created_at", ""), reverse=True)
        return records[:limit]

    def clear_records(self, *, filters: Optional[Dict[str, Any]] = None) -> None:
        """清空满足过滤条件的向量记录。"""
        if self.backend == "qdrant":
            self._qdrant_clear(filters=filters or {})
            return

        kept = [
            record
            for record in self._load_json_records()
            if not self._match_filters(record.get("payload", {}), filters or {})
        ]
        self._save_json_records(kept)

    def _initialize_backend(self) -> None:
        requested_backend = (self.config.vector_store_backend or "auto").strip().lower()
        if requested_backend in {"json", "file"}:
            self.backend = "json"
            self._ensure_json_store()
            return

        if self._can_use_qdrant():
            self.backend = "qdrant"
            return

        self.backend = "json"
        self._ensure_json_store()

    def _can_use_qdrant(self) -> bool:
        if QdrantClient is None or qdrant_models is None:
            return False

        url = (self.config.qdrant_url or "").strip()
        local_path = (self.config.qdrant_local_path or "").strip()
        if not url and not local_path:
            return False

        try:
            if url:
                self.client = QdrantClient(
                    url=url,
                    api_key=(self.config.qdrant_api_key or "").strip() or None,
                    timeout=10.0,
                )
            else:
                self.client = QdrantClient(path=local_path)
            self._ensure_qdrant_collection()
            return True
        except Exception:
            self.client = None
            return False

    def _ensure_qdrant_collection(self) -> None:
        if self.client is None or qdrant_models is None:
            return
        collections = self.client.get_collections().collections
        if any(item.name == self.collection_name for item in collections):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=self.config.embedding_dimensions,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    def _qdrant_upsert(self, *, record_id: str, vector: List[float], payload: Dict[str, Any]) -> None:
        if self.client is None:
            raise RuntimeError("Qdrant client is not initialized.")
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                {
                    "id": record_id,
                    "vector": vector,
                    "payload": payload,
                }
            ],
        )

    def _qdrant_search(
        self,
        *,
        vector: List[float],
        limit: int,
        score_threshold: float,
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if self.client is None or qdrant_models is None:
            return []
        query_filter = self._build_qdrant_filter(filters)
        points = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
        )
        return [
            {
                "id": str(point.id),
                "score": float(point.score),
                "payload": dict(point.payload or {}),
            }
            for point in points
        ]

    def _qdrant_list_recent(self, *, limit: int, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.client is None:
            return []
        all_points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self._build_qdrant_filter(filters),
            limit=max(limit, 100),
            with_payload=True,
            with_vectors=False,
        )
        items = [
            {
                "id": str(point.id),
                "payload": dict(point.payload or {}),
            }
            for point in all_points
        ]
        items.sort(key=lambda item: item["payload"].get("created_at", ""), reverse=True)
        return items[:limit]

    def _qdrant_clear(self, *, filters: Dict[str, Any]) -> None:
        if self.client is None or qdrant_models is None:
            return
        if not filters:
            self.client.delete_collection(self.collection_name)
            self._ensure_qdrant_collection()
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.FilterSelector(
                filter=self._build_qdrant_filter(filters),
            ),
        )

    def _build_qdrant_filter(self, filters: Dict[str, Any]) -> Any:
        if not filters or qdrant_models is None:
            return None
        conditions = [
            qdrant_models.FieldCondition(
                key=key,
                match=qdrant_models.MatchValue(value=value),
            )
            for key, value in filters.items()
        ]
        return qdrant_models.Filter(must=conditions)

    def _ensure_json_store(self) -> None:
        if not self.store_path.exists():
            self.store_path.write_text("[]", encoding="utf-8")

    def _load_json_records(self) -> List[Dict[str, Any]]:
        raw_text = self.store_path.read_text(encoding="utf-8") if self.store_path.exists() else "[]"
        payload = json.loads(raw_text or "[]")
        normalized_records: List[Dict[str, Any]] = []
        changed = False
        for record in payload:
            normalized = self._normalize_record(record)
            if normalized != record:
                changed = True
            normalized_records.append(normalized)
        if changed:
            self._save_json_records(normalized_records)
        return normalized_records

    def _save_json_records(self, records: List[Dict[str, Any]]) -> None:
        self.store_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
        if "id" in record and "payload" in record:
            return record

        if "memory_item" in record:
            memory_item = record["memory_item"]
            return {
                "id": record.get("id") or memory_item.get("id"),
                "vector": record.get("vector", []),
                "payload": {
                    "session_id": record.get("session_id") or memory_item.get("session_id"),
                    "content": memory_item.get("content", ""),
                    "memory_item": memory_item,
                    "created_at": memory_item.get("created_at", ""),
                },
            }

        if "chunk" in record:
            chunk = record["chunk"]
            chunk_id = chunk.get("chunk_id") or md5(
                f"{chunk.get('source', '')}:{chunk.get('content', '')}".encode("utf-8")
            ).hexdigest()
            chunk = {**chunk, "chunk_id": chunk_id}
            return {
                "id": record.get("id") or chunk_id,
                "vector": record.get("vector", []),
                "payload": {
                    "source": chunk.get("source", ""),
                    "chunk_id": chunk_id,
                    "content": chunk.get("content", ""),
                    "created_at": chunk_id,
                    "chunk": chunk,
                },
            }

        generated_id = md5(json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return {
            "id": record.get("id") or generated_id,
            "vector": record.get("vector", []),
            "payload": record.get("payload", {}),
        }

    @staticmethod
    def _match_filters(payload: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if payload.get(key) != value:
                return False
        return True
