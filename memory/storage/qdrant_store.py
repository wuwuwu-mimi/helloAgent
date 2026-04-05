from __future__ import annotations

import json
import logging
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import NAMESPACE_URL, UUID, uuid5

from memory.base import MemoryConfig, MemoryItem
from memory.embedding import BaseEmbeddingService

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
except ImportError:  # pragma: no cover - 当前环境可能未安装 qdrant-client
    QdrantClient = None
    qdrant_models = None

logger = logging.getLogger(__name__)


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

    def list_session_items(self, session_id: str) -> List[MemoryItem]:
        """读取某个 session 的全部语义记忆，供长期保留策略统一裁剪。"""
        records = self.list_all_records(filters={"session_id": session_id})
        items: List[MemoryItem] = []
        for record in records:
            payload = record.get("payload", {})
            if "memory_item" not in payload:
                continue
            items.append(MemoryItem.model_validate(payload["memory_item"]))
        items.sort(key=lambda item: item.created_at)
        return items

    def clear_session(self, session_id: str) -> None:
        """兼容语义记忆接口：清空某个 session 的向量记录。"""
        self.clear_records(filters={"session_id": session_id})

    def prune_session(self, session_id: str, keep_ids: List[str]) -> int:
        """只保留指定 id 的语义记忆，返回本次裁剪掉的条数。"""
        keep_id_set = set(keep_ids)
        if self.backend == "qdrant":
            return self._qdrant_prune_session(session_id=session_id, keep_ids=keep_id_set)

        records = self._load_json_records()
        kept_records = []
        pruned_count = 0
        for record in records:
            payload = record.get("payload", {})
            record_session_id = payload.get("session_id")
            record_id = str(payload.get("_record_id") or record.get("id") or "")
            if record_session_id == session_id and record_id not in keep_id_set:
                pruned_count += 1
                continue
            kept_records.append(record)
        if pruned_count > 0:
            self._save_json_records(kept_records)
        return pruned_count

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

    def list_all_records(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """读取满足过滤条件的全部记录，主要给保留策略使用。"""
        if self.backend == "qdrant":
            return self._qdrant_scroll_all(filters=filters or {})
        return [
            record
            for record in self._load_json_records()
            if self._match_filters(record.get("payload", {}), filters or {})
        ]

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
        except Exception as exc:
            logger.warning("Qdrant 初始化失败，已回退到 JSON 存储：%s", exc)
            self.client = None
            return False

    def _ensure_qdrant_collection(self) -> None:
        if self.client is None or qdrant_models is None:
            return
        collections = self.client.get_collections().collections
        expected_size = self._resolve_vector_size()
        if not any(item.name == self.collection_name for item in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=expected_size,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
        else:
            collection_info = self.client.get_collection(self.collection_name)
            current_size = self._extract_collection_vector_size(collection_info)
            if current_size is not None and current_size != expected_size:
                raise ValueError(
                    f"Qdrant collection `{self.collection_name}` 的向量维度是 {current_size}，"
                    f"但当前 embedding 维度是 {expected_size}。"
                    "请使用单独的 collection 名称，或切回匹配的 embedding 配置。"
                )

        # 修改说明：云端 Qdrant 对过滤字段通常要求先建 payload index，
        # 否则按 `session_id` / `source` 过滤删除或检索时会直接报 400。
        for field_name in ("session_id", "source"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                # 索引已存在或当前 collection 不需要该字段时，忽略即可。
                continue

    def _qdrant_upsert(self, *, record_id: str, vector: List[float], payload: Dict[str, Any]) -> None:
        if self.client is None:
            raise RuntimeError("Qdrant client is not initialized.")
        point_id = self._normalize_qdrant_point_id(record_id)
        normalized_payload = dict(payload)
        # 修改说明：云端 Qdrant 只接受 uint 或 UUID 作为 point id，
        # 这里统一把业务侧的字符串 record_id 映射成稳定 UUID，同时把原始 id 保存在 payload 里。
        normalized_payload.setdefault("_record_id", record_id)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": normalized_payload,
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
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(response, "points", []) or []
        return [
            {
                "id": str((point.payload or {}).get("_record_id", point.id)),
                "score": float(point.score),
                "payload": dict(point.payload or {}),
            }
            for point in points
        ]

    def _qdrant_list_recent(self, *, limit: int, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.client is None:
            return []
        items = self._qdrant_scroll_all(filters=filters)
        items.sort(key=lambda item: item["payload"].get("created_at", ""), reverse=True)
        return items[:limit]

    def _normalize_qdrant_point_id(self, record_id: str) -> str | int:
        """
        把业务层 record_id 规范成 Qdrant 可接受的 point id。

        修改说明：优先保留原本就是整数或 UUID 的 id；
        其他字符串则稳定映射到 UUID5，避免云端 Qdrant 因非法 point id 拒绝写入。
        """
        text = str(record_id).strip()
        if text.isdigit():
            return int(text)

        try:
            return str(UUID(text))
        except ValueError:
            stable_key = f"{self.collection_name}:{text}"
            return str(uuid5(NAMESPACE_URL, stable_key))

    def _resolve_vector_size(self) -> int:
        """
        解析当前 collection 应使用的向量维度。

        修改说明：接入 Ollama embedding 后，维度可能不再是固定的 96；
        这里优先读取 embedding service 的维度提示，避免 Qdrant collection 建错尺寸。
        """
        hinted_size = self.embedding_service.dimension_hint()
        if hinted_size and hinted_size > 0:
            return hinted_size
        return self.config.embedding_dimensions

    @staticmethod
    def _extract_collection_vector_size(collection_info: Any) -> int | None:
        """尽量从 Qdrant collection 配置里提取当前向量维度。"""
        vectors = getattr(getattr(collection_info, "config", None), "params", None)
        vectors = getattr(vectors, "vectors", None)
        if vectors is None:
            return None
        if hasattr(vectors, "size"):
            return int(vectors.size)
        if isinstance(vectors, dict):
            first_value = next(iter(vectors.values()), None)
            if first_value is not None and hasattr(first_value, "size"):
                return int(first_value.size)
        return None

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

    def _qdrant_prune_session(self, *, session_id: str, keep_ids: set[str]) -> int:
        if self.client is None or qdrant_models is None:
            return 0

        points = self._qdrant_scroll_all(filters={"session_id": session_id}, include_point_id=True)
        drop_point_ids = [
            point["point_id"]
            for point in points
            if str(point["payload"].get("_record_id") or point.get("id") or "") not in keep_ids
        ]
        if not drop_point_ids:
            return 0

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.PointIdsList(points=drop_point_ids),
        )
        return len(drop_point_ids)

    def _qdrant_scroll_all(
        self,
        *,
        filters: Dict[str, Any],
        include_point_id: bool = False,
    ) -> List[Dict[str, Any]]:
        if self.client is None:
            return []

        all_items: List[Dict[str, Any]] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._build_qdrant_filter(filters),
                limit=256,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for point in points:
                item = {
                    "id": str((point.payload or {}).get("_record_id", point.id)),
                    "payload": dict(point.payload or {}),
                }
                if include_point_id:
                    item["point_id"] = point.id
                all_items.append(item)
            if offset is None or not points:
                break
        return all_items

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
