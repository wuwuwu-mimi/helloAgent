from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    """统一描述一条记忆数据。"""

    id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str = "default"
    role: str = "assistant"
    content: str
    memory_type: str = "working"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """判断当前记忆是否已经过期。"""
        if self.expires_at is None:
            return False
        return (now or datetime.now()) >= self.expires_at


class MemoryConfig(BaseModel):
    """记忆系统的统一配置。"""

    enabled: bool = True
    memory_db_path: str = "data/memory.db"
    vector_store_backend: str = "auto"
    semantic_store_path: str = "data/qdrant_memory.json"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_local_path: str = ""
    qdrant_semantic_collection: str = "semantic_memory"
    qdrant_rag_collection: str = "rag_chunks"
    working_memory_max_items: int = 20
    working_memory_ttl_seconds: int = 60 * 30
    recall_top_k: int = 6
    max_prompt_memories: int = 5
    enable_semantic_memory: bool = True
    semantic_recall_top_k: int = 4
    semantic_score_threshold: float = 0.12
    embedding_backend: str = "hash"
    embedding_dimensions: int = 96
    rag_store_path: str = "data/rag_index.json"
    rag_chunk_size: int = 300
    rag_chunk_overlap: int = 60
    rag_top_k: int = 3
    persist_user_messages: bool = True
    persist_assistant_messages: bool = True
    persist_tool_messages: bool = False

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """从环境变量构造记忆系统配置。"""
        defaults = cls()
        return cls(
            enabled=_read_bool(defaults.enabled, "MEMORY_ENABLED"),
            memory_db_path=_first_env("MEMORY_DB_PATH") or defaults.memory_db_path,
            vector_store_backend=_first_env("VECTOR_STORE_BACKEND") or defaults.vector_store_backend,
            semantic_store_path=_first_env("SEMANTIC_STORE_PATH") or defaults.semantic_store_path,
            qdrant_url=_first_env("QDRANT_URL") or defaults.qdrant_url,
            qdrant_api_key=_first_env("QDRANT_API_KEY") or defaults.qdrant_api_key,
            qdrant_local_path=_first_env("QDRANT_LOCAL_PATH") or defaults.qdrant_local_path,
            qdrant_semantic_collection=_first_env("QDRANT_SEMANTIC_COLLECTION")
            or defaults.qdrant_semantic_collection,
            qdrant_rag_collection=_first_env("QDRANT_RAG_COLLECTION")
            or defaults.qdrant_rag_collection,
            working_memory_max_items=_read_int(
                defaults.working_memory_max_items,
                "WORKING_MEMORY_MAX_ITEMS",
            ),
            working_memory_ttl_seconds=_read_int(
                defaults.working_memory_ttl_seconds,
                "WORKING_MEMORY_TTL_SECONDS",
            ),
            recall_top_k=_read_int(defaults.recall_top_k, "MEMORY_RECALL_TOP_K"),
            max_prompt_memories=_read_int(
                defaults.max_prompt_memories,
                "MAX_PROMPT_MEMORIES",
            ),
            enable_semantic_memory=_read_bool(
                defaults.enable_semantic_memory,
                "ENABLE_SEMANTIC_MEMORY",
            ),
            semantic_recall_top_k=_read_int(
                defaults.semantic_recall_top_k,
                "SEMANTIC_RECALL_TOP_K",
            ),
            semantic_score_threshold=_read_float(
                defaults.semantic_score_threshold,
                "SEMANTIC_SCORE_THRESHOLD",
            ),
            embedding_backend=_first_env("EMBEDDING_BACKEND") or defaults.embedding_backend,
            embedding_dimensions=_read_int(
                defaults.embedding_dimensions,
                "EMBEDDING_DIMENSIONS",
            ),
            rag_store_path=_first_env("RAG_STORE_PATH") or defaults.rag_store_path,
            rag_chunk_size=_read_int(defaults.rag_chunk_size, "RAG_CHUNK_SIZE"),
            rag_chunk_overlap=_read_int(defaults.rag_chunk_overlap, "RAG_CHUNK_OVERLAP"),
            rag_top_k=_read_int(defaults.rag_top_k, "RAG_TOP_K"),
            persist_user_messages=_read_bool(
                defaults.persist_user_messages,
                "PERSIST_USER_MESSAGES",
            ),
            persist_assistant_messages=_read_bool(
                defaults.persist_assistant_messages,
                "PERSIST_ASSISTANT_MESSAGES",
            ),
            persist_tool_messages=_read_bool(
                defaults.persist_tool_messages,
                "PERSIST_TOOL_MESSAGES",
            ),
        )

    def working_expires_at(self) -> datetime:
        """生成工作记忆默认过期时间。"""
        return datetime.now() + timedelta(seconds=self.working_memory_ttl_seconds)


class BaseMemory(ABC):
    """各种记忆类型的统一抽象接口。"""

    @abstractmethod
    def add(self, item: MemoryItem) -> None:
        """写入一条记忆。"""

    @abstractmethod
    def recent(self, session_id: str, limit: int = 10) -> List[MemoryItem]:
        """获取最近的若干条记忆。"""

    @abstractmethod
    def search(self, session_id: str, query: str, limit: int = 10) -> List[MemoryItem]:
        """按 query 检索相关记忆。"""

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """清空指定 session 的记忆。"""


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return ""


def _read_bool(default: bool, *names: str) -> bool:
    raw = _first_env(*names)
    if not raw:
        return default
    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"无法将环境变量解析为 bool: {raw}")


def _read_int(default: int, *names: str) -> int:
    raw = _first_env(*names)
    return int(raw) if raw else default


def _read_float(default: float, *names: str) -> float:
    raw = _first_env(*names)
    return float(raw) if raw else default
