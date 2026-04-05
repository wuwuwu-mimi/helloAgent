from __future__ import annotations

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
    semantic_store_path: str = "data/qdrant_memory.json"
    working_memory_max_items: int = 20
    working_memory_ttl_seconds: int = 60 * 30
    recall_top_k: int = 6
    max_prompt_memories: int = 5
    enable_semantic_memory: bool = True
    semantic_recall_top_k: int = 4
    semantic_score_threshold: float = 0.12
    embedding_backend: str = "hash"
    embedding_dimensions: int = 96
    persist_user_messages: bool = True
    persist_assistant_messages: bool = True
    persist_tool_messages: bool = False

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
