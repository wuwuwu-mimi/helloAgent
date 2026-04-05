from __future__ import annotations

from typing import Any, Dict, List, Optional

from memory.base import MemoryConfig, MemoryItem
from memory.embedding import EmbeddingServiceFactory
from memory.storage.document_store import DocumentStore
from memory.storage.neo4j_store import Neo4jGraphStore
from memory.storage.qdrant_store import QdrantVectorStore
from memory.types.episodic import EpisodicMemory
from memory.types.semantic import SemanticMemory
from memory.types.working import WorkingMemory


class MemoryManager:
    """统一协调工作记忆与情景记忆。"""

    _PREFERENCE_MARKERS = (
        "喜欢",
        "不喜欢",
        "偏好",
        "习惯",
        "爱喝",
        "爱吃",
        "讨厌",
        "prefer",
        "like",
        "dislike",
        "favorite",
    )
    _FACT_MARKERS = (
        "支持",
        "包含",
        "项目",
        "系统",
        "能力",
        "功能",
        "helloagent",
        "react",
        "plan-and-solve",
        "reflection",
        "rag",
        "记忆系统",
    )

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        self.config = config or MemoryConfig()
        self.embedding_service = EmbeddingServiceFactory.create(self.config)
        self.working_memory = WorkingMemory(self.config)
        self.episodic_memory = EpisodicMemory(DocumentStore(self.config.memory_db_path))
        self.semantic_memory = SemanticMemory(
            config=self.config,
            store=QdrantVectorStore(
                config=self.config,
                embedding_service=self.embedding_service,
                collection_name=self.config.qdrant_semantic_collection,
                store_path=self.config.semantic_store_path,
            ),
            graph_store=Neo4jGraphStore(self.config),
            embedding_service=self.embedding_service,
        )

    def record_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        persist: bool = True,
    ) -> MemoryItem:
        """把一条消息写入记忆系统。"""
        item = MemoryItem(
            session_id=session_id,
            role=role,
            content=content,
            memory_type="working" if not persist else "episodic",
            metadata=metadata or {},
            expires_at=self.config.working_expires_at(),
        )
        self.working_memory.add(item.model_copy(update={"memory_type": "working"}))
        if persist:
            self.episodic_memory.add(item.model_copy(update={"memory_type": "episodic", "expires_at": None}))
            if self.config.enable_semantic_memory:
                # 修改说明：长期可复用的消息除了写入情景记忆，也同步写入向量记忆，
                # 这样后续查询就不仅能靠关键词匹配，还能走最小语义召回。
                self.semantic_memory.add(
                    item.model_copy(update={"memory_type": "semantic", "expires_at": None})
                )
        return item

    def recall(
        self,
        *,
        session_id: str,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        exclude_text: Optional[str] = None,
    ) -> List[MemoryItem]:
        """组合工作记忆和情景记忆，返回与当前问题相关的上下文。"""
        resolved_limit = limit or self.config.recall_top_k
        candidates: List[MemoryItem] = []

        if query and query.strip():
            candidates.extend(self.working_memory.search(session_id, query, resolved_limit))
            candidates.extend(self.episodic_memory.search(session_id, query, resolved_limit))
            if self.config.enable_semantic_memory:
                semantic_limit = min(resolved_limit, self.config.semantic_recall_top_k)
                candidates.extend(self.semantic_memory.search(session_id, query, semantic_limit))
        else:
            candidates.extend(self.working_memory.recent(session_id, resolved_limit))
            candidates.extend(self.episodic_memory.recent(session_id, resolved_limit))

        if not candidates:
            return []

        deduped = self._dedupe_items(
            candidates,
            limit=resolved_limit,
            exclude_text=exclude_text,
        )
        if deduped:
            return deduped

        if query and query.strip():
            # 修改说明：搜索结果如果只命中了“当前输入自身”，过滤后会变空；
            # 这时再回退到最近记忆，避免 Agent 看不到上一轮真正有用的上下文。
            fallback_candidates: List[MemoryItem] = []
            fallback_candidates.extend(self.working_memory.recent(session_id, resolved_limit))
            fallback_candidates.extend(self.episodic_memory.recent(session_id, resolved_limit))
            if self.config.enable_semantic_memory:
                semantic_limit = min(resolved_limit, self.config.semantic_recall_top_k)
                fallback_candidates.extend(self.semantic_memory.recent(session_id, semantic_limit))
            return self._dedupe_items(
                fallback_candidates,
                limit=resolved_limit,
                exclude_text=exclude_text,
            )

        return []

    def build_memory_prompt(
        self,
        *,
        session_id: str,
        query: Optional[str] = None,
        exclude_text: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> str:
        """把召回的记忆渲染成可直接注入 Prompt 的文本。"""
        items = self.recall(
            session_id=session_id,
            query=query,
            limit=limit or self.config.max_prompt_memories,
            exclude_text=exclude_text,
        )
        if not items:
            return ""

        lines = [
            "以下是与当前任务相关的历史记忆，请仅在有帮助时参考，不要擅自篡改其中的事实："
        ]
        for item in items:
            lines.append(f"- [{item.role}] {item.content}")
        return "\n".join(lines)

    def build_structured_memory_sections(
        self,
        *,
        session_id: str,
        query: Optional[str] = None,
        exclude_text: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, List[str]]:
        """
        把召回到的记忆整理成更适合上下文工程使用的分组结构。

        修改说明：相比单纯把历史记忆直接平铺出来，
        这里会先粗分成“用户偏好 / 项目事实 / 近期对话”，让模型更容易抓住重点。
        """
        items = self.recall(
            session_id=session_id,
            query=query,
            limit=limit or self.config.max_prompt_memories,
            exclude_text=exclude_text,
        )
        grouped: Dict[str, List[str]] = {
            "用户偏好": [],
            "项目事实": [],
            "近期对话": [],
        }
        for item in items:
            bucket = self._classify_memory_item(item)
            grouped[bucket].append(f"[{item.role}] {item.content}")
        return {title: lines for title, lines in grouped.items() if lines}

    def build_structured_memory_prompt(
        self,
        *,
        session_id: str,
        query: Optional[str] = None,
        exclude_text: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> str:
        """把结构化记忆分组渲染成单段文本，便于工具或调试直接展示。"""
        sections = self.build_structured_memory_sections(
            session_id=session_id,
            query=query,
            exclude_text=exclude_text,
            limit=limit,
        )
        if not sections:
            return ""
        blocks: List[str] = []
        for title, lines in sections.items():
            blocks.append(f"{title}：\n" + "\n".join(f"- {line}" for line in lines))
        return "\n\n".join(blocks)

    def clear_session(self, session_id: str) -> None:
        """清空某个 session 的全部记忆。"""
        self.working_memory.clear(session_id)
        self.episodic_memory.clear(session_id)
        self.semantic_memory.clear(session_id)

    @staticmethod
    def _dedupe_items(
        items: List[MemoryItem],
        *,
        limit: int,
        exclude_text: Optional[str] = None,
    ) -> List[MemoryItem]:
        """对召回结果去重、过滤并截断，避免工作记忆和长期记忆重复注入。"""
        deduped: List[MemoryItem] = []
        seen: set[tuple[str, str, str]] = set()
        for item in sorted(items, key=lambda current: current.created_at):
            if exclude_text and item.content == exclude_text:
                continue
            key = (item.role, item.content, item.created_at.isoformat())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[-limit:]

    @classmethod
    def _classify_memory_item(cls, item: MemoryItem) -> str:
        """根据内容做一个轻量分组，便于后续上下文路由。"""
        content = item.content.strip().lower()
        if any(marker in content for marker in cls._PREFERENCE_MARKERS):
            return "用户偏好"
        if any(marker in content for marker in cls._FACT_MARKERS):
            return "项目事实"
        return "近期对话"
