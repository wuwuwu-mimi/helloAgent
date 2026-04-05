from __future__ import annotations

from collections import defaultdict
from datetime import datetime
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
    _LOW_VALUE_PATTERNS = (
        "好的",
        "收到",
        "明白",
        "知道了",
        "ok",
        "okay",
        "thanks",
        "thank you",
        "嗯",
        "哦",
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
        self._decision_log: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._retention_log: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

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
        cleaned_content = content.strip()
        decision = self._plan_memory_record(
            session_id=session_id,
            role=role,
            content=cleaned_content,
            metadata=metadata or {},
            persist=persist,
        )
        enriched_metadata = dict(metadata or {})
        enriched_metadata.update(
            {
                "memory_value": decision["value_label"],
                "memory_reasons": list(decision["reasons"]),
                "memory_plan": {
                    "working": decision["store_working"],
                    "episodic": decision["store_episodic"],
                    "semantic": decision["store_semantic"],
                    "skipped": decision["skipped"],
                },
            }
        )
        item = MemoryItem(
            session_id=session_id,
            role=role,
            content=cleaned_content,
            memory_type="working" if not persist else "episodic",
            metadata=enriched_metadata,
            expires_at=self.config.working_expires_at(),
        )
        self._record_memory_decision(
            session_id=session_id,
            role=role,
            content=cleaned_content,
            decision=decision,
        )
        if decision["skipped"]:
            return item.model_copy(update={"memory_type": "skipped", "expires_at": None})

        if decision["store_working"]:
            self.working_memory.add(item.model_copy(update={"memory_type": "working"}))
        if decision["store_episodic"]:
            self.episodic_memory.add(item.model_copy(update={"memory_type": "episodic", "expires_at": None}))
        if decision["store_semantic"]:
            # 修改说明：长期可复用且有较高价值的消息才进入语义记忆，
            # 避免把“收到/确认”这类低价值文本也推到向量库里。
            self.semantic_memory.add(
                item.model_copy(update={"memory_type": "semantic", "expires_at": None})
            )
        self._apply_retention(session_id)
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
        normalized_query = (query or "").strip()

        if normalized_query:
            candidates.extend(
                self._annotate_recall_source(
                    self.working_memory.search(session_id, normalized_query, resolved_limit),
                    source="working_search",
                    query=normalized_query,
                )
            )
            candidates.extend(
                self._annotate_recall_source(
                    self.episodic_memory.search(session_id, normalized_query, resolved_limit),
                    source="episodic_search",
                    query=normalized_query,
                )
            )
            if self.config.enable_semantic_memory:
                semantic_limit = min(resolved_limit, self.config.semantic_recall_top_k)
                candidates.extend(
                    self._annotate_recall_source(
                        self.semantic_memory.search(session_id, normalized_query, semantic_limit),
                        source="semantic_search",
                        query=normalized_query,
                    )
                )
        else:
            candidates.extend(
                self._annotate_recall_source(
                    self.working_memory.recent(session_id, resolved_limit),
                    source="working_recent",
                    query=None,
                )
            )
            candidates.extend(
                self._annotate_recall_source(
                    self.episodic_memory.recent(session_id, resolved_limit),
                    source="episodic_recent",
                    query=None,
                )
            )

        if not candidates:
            return []

        deduped = self._dedupe_items(
            candidates,
            limit=resolved_limit,
            exclude_text=exclude_text,
        )
        if deduped:
            return deduped

        if normalized_query:
            # 修改说明：搜索结果如果只命中了“当前输入自身”，过滤后会变空；
            # 这时再回退到最近记忆，避免 Agent 看不到上一轮真正有用的上下文。
            fallback_candidates: List[MemoryItem] = []
            fallback_candidates.extend(
                self._annotate_recall_source(
                    self.working_memory.recent(session_id, resolved_limit),
                    source="working_recent_fallback",
                    query=normalized_query,
                )
            )
            fallback_candidates.extend(
                self._annotate_recall_source(
                    self.episodic_memory.recent(session_id, resolved_limit),
                    source="episodic_recent_fallback",
                    query=normalized_query,
                )
            )
            if self.config.enable_semantic_memory:
                semantic_limit = min(resolved_limit, self.config.semantic_recall_top_k)
                fallback_candidates.extend(
                    self._annotate_recall_source(
                        self.semantic_memory.recent(session_id, semantic_limit),
                        source="semantic_recent_fallback",
                        query=normalized_query,
                    )
                )
            return self._dedupe_items(
                fallback_candidates,
                limit=resolved_limit,
                exclude_text=exclude_text,
            )

        return []

    def build_recall_diagnostics(
        self,
        *,
        session_id: str,
        query: Optional[str] = None,
        exclude_text: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> str:
        """输出召回结果的来源和原因，便于解释为什么这些记忆被拿出来。"""
        items = self.recall(
            session_id=session_id,
            query=query,
            limit=limit or self.config.recall_top_k,
            exclude_text=exclude_text,
        )
        if not items:
            return ""

        lines = ["最近一次记忆召回解释："]
        for index, item in enumerate(items, start=1):
            recall_sources = item.metadata.get("recall_sources", [])
            recall_reason = str(item.metadata.get("recall_reason", "") or "").strip()
            memory_value = str(item.metadata.get("memory_value", "") or "").strip()
            lines.append(
                f"{index}. [{item.role}] sources={','.join(recall_sources) or item.memory_type} value={memory_value or '-'}"
            )
            if recall_reason:
                lines.append(f"   reason={recall_reason}")
            lines.append(f"   content={self._trim_summary_line(item.content, limit=88)}")
        return "\n".join(lines)

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
            explanation = self._render_recall_explanation(item)
            suffix = f" ({explanation})" if explanation else ""
            lines.append(f"- [{item.role}] {item.content}{suffix}")
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
            explanation = self._render_recall_explanation(item)
            suffix = f" ({explanation})" if explanation else ""
            grouped[bucket].append(f"[{item.role}] {item.content}{suffix}")
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

    def build_session_summary(
        self,
        *,
        session_id: str,
        query: Optional[str] = None,
        exclude_text: Optional[str] = None,
    ) -> str:
        """
        生成一个轻量的会话摘要。

        修改说明：长对话场景下，如果每次都把原始记忆条目整段塞进 prompt，
        上下文很快会膨胀；这里先用规则式方法提炼“偏好 / 事实 / 最近进展”的摘要层。
        """
        if not self.config.enable_session_summary:
            return ""

        recent_items = self._dedupe_items(
            [
                *self.working_memory.recent(session_id, self.config.session_summary_recent_items),
                *self.episodic_memory.recent(session_id, self.config.session_summary_recent_items),
            ],
            limit=self.config.session_summary_recent_items,
            exclude_text=exclude_text,
        )
        if len(recent_items) < self.config.session_summary_min_messages:
            return ""

        grouped_recent: Dict[str, List[MemoryItem]] = {
            "用户偏好": [],
            "项目事实": [],
            "近期对话": [],
        }
        for item in recent_items:
            grouped_recent[self._classify_memory_item(item)].append(item)

        summary_lines: List[str] = []

        preferences = grouped_recent.get("用户偏好", [])[:2]
        if preferences:
            summary_lines.append("用户偏好:")
            summary_lines.extend(
                f"- [{item.role}] {self._trim_summary_line(item.content)}"
                for item in preferences
            )

        facts = grouped_recent.get("项目事实", [])[:2]
        if facts:
            summary_lines.append("项目事实:")
            summary_lines.extend(
                f"- [{item.role}] {self._trim_summary_line(item.content)}"
                for item in facts
            )

        recent_dialogue = grouped_recent.get("近期对话", [])[-2:]
        if recent_dialogue:
            summary_lines.append("最近进展:")
            summary_lines.extend(
                f"- [{item.role}] {self._trim_summary_line(item.content)}"
                for item in recent_dialogue
            )

        if not summary_lines:
            return ""

        capped_lines = summary_lines[: self.config.session_summary_max_lines]
        return "\n".join(capped_lines)

    def clear_session(self, session_id: str) -> None:
        """清空某个 session 的全部记忆。"""
        self.working_memory.clear(session_id)
        self.episodic_memory.clear(session_id)
        self.semantic_memory.clear(session_id)
        self._decision_log.pop(session_id, None)
        self._retention_log.pop(session_id, None)

    def build_memory_diagnostics(self, session_id: str, limit: int = 10) -> str:
        """输出最近几条记忆写入决策，便于观察闭环策略是否生效。"""
        decisions = self._decision_log.get(session_id, [])
        if not decisions:
            return ""
        lines = ["最近的记忆写入决策："]
        for index, item in enumerate(decisions[-limit:], start=1):
            plan = item["plan"]
            lines.append(
                (
                    f"{index}. [{item['role']}] value={item['value_label']} | "
                    f"working={plan['working']} episodic={plan['episodic']} semantic={plan['semantic']} "
                    f"skipped={plan['skipped']}"
                )
            )
            lines.append(f"   reasons={', '.join(item['reasons'])}")
            lines.append(f"   content={self._trim_summary_line(item['content'], limit=88)}")
        return "\n".join(lines)

    def build_retention_diagnostics(self, session_id: str, limit: int = 5) -> str:
        """输出最近几次长期保留裁剪结果，便于观察哪些记忆被淘汰。"""
        records = self._retention_log.get(session_id, [])
        if not records:
            return ""

        lines = ["最近的长期保留裁剪："]
        for index, item in enumerate(records[-limit:], start=1):
            lines.append(
                (
                    f"{index}. store={item['store_name']} before={item['before_count']} "
                    f"kept={item['kept_count']} pruned={item['pruned_count']} "
                    f"limit={item['limit']}"
                )
            )
            if item["dropped"]:
                lines.append(f"   dropped={', '.join(item['dropped'])}")
            else:
                lines.append("   dropped=(无)")
        return "\n".join(lines)

    @staticmethod
    def _dedupe_items(
        items: List[MemoryItem],
        *,
        limit: int,
        exclude_text: Optional[str] = None,
    ) -> List[MemoryItem]:
        """对召回结果去重、过滤并截断，避免工作记忆和长期记忆重复注入。"""
        merged: Dict[tuple[str, str], MemoryItem] = {}
        for item in sorted(items, key=lambda current: current.created_at):
            if exclude_text and item.content == exclude_text:
                continue
            key = (item.role, MemoryManager._normalize_memory_text(item.content))
            if key not in merged:
                merged[key] = item
                continue
            existing = merged[key]
            chosen = item if item.created_at >= existing.created_at else existing
            sources = sorted(
                {
                    existing.memory_type,
                    item.memory_type,
                    *existing.metadata.get("recall_sources", []),
                    *item.metadata.get("recall_sources", []),
                }
            )
            chosen = chosen.model_copy(
                update={
                    "metadata": {
                        **chosen.metadata,
                        "recall_sources": sources,
                    }
                }
            )
            merged[key] = chosen
        deduped = sorted(merged.values(), key=lambda current: current.created_at)
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

    @staticmethod
    def _trim_summary_line(text: str, limit: int = 72) -> str:
        """裁剪摘要里的单行文本，避免单条摘要过长。"""
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return f"{compact[:limit].rstrip()}..."

    def _annotate_recall_source(
        self,
        items: List[MemoryItem],
        *,
        source: str,
        query: Optional[str],
    ) -> List[MemoryItem]:
        """给召回项补充来源和原因解释。"""
        annotated: List[MemoryItem] = []
        for item in items:
            metadata = dict(item.metadata)
            existing_sources = list(metadata.get("recall_sources", []))
            metadata["recall_sources"] = self._merge_recall_sources(existing_sources, [source])
            metadata["recall_reason"] = self._build_recall_reason(item, source=source, query=query)
            annotated.append(item.model_copy(update={"metadata": metadata}))
        return annotated

    @staticmethod
    def _merge_recall_sources(existing: List[str], extra: List[str]) -> List[str]:
        merged: List[str] = []
        for source in [*existing, *extra]:
            if source and source not in merged:
                merged.append(source)
        return merged

    def _build_recall_reason(self, item: MemoryItem, *, source: str, query: Optional[str]) -> str:
        """根据来源和 metadata 拼出一条简洁的召回解释。"""
        reason_parts: List[str] = []
        if source.startswith("semantic"):
            score = item.metadata.get("semantic_score")
            if score is not None:
                reason_parts.append(f"semantic_score={score}")
        if source.startswith("working"):
            reason_parts.append("来自近期工作记忆")
        elif source.startswith("episodic"):
            reason_parts.append("来自持久化情景记忆")
        elif source.startswith("semantic"):
            reason_parts.append("来自语义/图谱召回")
        if "fallback" in source:
            reason_parts.append("搜索不足后回退到 recent")
        if query:
            reason_parts.append(f"query={self._trim_summary_line(query, limit=24)}")
        return "; ".join(reason_parts)

    def _render_recall_explanation(self, item: MemoryItem) -> str:
        """把召回解释压成适合直接显示在 prompt/调试文本里的短说明。"""
        sources = list(item.metadata.get("recall_sources", []))
        score = item.metadata.get("semantic_score")
        parts: List[str] = []
        if sources:
            parts.append(f"source={'+'.join(sources)}")
        if score is not None:
            parts.append(f"score={score}")
        return "; ".join(parts)

    def _plan_memory_record(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any],
        persist: bool,
    ) -> Dict[str, Any]:
        """为一条消息选择记忆写入策略。"""
        reasons: List[str] = []
        if not content:
            return self._build_memory_plan(
                skipped=True,
                value_label="low",
                reasons=["empty_content"],
            )

        content_kind = self._classify_content_kind(content, role=role, metadata=metadata)
        value_label = self._score_memory_value(content, role=role, metadata=metadata, content_kind=content_kind)
        reasons.append(f"kind={content_kind}")
        reasons.append(f"value={value_label}")

        if self.config.enable_memory_value_filter and self._is_low_value_message(
            content,
            role=role,
            metadata=metadata,
            content_kind=content_kind,
        ):
            reasons.append("filtered_low_value")
            return self._build_memory_plan(skipped=True, value_label=value_label, reasons=reasons)

        if self._has_recent_duplicate(session_id, role=role, content=content):
            reasons.append("duplicate_recent")
            return self._build_memory_plan(skipped=True, value_label=value_label, reasons=reasons)

        store_working = True
        store_episodic = persist
        store_semantic = (
            persist
            and self.config.enable_semantic_memory
            and len(content) >= self.config.semantic_min_content_length
            and self._should_store_in_semantic(role=role, metadata=metadata, content_kind=content_kind)
        )
        if store_semantic:
            reasons.append("semantic_candidate")
        if store_episodic:
            reasons.append("persist_enabled")
        else:
            reasons.append("working_only")

        return self._build_memory_plan(
            store_working=store_working,
            store_episodic=store_episodic,
            store_semantic=store_semantic,
            skipped=False,
            value_label=value_label,
            reasons=reasons,
        )

    @staticmethod
    def _build_memory_plan(
        *,
        store_working: bool = False,
        store_episodic: bool = False,
        store_semantic: bool = False,
        skipped: bool,
        value_label: str,
        reasons: List[str],
    ) -> Dict[str, Any]:
        return {
            "store_working": store_working,
            "store_episodic": store_episodic,
            "store_semantic": store_semantic,
            "skipped": skipped,
            "value_label": value_label,
            "reasons": reasons,
        }

    def _record_memory_decision(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        decision: Dict[str, Any],
    ) -> None:
        """记录一条写入决策，便于后续解释为什么某条消息被保留或忽略。"""
        self._decision_log[session_id].append(
            {
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "content": content,
                "value_label": decision["value_label"],
                "reasons": list(decision["reasons"]),
                "plan": {
                    "working": decision["store_working"],
                    "episodic": decision["store_episodic"],
                    "semantic": decision["store_semantic"],
                    "skipped": decision["skipped"],
                },
            }
        )
        if len(self._decision_log[session_id]) > 40:
            self._decision_log[session_id] = self._decision_log[session_id][-40:]

    def _apply_retention(self, session_id: str) -> None:
        """按价值优先、时间兜底的方式裁剪长期记忆，避免持久化层无限膨胀。"""
        if not self.config.enable_memory_retention:
            return
        self._apply_single_retention(
            session_id=session_id,
            store_name="episodic",
            items=self.episodic_memory.list_all(session_id),
            max_items=max(0, self.config.episodic_retention_max_items),
            prune_callback=self.episodic_memory.prune,
        )
        if self.config.enable_semantic_memory:
            self._apply_single_retention(
                session_id=session_id,
                store_name="semantic",
                items=self.semantic_memory.list_all(session_id),
                max_items=max(0, self.config.semantic_retention_max_items),
                prune_callback=self.semantic_memory.prune,
            )

    def _apply_single_retention(
        self,
        *,
        session_id: str,
        store_name: str,
        items: List[MemoryItem],
        max_items: int,
        prune_callback: Any,
    ) -> None:
        """对单个持久化存储执行裁剪，并把结果记录到 retention log。"""
        if max_items <= 0 or len(items) <= max_items:
            return

        keep_items = self._select_items_to_keep(items, max_items=max_items)
        keep_ids = [item.id for item in keep_items]
        keep_id_set = set(keep_ids)
        dropped_items = [item for item in items if item.id not in keep_id_set]
        pruned_count = int(prune_callback(session_id, keep_ids))
        self._record_retention_result(
            session_id=session_id,
            store_name=store_name,
            before_count=len(items),
            kept_count=len(keep_items),
            pruned_count=pruned_count,
            limit=max_items,
            dropped_items=dropped_items,
        )

    def _select_items_to_keep(self, items: List[MemoryItem], *, max_items: int) -> List[MemoryItem]:
        """
        在“高价值优先保留、同价值时偏向最新消息”的规则下选择保留集。

        修改说明：这里不只是简单保留最近 N 条，
        而是优先保留偏好、事实、工具成功结果、最终答案等更可复用的长期记忆，
        再用时间顺序兜底，减少重要事实被普通闲聊挤掉。
        """
        scored_items = [
            (
                self._score_retention_priority(item, recency_index=index, total=len(items)),
                item.created_at,
                item,
            )
            for index, item in enumerate(items)
        ]
        scored_items.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        chosen = [entry[2] for entry in scored_items[:max_items]]
        return sorted(chosen, key=lambda item: item.created_at)

    def _score_retention_priority(self, item: MemoryItem, *, recency_index: int, total: int) -> int:
        """为长期保留策略计算优先级分数。"""
        metadata = item.metadata or {}
        value_label = str(metadata.get("memory_value", "") or "").lower()
        reasons = list(metadata.get("memory_reasons", []))
        content_kind = ""
        for reason in reasons:
            if isinstance(reason, str) and reason.startswith("kind="):
                content_kind = reason.split("=", 1)[1].strip().lower()
                break

        value_score = {"high": 300, "medium": 200, "low": 100}.get(value_label, 150)
        kind_bonus = {
            "preference": 90,
            "fact": 85,
            "tool_result_success": 80,
            "final_answer": 60,
            "tool_result_failure": 20,
            "user_dialogue": 10,
            "assistant_dialogue": 0,
        }.get(content_kind, 0)
        high_value_bonus = 40 if self.config.retention_keep_high_value and value_label == "high" else 0
        user_bonus = 10 if item.role == "user" else 0
        recency_bonus = recency_index if total > 1 else 1
        return value_score + kind_bonus + high_value_bonus + user_bonus + recency_bonus

    def _record_retention_result(
        self,
        *,
        session_id: str,
        store_name: str,
        before_count: int,
        kept_count: int,
        pruned_count: int,
        limit: int,
        dropped_items: List[MemoryItem],
    ) -> None:
        """记录一次长期保留裁剪结果，便于后续调试与观察。"""
        self._retention_log[session_id].append(
            {
                "timestamp": datetime.now().isoformat(),
                "store_name": store_name,
                "before_count": before_count,
                "kept_count": kept_count,
                "pruned_count": pruned_count,
                "limit": limit,
                "dropped": [
                    self._trim_summary_line(f"[{item.role}] {item.content}", limit=60)
                    for item in dropped_items
                ],
            }
        )
        if len(self._retention_log[session_id]) > 20:
            self._retention_log[session_id] = self._retention_log[session_id][-20:]

    def _has_recent_duplicate(self, session_id: str, *, role: str, content: str) -> bool:
        """检查最近窗口内是否已有相同角色和内容的记忆。"""
        normalized = self._normalize_memory_text(content)
        if not normalized:
            return False
        window = max(1, self.config.memory_duplicate_window)
        recent_items = [
            *self.working_memory.recent(session_id, window),
            *self.episodic_memory.recent(session_id, window),
        ]
        for item in recent_items:
            if item.role != role:
                continue
            if self._normalize_memory_text(item.content) == normalized:
                return True
        return False

    def _classify_content_kind(
        self,
        content: str,
        *,
        role: str,
        metadata: Dict[str, Any],
    ) -> str:
        normalized = content.strip().lower()
        if metadata.get("source") == "tool_result":
            return "tool_result_success" if metadata.get("tool_success", True) else "tool_result_failure"
        if any(marker in normalized for marker in self._PREFERENCE_MARKERS):
            return "preference"
        if any(marker in normalized for marker in self._FACT_MARKERS):
            return "fact"
        memory_stage = str(metadata.get("memory_stage", "") or "").lower()
        if memory_stage.endswith("finish") or memory_stage.endswith("final"):
            return "final_answer"
        if role == "user":
            return "user_dialogue"
        if role == "assistant":
            return "assistant_dialogue"
        return "other"

    def _score_memory_value(
        self,
        content: str,
        *,
        role: str,
        metadata: Dict[str, Any],
        content_kind: str,
    ) -> str:
        """给消息打一个粗粒度价值标签。"""
        del role
        if content_kind in {"preference", "fact", "tool_result_success"}:
            return "high"
        if content_kind == "final_answer":
            return "high" if len(content) >= self.config.semantic_min_content_length else "medium"
        if metadata.get("source") == "tool_result" and not metadata.get("tool_success", True):
            return "medium"
        if len(content) >= max(self.config.memory_min_content_length * 3, 12):
            return "medium"
        return "low"

    def _is_low_value_message(
        self,
        content: str,
        *,
        role: str,
        metadata: Dict[str, Any],
        content_kind: str,
    ) -> bool:
        """过滤明显噪声消息，避免把无价值文本塞进长期记忆。"""
        normalized = content.strip().lower()
        simplified = normalized.strip("。！？!?.,，、；;：:~ ")
        if not normalized:
            return True
        if content_kind in {"preference", "fact", "tool_result_success"}:
            return False
        if content_kind == "final_answer" and simplified in self._LOW_VALUE_PATTERNS:
            return True
        if content_kind == "final_answer":
            return False
        if metadata.get("source") == "tool_result":
            return False
        if len(normalized) < self.config.memory_min_content_length:
            return True
        if simplified in self._LOW_VALUE_PATTERNS:
            return True
        if role == "assistant" and len(normalized) <= 12 and any(
            simplified == pattern for pattern in self._LOW_VALUE_PATTERNS
        ):
            return True
        return False

    def _should_store_in_semantic(
        self,
        *,
        role: str,
        metadata: Dict[str, Any],
        content_kind: str,
    ) -> bool:
        """判断一条持久化消息是否值得进入语义记忆。"""
        if content_kind in {"preference", "fact", "tool_result_success", "final_answer"}:
            return True
        if role == "user" and content_kind == "user_dialogue":
            return False
        if metadata.get("source") == "tool_result":
            return bool(metadata.get("tool_success", False))
        return False

    @staticmethod
    def _normalize_memory_text(text: str) -> str:
        """归一化记忆文本，便于做重复检测。"""
        return " ".join(text.strip().lower().split())
