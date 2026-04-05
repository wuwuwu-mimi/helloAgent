from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from agents.agent_base import Agent
from core import ChatResult, Config, ContextBuilder, HelloAgentsLLM, Message, ToolCall, ToolFunction
from memory.manager import MemoryManager
from tools.builtin.tool_base import ToolResult
from tools.builtin.toolRegistry import ToolRegistry


class ReasoningAgentBase(Agent):
    """带有统一消息构建、LLM 调用和运行期历史管理能力的公共父类。"""

    _PREFERENCE_PATTERN = re.compile(
        r"(?:用户|我|这个用户)?(?P<neg>不|别|别再|并不|不是很)?(?P<verb>喜欢|爱喝|偏好|讨厌)"
        r"(?P<object>[\u4e00-\u9fffA-Za-z0-9\-]{2,20})"
    )
    _SUPPORT_PATTERN = re.compile(
        r"(?P<subject>helloagent|项目|系统|记忆系统)?(?:当前|目前)?(?P<neg>不)?支持"
        r"(?P<object>[\u4e00-\u9fffA-Za-z0-9\-]{2,30})",
        re.IGNORECASE,
    )
    _CONTAINS_PATTERN = re.compile(
        r"(?P<subject>记忆系统|系统|项目)?(?:已经)?(?P<neg>不)?包含"
        r"(?P<object>[\u4e00-\u9fffA-Za-z0-9\-]{2,30})"
    )

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        prompt_template: str,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        memory_manager: Optional[MemoryManager] = None,
        session_id: Optional[str] = None,
    ) -> None:
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.prompt_template = prompt_template
        self.current_history: List[str] = []
        self.memory_manager = memory_manager
        self.session_id = session_id or name
        self.current_input: str = ""
        self.tool_observations: List[Dict[str, Any]] = []
        self._rag_context_cache: Dict[str, str] = {}
        self._rag_evidence_cache: Dict[str, List[Dict[str, str]]] = {}
        self.enable_native_tool_calling = False
        self._native_tool_execution_in_progress = False
        self._last_tool_result_snapshot: Optional[Dict[str, Any]] = None

    def _start_new_run(self, input_text: str) -> None:
        """
        为一次新的任务重置运行态。

        修改说明：把“清空历史 + 注入 system/user 消息”的样板逻辑抽到公共父类，
        这样后续新增 Agent 范式时不用再重复写一遍。
        """
        self.current_history = []
        self.current_input = input_text
        self.tool_observations = []
        self._rag_context_cache = {}
        self._rag_evidence_cache = {}
        self.clear_history()

        if self.system_prompt:
            self.add_message(Message.system(self.system_prompt))
        self._remember_message(Message.user(input_text))

    def _build_messages(self, prompt: str) -> List[Message]:
        """把本轮 prompt 包装成发给 LLM 的消息列表。"""
        messages: List[Message] = []
        context_packet = self._build_context_packet()
        rendered_context = context_packet.render(
            max_chars=self.config.context_max_chars,
            max_sections=self.config.context_max_sections,
            section_max_chars=self.config.context_section_max_chars,
        )
        if rendered_context:
            # 修改说明：把 system prompt、记忆和运行规则都收敛到结构化上下文里，
            # 这样后续继续做上下文裁剪、优先级合并时不需要重写消息拼装逻辑。
            messages.append(Message.system(rendered_context, metadata={"source": "context_engineering"}))
        messages.append(Message.user(prompt))
        return messages

    def _request_text(self, prompt: str, **kwargs: Any) -> str:
        """
        统一调用 LLM，并返回清理后的文本结果。

        修改说明：把重复出现的 `llm.chat(...).text.strip()` 抽成公共方法，
        让子类把注意力放在“如何组织流程”而不是“如何发请求”上。
        """
        result = self._request_result(prompt, **kwargs)
        return (result.text or "").strip()

    def _request_result(self, prompt: str, **kwargs: Any) -> ChatResult:
        """
        统一调用 LLM，并保留完整结果对象。

        修改说明：后面做原生 tool calling 时，除了文本，还需要拿到 `tool_calls`；
        所以这里在父类加一个完整结果入口，文本版和 schema 版都可以复用。
        """
        return self.llm.chat(
            messages=self._build_messages(prompt),
            stream=False,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs,
        )

    def _request_result_with_messages(self, messages: List[Message], **kwargs: Any) -> ChatResult:
        """
        使用已经拼装好的消息列表调用 LLM。

        修改说明：原生 tool calling 需要在一轮里不断把 assistant/tool 消息回填到同一段对话中，
        因此补一个“直接吃消息列表”的公共入口，避免各个 Agent 自己重复写 `llm.chat(...)`。
        """
        return self.llm.chat(
            messages=messages,
            stream=False,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs,
        )

    def _render_context_packet(self) -> str:
        """把结构化上下文渲染成最终 system prompt 文本。"""
        context_packet = self._build_context_packet()
        return context_packet.render(
            max_chars=self.config.context_max_chars,
            max_sections=self.config.context_max_sections,
            section_max_chars=self.config.context_section_max_chars,
        )

    def _build_context_system_message(
        self,
        rendered_context: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """把渲染后的结构化上下文包装成 system 消息。"""
        payload = {"source": "context_engineering"}
        if metadata:
            payload.update(metadata)
        return Message.system(rendered_context, metadata=payload)

    @staticmethod
    def _build_native_user_message(prompt: str, *, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """构造原生 tool calling 循环里的 user 消息。"""
        return Message.user(prompt, metadata=metadata or {})

    def _build_native_tool_message(
        self,
        observation: str,
        *,
        tool_call_id: str,
        tool_name: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """把工具执行结果包装成标准 tool message。"""
        payload = {"native_tool_calling": True}
        if metadata:
            payload.update(metadata)
        return Message.tool(
            observation,
            tool_call_id=tool_call_id,
            name=tool_name,
            metadata=payload,
        )

    def _build_native_tool_calling_messages(
        self,
        prompt: str,
        *,
        user_metadata: Optional[Dict[str, Any]] = None,
        context_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Message]:
        """构造原生 tool calling 起始消息。"""
        rendered_context = self._render_context_packet()
        messages: List[Message] = []
        if rendered_context:
            messages.append(
                self._build_context_system_message(
                    rendered_context,
                    metadata=context_metadata,
                )
            )
        messages.append(self._build_native_user_message(prompt, metadata=user_metadata))
        return messages

    def _append_history_entry(
        self,
        entry: str,
        *,
        history_target: Optional[List[str]] = None,
        append_current_history: bool = True,
    ) -> None:
        """按需把调试历史写入当前 Agent 历史或局部历史。"""
        if append_current_history:
            self.current_history.append(entry)
        if history_target is not None:
            history_target.append(entry)

    def _should_use_native_tool_calling(self) -> bool:
        """判断当前是否启用原生 tool calling。"""
        mode = (self.config.tool_calling_mode or "text").strip().lower()
        if mode not in {"native", "auto"}:
            return False
        return self.enable_native_tool_calling and bool(self.tool_registry.list_tools())

    def _build_assistant_message_from_result(
        self,
        result: ChatResult,
        *,
        round_index: int,
        history_target: Optional[List[str]] = None,
        append_current_history: bool = True,
        history_label_prefix: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """把 LLM 返回的文本与 tool_calls 统一转成 assistant 消息。"""
        tool_calls = [
            ToolCall(
                id=item.get("id"),
                type=item.get("type", "function"),
                function=ToolFunction(
                    name=item.get("function", {}).get("name", ""),
                    arguments=item.get("function", {}).get("arguments", "") or "",
                ),
            )
            for item in (result.tool_calls or [])
        ]
        preview_parts: List[str] = []
        if result.text:
            preview_parts.append(f"{history_label_prefix}Assistant: {(result.text or '').strip()}")
        for item in tool_calls:
            preview_parts.append(
                f"{history_label_prefix}ToolCall: {item.function.name}[{item.function.arguments}]"
            )
        target_entries = preview_parts or [f"{history_label_prefix}Assistant: <empty round {round_index}>"]
        for entry in target_entries:
            self._append_history_entry(
                entry,
                history_target=history_target,
                append_current_history=append_current_history,
            )
        payload = {"native_tool_calling": True, "round": round_index}
        if metadata:
            payload.update(metadata)
        return Message.assistant(
            result.text or None,
            tool_calls=tool_calls,
            metadata=payload,
        )

    def _execute_native_tool_call(
        self,
        tool_call: Dict[str, Any],
        *,
        history_target: Optional[List[str]] = None,
        append_current_history: bool = True,
        action_label: str = "Action",
        observation_label: str = "Observation",
    ) -> Dict[str, Any]:
        """执行一条原生 tool call，并把结果整理成标准 Observation。"""
        function = tool_call.get("function", {}) or {}
        tool_name = str(function.get("name", "")).strip()
        arguments = function.get("arguments", "")
        self._append_history_entry(
            f"{action_label}: {tool_name}[{arguments}]",
            history_target=history_target,
            append_current_history=append_current_history,
        )
        self._native_tool_execution_in_progress = True
        try:
            observation = self._handle_action(tool_name, arguments)
        finally:
            self._native_tool_execution_in_progress = False
        snapshot = self._consume_tool_result_snapshot(tool_name, observation)
        self._append_history_entry(
            f"{observation_label}: {observation}",
            history_target=history_target,
            append_current_history=append_current_history,
        )
        return snapshot

    def _run_native_tool_calling_loop(
        self,
        *,
        prompt: str,
        max_rounds: int,
        history_target: Optional[List[str]] = None,
        append_current_history: bool = True,
        history_label_prefix: str = "",
        action_label: str = "Action",
        observation_label: str = "Observation",
        empty_observation_message: str = "Native tool calling returned empty response.",
        assistant_metadata_factory: Optional[Callable[[int], Dict[str, Any]]] = None,
        tool_metadata_factory: Optional[Callable[[int, Dict[str, Any]], Dict[str, Any]]] = None,
        on_assistant_message: Optional[Callable[[Message, ChatResult, int], None]] = None,
        on_tool_observation: Optional[Callable[[str, Dict[str, Any], int], None]] = None,
        on_tool_message: Optional[Callable[[Message, Dict[str, Any], str, int], None]] = None,
        **kwargs: Any,
    ) -> str:
        """
        驱动一段可复用的原生 tool calling 循环。

        修改说明：把“构造消息 -> 请求模型 -> 回填 assistant/tool message -> 执行工具”的主链路
        下沉到公共父类，React / Plan / Reflection 只保留各自的 prompt、轮次和收尾逻辑。
        """
        messages = self._build_native_tool_calling_messages(prompt)

        for round_index in range(1, max_rounds + 1):
            result = self._request_result_with_messages(
                messages,
                tools=self.tool_registry.get_available_tools(),
                tool_choice="auto",
                **kwargs,
            )
            assistant_message = self._build_assistant_message_from_result(
                result,
                round_index=round_index,
                history_target=history_target,
                append_current_history=append_current_history,
                history_label_prefix=history_label_prefix,
                metadata=assistant_metadata_factory(round_index) if assistant_metadata_factory else None,
            )
            messages.append(assistant_message)
            if on_assistant_message is not None:
                on_assistant_message(assistant_message, result, round_index)

            if result.tool_calls:
                for tool_call in result.tool_calls:
                    execution_record = self._execute_native_tool_call(
                        tool_call,
                        history_target=history_target,
                        append_current_history=append_current_history,
                        action_label=action_label,
                        observation_label=observation_label,
                    )
                    observation = str(execution_record.get("observation", "") or "")
                    if on_tool_observation is not None:
                        on_tool_observation(observation, tool_call, round_index)
                    tool_message_metadata = dict(execution_record.get("message_metadata", {}) or {})
                    if tool_metadata_factory:
                        tool_message_metadata.update(tool_metadata_factory(round_index, tool_call))
                    tool_message = self._build_native_tool_message(
                        observation,
                        tool_call_id=tool_call.get("id") or f"tool_call_{round_index}",
                        tool_name=tool_call.get("function", {}).get("name"),
                        metadata=tool_message_metadata or None,
                    )
                    messages.append(tool_message)
                    if on_tool_message is not None:
                        on_tool_message(tool_message, tool_call, observation, round_index)
                continue

            final_text = (result.text or "").strip()
            if final_text:
                return final_text
            self._append_history_entry(
                f"{observation_label}: {empty_observation_message}",
                history_target=history_target,
                append_current_history=append_current_history,
            )

        return ""

    def _handle_action(self, action_type: str, action_input: Optional[str]) -> str:
        """子类需要实现具体的工具执行逻辑。"""
        raise NotImplementedError

    def _build_memory_context(self) -> str:
        """读取与当前输入相关的历史记忆，并整理成 prompt 片段。"""
        if self.memory_manager is None:
            return ""
        return self.memory_manager.build_memory_prompt(
            session_id=self.session_id,
            query=self.current_input,
            exclude_text=self.current_input,
        )

    def _build_context_packet(self):
        """构建本轮请求的结构化上下文。"""
        route = self._resolve_context_route(self.current_input)
        builder = ContextBuilder()
        if self.system_prompt:
            builder.add_system_prompt(self.system_prompt)
        builder.add_notes(
            "上下文策略",
            (
                f"当前判定的上下文路由: {route['route_name']}。\n"
                f"优先级顺序: 工具观察={route['tool_priority']} / 记忆={route['memory_priority']} / 检索={route['rag_priority']}。"
            ),
            priority=96,
            source="context_router",
        )
        builder.add_runtime_rules(
            [
                "优先遵守工具事实和显式上下文，不要编造未观察到的信息。",
                "如果历史记忆或检索上下文不足以支持结论，应明确说明不确定性。",
            ]
        )
        memory_sections = self._build_auto_memory_sections(route)
        session_summary = self._build_session_summary(route)
        if session_summary:
            builder.add_notes(
                "会话摘要",
                session_summary,
                priority=int(route["memory_priority"]) + 5,
                source="session_summary",
            )
        for section in memory_sections:
            builder.add_notes(
                section["title"],
                section["content"],
                priority=section["priority"],
                source="memory",
            )
        tool_context = self._build_tool_observation_context()
        if tool_context:
            builder.add_notes(
                "工具观察",
                tool_context,
                priority=route["tool_priority"],
                source="tool_observation",
            )
        rag_context = self._build_auto_rag_context(route)
        if rag_context:
            builder.add_notes(
                "检索上下文",
                rag_context,
                priority=route["rag_priority"],
                source="retrieval",
            )
        conflict_note = self._build_conflict_resolution_note(route=route, memory_sections=memory_sections)
        if conflict_note:
            builder.add_notes(
                "冲突消解",
                conflict_note,
                priority=94,
                source="conflict_resolution",
            )
        return builder.build()

    def _build_session_summary(self, route: Dict[str, int | str | bool]) -> str:
        """生成当前会话的轻量摘要，作为详细记忆前的一层压缩上下文。"""
        if self.memory_manager is None:
            return ""
        if bool(route.get("prefer_rag")) and not bool(route.get("prefer_memory")):
            # 修改说明：纯文档问答时减少摘要注入，避免会话摘要把 prompt 重点从证据检索拉偏。
            return ""
        return self.memory_manager.build_session_summary(
            session_id=self.session_id,
            query=self.current_input,
            exclude_text=self.current_input,
        )

    def _build_auto_memory_sections(self, route: Dict[str, int | str | bool]) -> List[Dict[str, int | str]]:
        """按当前路由策略生成结构化记忆 section。"""
        if self.memory_manager is None:
            return []

        if not self.config.auto_memory_context:
            memory_context = self._build_memory_context()
            if not memory_context:
                return []
            return [
                {
                    "title": "相关记忆",
                    "content": memory_context,
                    "priority": int(route["memory_priority"]),
                }
            ]

        sections = self.memory_manager.build_structured_memory_sections(
            session_id=self.session_id,
            query=self.current_input,
            exclude_text=self.current_input,
            limit=int(route["memory_limit"]),
        )
        if not sections:
            return []

        title_offsets = {
            "用户偏好": 3,
            "项目事实": 2,
            "近期对话": 1,
        }
        rendered_sections: List[Dict[str, int | str]] = []
        base_priority = int(route["memory_priority"])
        for title, lines in sections.items():
            body = "\n".join(f"- {line}" for line in lines)
            rendered_sections.append(
                {
                    "title": title,
                    "content": body,
                    "priority": base_priority + title_offsets.get(title, 0),
                }
            )
        return rendered_sections

    def _build_tool_observation_context(self) -> str:
        """把当前运行过程中已经确认过的工具观察整理成高优先级上下文。"""
        if not self.tool_observations:
            return ""

        recent_items = self.tool_observations[-self.config.tool_context_observation_limit :]
        lines = ["以下内容来自本轮已经执行过的工具，请优先相信这些观察结果："]
        for index, item in enumerate(recent_items, start=1):
            suffix = self._summarize_tool_observation_metadata(item)
            lines.append(f"{index}. [{item['tool']}] {item['observation']}{suffix}")
        return "\n".join(lines)

    def _build_auto_rag_context(self, route: Dict[str, int | str | bool]) -> str:
        """
        自动从 rag_tool 拉取与当前问题相关的检索上下文。

        修改说明：这样模型在真正决定“要不要手动调用 rag_tool”之前，
        就已经能先看到一层压缩后的检索结果；对问答类任务更稳，也更像实际项目里的预检索流程。
        """
        if not self.config.auto_rag_context:
            return ""
        query = self.current_input.strip()
        if not query or not self._should_use_auto_rag(query, route):
            return ""
        if query in self._rag_context_cache:
            return self._rag_context_cache[query]

        rag_tool = self.tool_registry.get_tool("rag_tool")
        if rag_tool is None or not hasattr(rag_tool, "rag_pipeline"):
            return ""

        try:
            sources = rag_tool.rag_pipeline.list_sources()
            if not sources:
                self._rag_context_cache[query] = ""
                return ""
            matches = rag_tool.rag_pipeline.search(
                query,
                limit=int(route["rag_limit"]),
            )
        except Exception:
            matches = []

        if not matches:
            context = ""
        else:
            self._rag_evidence_cache[query] = [
                {
                    "source": item.chunk.source,
                    "content": item.chunk.content,
                    "score": f"{item.score:.4f}",
                }
                for item in matches
            ]
            lines = ["以下内容来自自动 RAG 检索，请优先参考这些证据片段：", "参考结论："]
            for index, item in enumerate(matches, start=1):
                summary = " ".join(item.chunk.content.split())
                if len(summary) > 120:
                    summary = f"{summary[:120].rstrip()}..."
                lines.append(f"{index}. {summary}")
            lines.append("证据片段：")
            for index, item in enumerate(matches, start=1):
                lines.append(
                    f"{index}. 来源: {item.chunk.source} | 综合分数: {item.score:.4f}\n{item.chunk.content}"
                )
            context = "\n".join(lines)
        if not matches:
            self._rag_evidence_cache[query] = []
        self._rag_context_cache[query] = context
        return context

    @staticmethod
    def _should_use_auto_rag(query: str, route: Dict[str, int | str | bool]) -> bool:
        """
        粗略判断当前输入是否适合做自动检索。

        修改说明：像“加索引 / 清空索引 / 列出来源”这类工具管理动作不适合自动做 RAG 召回，
        否则只会给 prompt 里塞入无关上下文。
        """
        normalized = query.strip().lower()
        skip_keywords = (
            "加入索引",
            "建立索引",
            "添加文档",
            "清空索引",
            "列出来源",
            "sources",
            "action:add",
            "action:clear",
            "rag_tool add",
            "rag_tool clear",
        )
        if bool(route.get("prefer_memory")) and not bool(route.get("prefer_rag")):
            return False
        return not any(keyword in normalized for keyword in skip_keywords)

    def _resolve_context_route(self, query: str) -> Dict[str, int | str | bool]:
        """
        为当前问题选择一个轻量上下文路由策略。

        修改说明：当前先用启发式路由把问题粗分成“记忆优先 / 检索优先 / 工具优先 / 平衡”，
        后面如果做更复杂的 planner，也可以直接替换这一层。
        """
        normalized = query.strip().lower()
        memory_keywords = (
            "记得",
            "还记得",
            "我的",
            "我喜欢",
            "偏好",
            "习惯",
            "名字",
            "刚才",
            "之前",
            "preference",
            "remember",
            "profile",
        )
        rag_keywords = (
            "文档",
            "资料",
            "根据文档",
            "项目",
            "系统",
            "支持哪些",
            "知识库",
            "rag",
            "介绍一下",
            "说明",
        )
        tool_keywords = (
            "调用",
            "工具",
            "get_time",
            "当前时间",
            "本地时间",
            "执行",
            "查询",
            "计算",
        )

        if any(keyword in normalized for keyword in tool_keywords):
            route_name = "tool_first"
            return {
                "route_name": route_name,
                "memory_priority": 76,
                "tool_priority": 92,
                "rag_priority": 68,
                "memory_limit": self.config.auto_memory_context_limit,
                "rag_limit": max(1, self.config.auto_rag_context_limit - 1),
                "prefer_memory": False,
                "prefer_rag": False,
            }
        if any(keyword in normalized for keyword in memory_keywords):
            route_name = "memory_first"
            return {
                "route_name": route_name,
                "memory_priority": 88,
                "tool_priority": 82,
                "rag_priority": 62,
                "memory_limit": self.config.auto_memory_context_limit + 1,
                "rag_limit": max(1, self.config.auto_rag_context_limit - 1),
                "prefer_memory": True,
                "prefer_rag": False,
            }
        if any(keyword in normalized for keyword in rag_keywords):
            route_name = "rag_first"
            return {
                "route_name": route_name,
                "memory_priority": 74,
                "tool_priority": 82,
                "rag_priority": 88,
                "memory_limit": max(2, self.config.auto_memory_context_limit - 1),
                "rag_limit": self.config.auto_rag_context_limit + 1,
                "prefer_memory": False,
                "prefer_rag": True,
            }
        return {
            "route_name": "balanced",
            "memory_priority": 80,
            "tool_priority": 85,
            "rag_priority": 78,
            "memory_limit": self.config.auto_memory_context_limit,
            "rag_limit": self.config.auto_rag_context_limit,
            "prefer_memory": False,
            "prefer_rag": True,
        }

    def _remember_tool_observation(
        self,
        tool_name: str,
        observation: str,
        *,
        result: Optional[ToolResult] = None,
    ) -> None:
        """记录本轮工具观察，供上下文工程优先注入。"""
        cleaned = observation.strip()
        if not cleaned:
            return
        payload: Dict[str, Any] = {
            "tool": tool_name,
            "observation": cleaned,
        }
        if result is not None:
            payload.update(
                {
                    "success": result.success,
                    "meta": dict(result.meta),
                    "data_preview": self._preview(self._stringify_tool_payload(result.data), limit=160),
                }
            )
        self.tool_observations.append(payload)
        if tool_name == "rag_tool":
            # 修改说明：rag_tool 会改变可检索上下文，执行后清掉缓存，保证下一轮拿到最新索引状态。
            self._rag_context_cache = {}
            self._rag_evidence_cache = {}

    def _remember_tool_result_memory(self, tool_name: str, observation: str, result: ToolResult) -> None:
        """把工具结果按统一协议写入记忆系统，便于后续检索和排查。"""
        if self.memory_manager is None or not observation.strip():
            return
        self.memory_manager.record_message(
            session_id=self.session_id,
            role="tool",
            content=observation,
            metadata=self._build_tool_result_memory_metadata(tool_name, result, observation),
            persist=self._should_persist_role("tool"),
        )

    def _build_tool_result_memory_metadata(
        self,
        tool_name: str,
        result: ToolResult,
        observation: str,
    ) -> Dict[str, Any]:
        """把 ToolResult 压缩成适合写进 metadata 的结构。"""
        return {
            "source": "tool_result",
            "tool_name": tool_name,
            "tool_success": result.success,
            "tool_result_meta": dict(result.meta),
            "tool_result_data_preview": self._preview(self._stringify_tool_payload(result.data), limit=240),
            "tool_error": result.error,
            "tool_observation": observation,
        }

    def _stash_tool_result_snapshot(self, tool_name: str, observation: str, result: ToolResult) -> None:
        """缓存最近一次工具执行的结构化快照，供 native tool message 回填。"""
        self._last_tool_result_snapshot = {
            "tool_name": tool_name,
            "observation": observation,
            "result": result,
            "message_metadata": self._build_tool_result_memory_metadata(tool_name, result, observation),
        }

    def _consume_tool_result_snapshot(
        self,
        tool_name: str,
        observation: str,
    ) -> Dict[str, Any]:
        """取出最近一次工具结果快照；如果没有，就返回最小回退结构。"""
        snapshot = self._last_tool_result_snapshot
        self._last_tool_result_snapshot = None
        if snapshot is None:
            return {
                "tool_name": tool_name,
                "observation": observation,
                "message_metadata": {
                    "source": "tool_result",
                    "tool_name": tool_name,
                    "tool_observation": observation,
                },
            }
        return snapshot

    @staticmethod
    def _stringify_tool_payload(payload: Any) -> str:
        """把 tool data / meta 之类的结构化内容压成可读短文本。"""
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        return repr(payload)

    def _summarize_tool_observation_metadata(self, item: Dict[str, Any]) -> str:
        """把工具结果里的 meta/data 摘成一小段附加说明，避免上下文过长。"""
        parts: List[str] = []
        meta = item.get("meta")
        if isinstance(meta, dict):
            for key in ("action", "count", "written", "cleared"):
                if key in meta:
                    parts.append(f"{key}={meta[key]}")
        data_preview = str(item.get("data_preview", "") or "").strip()
        if data_preview:
            parts.append(f"data={self._preview(data_preview, limit=80)}")
        if not parts:
            return ""
        return f" | {'; '.join(parts)}"

    def _build_conflict_resolution_note(
        self,
        *,
        route: Dict[str, int | str | bool],
        memory_sections: List[Dict[str, int | str]],
    ) -> str:
        """
        检查 memory / rag / tool observation 之间是否存在明显冲突，并给出消解规则。

        修改说明：这一步不是做“真相判断器”，而是给模型一个清晰的取舍框架，
        避免它在上下文出现相反说法时自己随意拼接出自相矛盾的回答。
        """
        if not self.config.enable_context_conflict_resolution:
            return ""

        policy_lines = [
            "当不同上下文来源出现冲突时，请按下面规则处理：",
            "1. 工具观察始终优先于自动记忆和自动 RAG。",
            "2. 如果是“用户偏好 / 个性化信息”冲突，优先采用记忆。",
            "3. 如果是“项目事实 / 文档事实”冲突，优先采用 RAG 证据。",
            "4. 如果仍然无法判断，请明确说明存在冲突，不要擅自融合成一个新事实。",
        ]

        claims: List[Dict[str, str]] = []
        claims.extend(self._extract_claims_from_memory_sections(memory_sections))
        claims.extend(self._extract_claims_from_tool_observations())
        claims.extend(self._extract_claims_from_rag_evidence())
        conflicts = self._detect_conflicts(claims)

        if not conflicts:
            return "\n".join(policy_lines)

        lines = policy_lines + ["", "当前检测到的潜在冲突："]
        for index, conflict in enumerate(conflicts, start=1):
            winner = self._resolve_conflict_winner(conflict["category"], conflict["claims"], route)
            lines.append(
                f"{index}. 主题: {conflict['topic']} | 类型: {conflict['category']} | 当前优先采用: {winner['source_label']}"
            )
            lines.append(f"   原因: {winner['reason']}")
            for claim in conflict["claims"]:
                lines.append(
                    f"   - 来源={claim['source_label']} | 说法={claim['text']}"
                )
        return "\n".join(lines)

    def _extract_claims_from_memory_sections(
        self,
        memory_sections: List[Dict[str, int | str]],
    ) -> List[Dict[str, str]]:
        claims: List[Dict[str, str]] = []
        for section in memory_sections:
            title = str(section["title"])
            lines = str(section["content"]).splitlines()
            for line in lines:
                cleaned = line.lstrip("- ").strip()
                if not cleaned:
                    continue
                claims.extend(
                    self._extract_claims_from_text(
                        cleaned,
                        source="memory",
                        source_label=f"记忆/{title}",
                    )
                )
        return claims

    def _extract_claims_from_tool_observations(self) -> List[Dict[str, str]]:
        claims: List[Dict[str, str]] = []
        for item in self.tool_observations:
            claims.extend(
                self._extract_claims_from_text(
                    item["observation"],
                    source="tool",
                    source_label=f"工具/{item['tool']}",
                )
            )
        return claims

    def _extract_claims_from_rag_evidence(self) -> List[Dict[str, str]]:
        claims: List[Dict[str, str]] = []
        for item in self._rag_evidence_cache.get(self.current_input.strip(), []):
            claims.extend(
                self._extract_claims_from_text(
                    item["content"],
                    source="rag",
                    source_label=f"RAG/{item['source']}",
                )
            )
        return claims

    def _extract_claims_from_text(
        self,
        text: str,
        *,
        source: str,
        source_label: str,
    ) -> List[Dict[str, str]]:
        claims: List[Dict[str, str]] = []
        compact = " ".join(text.split())

        for match in self._PREFERENCE_PATTERN.finditer(compact):
            obj = self._normalize_claim_object(match.group("object"))
            polarity = "negative" if self._is_negative_preference(match.group("neg"), match.group("verb")) else "positive"
            claims.append(
                {
                    "topic": f"preference:{obj}",
                    "category": "用户偏好",
                    "polarity": polarity,
                    "source": source,
                    "source_label": source_label,
                    "text": compact,
                }
            )

        for match in self._SUPPORT_PATTERN.finditer(compact):
            obj = self._normalize_claim_object(match.group("object"))
            polarity = "negative" if match.group("neg") else "positive"
            claims.append(
                {
                    "topic": f"support:{obj}",
                    "category": "项目事实",
                    "polarity": polarity,
                    "source": source,
                    "source_label": source_label,
                    "text": compact,
                }
            )

        for match in self._CONTAINS_PATTERN.finditer(compact):
            obj = self._normalize_claim_object(match.group("object"))
            polarity = "negative" if match.group("neg") else "positive"
            claims.append(
                {
                    "topic": f"contains:{obj}",
                    "category": "项目事实",
                    "polarity": polarity,
                    "source": source,
                    "source_label": source_label,
                    "text": compact,
                }
            )

        return claims

    @staticmethod
    def _normalize_claim_object(value: str) -> str:
        return value.strip().lower()

    @staticmethod
    def _is_negative_preference(negation: str | None, verb: str | None) -> bool:
        if verb == "讨厌":
            return True
        return bool(negation)

    @staticmethod
    def _detect_conflicts(claims: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, str]]] = {}
        for claim in claims:
            grouped.setdefault(claim["topic"], []).append(claim)

        conflicts: List[Dict[str, Any]] = []
        for topic, items in grouped.items():
            polarities = {item["polarity"] for item in items}
            sources = {item["source"] for item in items}
            if len(polarities) < 2 or len(sources) < 2:
                continue
            conflicts.append(
                {
                    "topic": topic,
                    "category": items[0]["category"],
                    "claims": items,
                }
            )
        return conflicts

    @staticmethod
    def _resolve_conflict_winner(
        category: str,
        claims: List[Dict[str, str]],
        route: Dict[str, int | str | bool],
    ) -> Dict[str, str]:
        tool_claim = next((item for item in claims if item["source"] == "tool"), None)
        if tool_claim is not None:
            return {
                "source_label": tool_claim["source_label"],
                "reason": "工具调用得到的 observation 是当前最直接的事实来源。",
            }

        if category == "用户偏好":
            memory_claim = next((item for item in claims if item["source"] == "memory"), None)
            if memory_claim is not None:
                return {
                    "source_label": memory_claim["source_label"],
                    "reason": "这类信息属于个性化记忆，默认优先采用记忆中的历史偏好。",
                }

        if category == "项目事实":
            rag_claim = next((item for item in claims if item["source"] == "rag"), None)
            if rag_claim is not None:
                return {
                    "source_label": rag_claim["source_label"],
                    "reason": "这类信息更接近文档事实，默认优先采用 RAG 检索到的证据。",
                }

        preferred_source = "memory" if bool(route.get("prefer_memory")) else "rag"
        preferred_claim = next((item for item in claims if item["source"] == preferred_source), None)
        if preferred_claim is not None:
            route_name = str(route.get("route_name", "balanced"))
            return {
                "source_label": preferred_claim["source_label"],
                "reason": f"当前问题命中 `{route_name}` 路由，因此优先采用该来源。",
            }

        fallback_claim = claims[0]
        return {
            "source_label": fallback_claim["source_label"],
            "reason": "未找到更高优先级来源，暂时保留首个来源并提示不确定性。",
        }

    def _remember_message(self, message: Message, persist: Optional[bool] = None) -> None:
        """把消息放进运行态历史，并按配置决定是否写入长期记忆。"""
        self.add_message(message)

        if self.memory_manager is None or not (message.content or "").strip():
            return

        resolved_persist = persist
        if resolved_persist is None:
            resolved_persist = self._should_persist_role(message.role)

        self.memory_manager.record_message(
            session_id=self.session_id,
            role=message.role,
            content=message.content or "",
            metadata=message.metadata,
            persist=resolved_persist,
        )

    def _remember_assistant_text(self, text: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """在一次任务结束时写入最终 assistant 回复。"""
        content = text.strip()
        if not content:
            return
        self._remember_message(
            Message.assistant(content, metadata=metadata or {"memory_stage": "final_answer"})
        )

    def _should_persist_role(self, role: str) -> bool:
        """根据配置决定某种角色的消息是否进入持久记忆。"""
        if self.memory_manager is None:
            return False
        config = self.memory_manager.config
        if role == "user":
            return config.persist_user_messages
        if role == "assistant":
            return config.persist_assistant_messages
        if role == "tool":
            return config.persist_tool_messages
        return False

    @staticmethod
    def _render_history(history: List[str], empty_text: str = "暂无历史记录") -> str:
        """把历史列表渲染成 prompt 文本；为空时返回默认占位说明。"""
        return "\n".join(history) or empty_text

    @staticmethod
    def _preview(text: str, limit: int = 200) -> str:
        """生成日志预览文本，避免长输出把终端刷满。"""
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return f"{compact[:limit]} ..."
