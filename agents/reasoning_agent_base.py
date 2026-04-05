from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.agent_base import Agent
from core import Config, ContextBuilder, HelloAgentsLLM, Message
from memory.manager import MemoryManager
from tools.builtin.toolRegistry import ToolRegistry


class ReasoningAgentBase(Agent):
    """带有统一消息构建、LLM 调用和运行期历史管理能力的公共父类。"""

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
        self.tool_observations: List[Dict[str, str]] = []
        self._rag_context_cache: Dict[str, str] = {}

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
        result = self.llm.chat(
            messages=self._build_messages(prompt),
            stream=False,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs,
        )
        return (result.text or "").strip()

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
        for section in self._build_auto_memory_sections(route):
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
        return builder.build()

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
            lines.append(f"{index}. [{item['tool']}] {item['observation']}")
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

    def _remember_tool_observation(self, tool_name: str, observation: str) -> None:
        """记录本轮工具观察，供上下文工程优先注入。"""
        cleaned = observation.strip()
        if not cleaned:
            return
        self.tool_observations.append({"tool": tool_name, "observation": cleaned})
        if tool_name == "rag_tool":
            # 修改说明：rag_tool 会改变可检索上下文，执行后清掉缓存，保证下一轮拿到最新索引状态。
            self._rag_context_cache = {}

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
