from __future__ import annotations

from typing import Any, List, Optional

from agents.agent_base import Agent
from core import Config, HelloAgentsLLM, Message
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

    def _start_new_run(self, input_text: str) -> None:
        """
        为一次新的任务重置运行态。

        修改说明：把“清空历史 + 注入 system/user 消息”的样板逻辑抽到公共父类，
        这样后续新增 Agent 范式时不用再重复写一遍。
        """
        self.current_history = []
        self.current_input = input_text
        self.clear_history()

        if self.system_prompt:
            self.add_message(Message.system(self.system_prompt))
        self._remember_message(Message.user(input_text))

    def _build_messages(self, prompt: str) -> List[Message]:
        """把本轮 prompt 包装成发给 LLM 的消息列表。"""
        messages: List[Message] = []
        if self.system_prompt:
            messages.append(Message.system(self.system_prompt))
        memory_context = self._build_memory_context()
        if memory_context:
            # 修改说明：记忆上下文作为额外 system message 注入，避免污染原始用户问题。
            messages.append(Message.system(memory_context, metadata={"source": "memory"}))
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
