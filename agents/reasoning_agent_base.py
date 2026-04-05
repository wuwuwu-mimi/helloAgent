from __future__ import annotations

from typing import Any, List, Optional

from agents.agent_base import Agent
from core import Config, HelloAgentsLLM, Message
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
    ) -> None:
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.prompt_template = prompt_template
        self.current_history: List[str] = []

    def _start_new_run(self, input_text: str) -> None:
        """
        为一次新的任务重置运行态。

        修改说明：把“清空历史 + 注入 system/user 消息”的样板逻辑抽到公共父类，
        这样后续新增 Agent 范式时不用再重复写一遍。
        """
        self.current_history = []
        self.clear_history()

        if self.system_prompt:
            self.add_message(Message.system(self.system_prompt))
        self.add_message(Message.user(input_text))

    def _build_messages(self, prompt: str) -> List[Message]:
        """把本轮 prompt 包装成发给 LLM 的消息列表。"""
        messages: List[Message] = []
        if self.system_prompt:
            messages.append(Message.system(self.system_prompt))
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
