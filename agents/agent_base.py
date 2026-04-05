from abc import ABC, abstractmethod
from typing import List, Optional

from core import Config, HelloAgentsLLM, Message


class Agent(ABC):
    """所有 Agent 的基础抽象类。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: List[Message] = []

    @abstractmethod
    def run(self, input_text: str, stream: bool = True, **kwargs) -> str:
        """运行 Agent 主流程。"""

    def add_message(self, message: Message):
        """向内部历史记录中追加一条消息。"""
        self._history.append(message)

    def clear_history(self):
        """清空内部历史记录。"""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """获取当前历史记录的副本。"""
        return self._history.copy()

    def __str__(self):
        return f"Agent(name={self.name},provider={self.llm.provider})"