from abc import ABC, abstractmethod
from typing import Optional, List

from core import HelloAgentsLLM, Config, Message


class Agent(ABC):
    """Agent 基类"""

    def __init__(self,
                 name: str,
                 llm: HelloAgentsLLM,
                 system_prompt: Optional[str] = None,
                 config: Optional[Config] = None):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: List[Message] = []

    @abstractmethod
    def run(self, input_text: str, stream: bool = True, **kwargs) -> str:
        """运行Agent"""
        pass

    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)

    def clear_history(self):
        """清空历史记录"""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()

    def __str__(self):
        return f"Agent(name={self.name},provider={self.llm.provider})"
