from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel


class ToolParameter(BaseModel):
    """定义工具参数，便于统一生成说明文本和参数 Schema。"""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class Tool(ABC):
    """所有工具的抽象基类。"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具，并返回字符串结果。"""

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """返回工具参数定义列表。"""

    def get_parameters_schema(self) -> Dict[str, Any]:
        """
        生成兼容 function calling 的参数 Schema。

        这样注册器和 Agent 不需要自己再拼装一遍参数描述，
        后面如果切到原生 tool calling，也可以直接复用。
        """
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for parameter in self.get_parameters():
            field_schema: Dict[str, Any] = {
                "type": parameter.type,
                "description": parameter.description,
            }
            if parameter.default is not None:
                field_schema["default"] = parameter.default

            properties[parameter.name] = field_schema
            if parameter.required:
                required.append(parameter.name)

        schema: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        return schema

    def format_for_prompt(self) -> str:
        """
        生成适合直接写进 Prompt 的工具说明文本。

        文本版 ReAct 主要依赖这段说明来决定该怎么写 Action。
        """
        parameters = self.get_parameters()
        if not parameters:
            return f"- {self.name}: {self.description} 用法: {self.name}[]"

        parameter_text = ", ".join(
            f"{item.name}<{item.type}>{' 必填' if item.required else ' 可选'}: {item.description}"
            for item in parameters
        )
        return f"- {self.name}: {self.description} 参数: {parameter_text}"