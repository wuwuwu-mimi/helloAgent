from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from tools.builtin.tool_base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """维护工具对象，并提供统一的注册、查询和描述生成能力。"""

    def __init__(self) -> None:
        # 修改说明：注册表里直接保存 Tool 对象，这样工具的名称、描述、参数定义和执行逻辑都能收敛到一个类里。
        self.tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool) -> None:
        """注册一个工具对象；同名工具会被新的定义覆盖。"""
        if tool.name in self.tools:
            logger.warning("工具 `%s` 已存在，新的定义会覆盖旧版本。", tool.name)
        else:
            # 修改说明：注册日志降到 DEBUG，避免 main.py 演示时终端输出过于嘘杂。
            logger.debug("注册工具: %s", tool.name)
        self.tools[tool.name] = tool

    def registerTool(self, tool: Tool) -> None:
        """兼容旧命名风格，内部统一转到 `register_tool()`。"""
        self.register_tool(tool)

    def get_tool(self, name: str) -> Optional[Tool]:
        """根据工具名获取 Tool 对象。"""
        return self.tools.get(name)

    def getTool(self, name: str) -> Optional[Tool]:
        """兼容旧命名风格，内部统一转到 `get_tool()`。"""
        return self.get_tool(name)

    def list_tools(self) -> List[Tool]:
        """返回当前所有已注册工具对象。"""
        return list(self.tools.values())

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        生成兼容 function calling 的工具描述列表。

        当前文本版 Agent 主要拿它来做工具清单展示；
        后面如果切到原生 tool calling，这份结构也可以直接复用。
        """
        tools: List[Dict[str, Any]] = []
        for tool in self.list_tools():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.get_parameters_schema(),
                    },
                }
            )
        return tools

    def getAvailableTools(self) -> List[Dict[str, Any]]:
        """兼容旧命名风格，内部统一转到 `get_available_tools()`。"""
        return self.get_available_tools()

    def describe_tools(self) -> str:
        """把所有工具渲染成适合直接写进 Prompt 的文本。"""
        tools = self.list_tools()
        if not tools:
            return "- 当前没有可用工具"
        return "\n".join(tool.format_for_prompt() for tool in tools)
