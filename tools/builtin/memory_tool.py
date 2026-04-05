from __future__ import annotations

from typing import Any, Dict, List

from memory.manager import MemoryManager
from tools.builtin.tool_base import Tool, ToolParameter


class MemoryTool(Tool):
    """为 Agent 暴露最小可用的记忆读写能力。"""

    def __init__(self, memory_manager: MemoryManager, session_id: str) -> None:
        super().__init__(
            name="memory_tool",
            description="读取、写入或清空当前会话的记忆。建议使用 JSON 对象参数。",
        )
        self.memory_manager = memory_manager
        self.session_id = session_id

    def run(self, parameters: Dict[str, Any]) -> str:
        """
        支持四种动作：
        - recent: 查看最近记忆
        - search: 按 query 搜索记忆
        - context: 返回结构化记忆上下文
        - summary: 返回当前会话摘要
        - remember: 手动写入一条记忆
        - clear: 清空当前会话记忆
        """
        action = str(parameters.get("action", "recent")).strip().lower()
        limit = int(parameters.get("limit", 5) or 5)

        if action == "recent":
            items = self.memory_manager.recall(session_id=self.session_id, limit=limit)
            return self._format_items(items)

        if action == "search":
            query = str(parameters.get("query", "")).strip()
            items = self.memory_manager.recall(
                session_id=self.session_id,
                query=query,
                limit=limit,
            )
            return self._format_items(items)

        if action == "context":
            query = str(parameters.get("query", "")).strip()
            rendered = self.memory_manager.build_structured_memory_prompt(
                session_id=self.session_id,
                query=query or None,
                exclude_text=query or None,
                limit=limit,
            )
            return rendered or "没有找到相关记忆。"

        if action == "summary":
            query = str(parameters.get("query", "")).strip()
            rendered = self.memory_manager.build_session_summary(
                session_id=self.session_id,
                query=query or None,
                exclude_text=query or None,
            )
            return rendered or "当前还没有足够的会话内容可供摘要。"

        if action == "remember":
            content = str(parameters.get("content", "")).strip()
            if not content:
                return "memory_tool remember 需要提供 content。"
            self.memory_manager.record_message(
                session_id=self.session_id,
                role="assistant",
                content=content,
                metadata={"source": "memory_tool"},
            )
            return f"已写入记忆: {content}"

        if action == "clear":
            self.memory_manager.clear_session(self.session_id)
            return "当前会话记忆已清空。"

        return f"不支持的 memory_tool action: {action}"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="动作类型，可选 recent/search/context/summary/remember/clear。",
                choices=["recent", "search", "context", "summary", "remember", "clear"],
            ),
            ToolParameter(
                name="query",
                type="string",
                description="当 action=search 时使用的搜索词。",
                required=False,
                default="",
            ),
            ToolParameter(
                name="content",
                type="string",
                description="当 action=remember 时要写入的记忆内容。",
                required=False,
                default="",
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="返回记忆条数上限。",
                required=False,
                default=5,
            ),
        ]

    @staticmethod
    def _format_items(items: List[Any]) -> str:
        if not items:
            return "没有找到相关记忆。"
        return "\n".join(f"- [{item.role}] {item.content}" for item in items)
