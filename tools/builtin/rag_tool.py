from __future__ import annotations

from typing import Any, Dict, List

from memory.rag import RagPipeline
from tools.builtin.tool_base import Tool, ToolParameter


class RagTool(Tool):
    """为 Agent 暴露最小可用的本地 RAG 能力。"""

    def __init__(self, rag_pipeline: RagPipeline) -> None:
        super().__init__(
            name="rag_tool",
            description="索引本地文档并检索相关内容，建议使用 JSON 对象参数。",
        )
        self.rag_pipeline = rag_pipeline

    def run(self, parameters: Dict[str, Any]) -> str:
        """
        支持五种动作：
        - add: 把本地文档切片并建立索引
        - search: 返回最相关的文档片段
        - answer: 返回带“参考结论 + 证据片段”的上下文摘要
        - context: 返回结构化检索上下文，适合进一步做 prompt 拼装
        - clear: 清空当前 RAG 索引
        """
        action = str(parameters.get("action", "search")).strip().lower()
        limit = int(parameters.get("limit", self.rag_pipeline.config.rag_top_k) or self.rag_pipeline.config.rag_top_k)

        if action == "add":
            path = str(parameters.get("path", "")).strip()
            if not path:
                return "rag_tool add 需要提供 path。"
            count = self.rag_pipeline.add_document(path)
            return f"已索引文档 `{path}`，共写入 {count} 个切片。"

        if action == "search":
            query = str(parameters.get("query", "")).strip()
            if not query:
                return "rag_tool search 需要提供 query。"
            matches = self.rag_pipeline.search(query, limit=limit)
            if not matches:
                return "没有检索到相关文档。"
            return "\n\n".join(
                f"来源: {item.chunk.source} | 分数: {item.score:.4f}\n{item.chunk.content}"
                for item in matches
            )

        if action == "answer":
            query = str(parameters.get("query", "")).strip()
            if not query:
                return "rag_tool answer 需要提供 query。"
            return self.rag_pipeline.answer(query, limit=limit)

        if action == "context":
            query = str(parameters.get("query", "")).strip()
            if not query:
                return "rag_tool context 需要提供 query。"
            matches = self.rag_pipeline.search(query, limit=limit)
            if not matches:
                return "没有检索到相关文档。"
            return self.rag_pipeline.build_answer_context(query=query, matches=matches)

        if action == "clear":
            self.rag_pipeline.clear()
            return "当前 RAG 索引已清空。"

        if action == "sources":
            sources = self.rag_pipeline.list_sources()
            return "\n".join(sources) if sources else "当前还没有已索引文档。"

        return f"不支持的 rag_tool action: {action}"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="动作类型，可选 add/search/answer/context/clear/sources。",
                choices=["add", "search", "answer", "context", "clear", "sources"],
            ),
            ToolParameter(
                name="path",
                type="string",
                description="当 action=add 时使用的本地文档路径。",
                required=False,
                default="",
            ),
            ToolParameter(
                name="query",
                type="string",
                description="当 action=search 或 answer 时使用的查询内容。",
                required=False,
                default="",
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="返回切片条数上限。",
                required=False,
                default=3,
            ),
        ]
