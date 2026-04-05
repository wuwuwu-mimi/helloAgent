from __future__ import annotations

from typing import Any, Dict, List

from memory.rag import RagPipeline
from tools.builtin.tool_base import Tool, ToolConditionalRule, ToolParameter, ToolResult


class RagTool(Tool):
    """为 Agent 暴露最小可用的本地 RAG 能力。"""

    def __init__(self, rag_pipeline: RagPipeline) -> None:
        super().__init__(
            name="rag_tool",
            description="索引本地文档并检索相关内容，建议使用 JSON 对象参数。",
        )
        self.rag_pipeline = rag_pipeline

    def run(self, parameters: Dict[str, Any]) -> ToolResult:
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
                return ToolResult.fail(
                    "rag_tool add 需要提供 path。",
                    meta={"tool": "rag_tool", "action": action},
                )
            count = self.rag_pipeline.add_document(path)
            return ToolResult.ok(
                f"已索引文档 `{path}`，共写入 {count} 个切片。",
                data={"path": path, "chunk_count": count},
                meta={"tool": "rag_tool", "action": action, "written": count},
            )

        if action == "search":
            query = str(parameters.get("query", "")).strip()
            if not query:
                return ToolResult.fail(
                    "rag_tool search 需要提供 query。",
                    meta={"tool": "rag_tool", "action": action},
                )
            matches = self.rag_pipeline.search(query, limit=limit)
            if not matches:
                return ToolResult.ok(
                    "没有检索到相关文档。",
                    data={"query": query, "matches": []},
                    meta={"tool": "rag_tool", "action": action, "count": 0},
                )
            content = "\n\n".join(
                f"来源: {item.chunk.source} | 分数: {item.score:.4f}\n{item.chunk.content}"
                for item in matches
            )
            return ToolResult.ok(
                content,
                data={
                    "query": query,
                    "matches": [item.model_dump() for item in matches],
                },
                meta={"tool": "rag_tool", "action": action, "count": len(matches)},
            )

        if action == "answer":
            query = str(parameters.get("query", "")).strip()
            if not query:
                return ToolResult.fail(
                    "rag_tool answer 需要提供 query。",
                    meta={"tool": "rag_tool", "action": action},
                )
            answer = self.rag_pipeline.answer(query, limit=limit)
            return ToolResult.ok(
                answer,
                data={"query": query, "answer": answer},
                meta={"tool": "rag_tool", "action": action, "limit": limit},
            )

        if action == "context":
            query = str(parameters.get("query", "")).strip()
            if not query:
                return ToolResult.fail(
                    "rag_tool context 需要提供 query。",
                    meta={"tool": "rag_tool", "action": action},
                )
            matches = self.rag_pipeline.search(query, limit=limit)
            if not matches:
                return ToolResult.ok(
                    "没有检索到相关文档。",
                    data={"query": query, "matches": []},
                    meta={"tool": "rag_tool", "action": action, "count": 0},
                )
            context = self.rag_pipeline.build_answer_context(query=query, matches=matches)
            return ToolResult.ok(
                context,
                data={
                    "query": query,
                    "matches": [item.model_dump() for item in matches],
                },
                meta={"tool": "rag_tool", "action": action, "count": len(matches)},
            )

        if action == "clear":
            self.rag_pipeline.clear()
            return ToolResult.ok(
                "当前 RAG 索引已清空。",
                meta={"tool": "rag_tool", "action": action, "cleared": True},
            )

        if action == "sources":
            sources = self.rag_pipeline.list_sources()
            content = "\n".join(sources) if sources else "当前还没有已索引文档。"
            return ToolResult.ok(
                content,
                data={"sources": sources},
                meta={"tool": "rag_tool", "action": action, "count": len(sources)},
            )

        return ToolResult.fail(
            f"不支持的 rag_tool action: {action}",
            data={"action": action},
            meta={"tool": "rag_tool", "action": action},
        )

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
                min_length=1,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="当 action=search 或 answer 时使用的查询内容。",
                required=False,
                default="",
                min_length=1,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="返回切片条数上限。",
                required=False,
                default=3,
                minimum=1,
            ),
        ]

    def validate_normalized_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """当前 action 级校验已经下沉到条件 schema，这里保留扩展入口。"""
        return parameters

    def get_conditional_rules(self) -> List[ToolConditionalRule]:
        """
        导出 rag_tool 的条件 schema。

        修改说明：把 action 与 path/query 的依赖关系显式导出后，
        原生 tool calling 模式也能拿到更贴近真实约束的 schema。
        """
        return [
            ToolConditionalRule(field="action", equals="add", non_empty=["path"]),
            ToolConditionalRule(field="action", equals="search", non_empty=["query"]),
            ToolConditionalRule(field="action", equals="answer", non_empty=["query"]),
            ToolConditionalRule(field="action", equals="context", non_empty=["query"]),
        ]
