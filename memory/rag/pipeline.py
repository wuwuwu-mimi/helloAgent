from __future__ import annotations

from typing import List

from memory.rag.document import DocumentChunk


class RagPipeline:
    """
    RAG 管道占位实现。

    当前项目先把结构搭出来，真正的检索增强处理后续再补。
    """

    def run(self, query: str, documents: List[DocumentChunk]) -> str:
        del documents
        return f"RAG pipeline placeholder for query: {query}"
