from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from memory.base import MemoryConfig
from memory.embedding import BaseEmbeddingService, EmbeddingServiceFactory
from memory.rag.document import DocumentChunk, DocumentProcessor, RetrievedChunk


class RagPipeline:
    """一个最小可用的本地 RAG 管道。"""

    def __init__(
        self,
        config: MemoryConfig,
        embedding_service: BaseEmbeddingService | None = None,
    ) -> None:
        self.config = config
        self.embedding_service = embedding_service or EmbeddingServiceFactory.create(config)
        self.processor = DocumentProcessor(
            chunk_size=config.rag_chunk_size,
            chunk_overlap=config.rag_chunk_overlap,
        )
        self.store_path = Path(config.rag_store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self.store_path.write_text("[]", encoding="utf-8")

    def add_document(self, path: str) -> int:
        """
        读取文档、切片并建立本地向量索引。

        修改说明：这里先用 JSON 文件保存切片和向量，
        先把“文档入库 -> 检索 -> 返回上下文”的最小闭环跑起来。
        """
        chunks = self.processor.load(path)
        if not chunks:
            return 0

        records = self._load_records()
        source = str(Path(path).resolve())
        records = [record for record in records if record["chunk"]["source"] != source]

        for chunk in chunks:
            records.append(
                {
                    "chunk": chunk.model_dump(mode="json"),
                    "vector": self.embedding_service.embed(chunk.content),
                }
            )

        self._save_records(records)
        return len(chunks)

    def search(self, query: str, limit: int | None = None) -> List[RetrievedChunk]:
        """按向量相似度返回最相关的文档切片。"""
        query_vector = self.embedding_service.embed(query)
        scored_items: List[RetrievedChunk] = []
        for record in self._load_records():
            chunk = DocumentChunk.model_validate(record["chunk"])
            score = self.embedding_service.cosine_similarity(query_vector, record.get("vector", []))
            if score <= 0:
                continue
            scored_items.append(RetrievedChunk(chunk=chunk, score=score))

        resolved_limit = limit or self.config.rag_top_k
        scored_items.sort(key=lambda item: item.score, reverse=True)
        return scored_items[:resolved_limit]

    def answer(self, query: str, limit: int | None = None) -> str:
        """把检索结果整理成可直接给模型参考的上下文摘要。"""
        matches = self.search(query, limit=limit)
        if not matches:
            return "没有检索到相关文档。"

        lines = ["以下是检索到的参考资料片段："]
        for index, item in enumerate(matches, start=1):
            lines.append(
                f"{index}. 来源: {item.chunk.source} | 分数: {item.score:.4f}\n{item.chunk.content}"
            )
        return "\n\n".join(lines)

    def clear(self) -> None:
        """清空当前 RAG 索引。"""
        self._save_records([])

    def list_sources(self) -> List[str]:
        """列出当前已入库的文档来源。"""
        sources = {record["chunk"]["source"] for record in self._load_records()}
        return sorted(sources)

    def run(self, query: str, documents: List[DocumentChunk]) -> str:
        """
        保留旧接口兼容性。

        修改说明：旧版本只接受外部传入的切片列表，
        现在仍允许这样调用，但内部会统一走检索格式输出。
        """
        if documents:
            temp_records: List[Dict[str, Any]] = []
            for chunk in documents:
                temp_records.append(
                    {
                        "chunk": chunk.model_dump(mode="json"),
                        "vector": self.embedding_service.embed(chunk.content),
                    }
                )
            matches = self._search_records(query, temp_records, limit=self.config.rag_top_k)
            return self._format_matches(matches)
        return self.answer(query)

    def _search_records(
        self,
        query: str,
        records: List[Dict[str, Any]],
        *,
        limit: int,
    ) -> List[RetrievedChunk]:
        query_vector = self.embedding_service.embed(query)
        matches: List[RetrievedChunk] = []
        for record in records:
            chunk = DocumentChunk.model_validate(record["chunk"])
            score = self.embedding_service.cosine_similarity(query_vector, record.get("vector", []))
            if score <= 0:
                continue
            matches.append(RetrievedChunk(chunk=chunk, score=score))
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:limit]

    def _format_matches(self, matches: List[RetrievedChunk]) -> str:
        if not matches:
            return "没有检索到相关文档。"
        return "\n\n".join(
            f"来源: {item.chunk.source} | 分数: {item.score:.4f}\n{item.chunk.content}"
            for item in matches
        )

    def _load_records(self) -> List[Dict[str, Any]]:
        raw_text = self.store_path.read_text(encoding="utf-8")
        return json.loads(raw_text or "[]")

    def _save_records(self, records: List[Dict[str, Any]]) -> None:
        self.store_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
