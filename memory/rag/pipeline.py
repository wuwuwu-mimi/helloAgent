from __future__ import annotations

from pathlib import Path
from memory.base import MemoryConfig
from memory.embedding import BaseEmbeddingService, EmbeddingServiceFactory
from memory.rag.document import DocumentChunk, DocumentProcessor, RetrievedChunk
from memory.storage.qdrant_store import QdrantVectorStore


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
        self.store = QdrantVectorStore(
            config=config,
            embedding_service=self.embedding_service,
            collection_name=config.qdrant_rag_collection,
            store_path=config.rag_store_path,
        )

    def add_document(self, path: str) -> int:
        """
        读取文档、切片并建立本地向量索引。

        修改说明：这里先用 JSON 文件保存切片和向量，
        先把“文档入库 -> 检索 -> 返回上下文”的最小闭环跑起来。
        """
        chunks = self.processor.load(path)
        if not chunks:
            return 0

        raw_source = str(Path(path))
        source = str(Path(path).resolve())
        self.store.clear_records(filters={"source": raw_source})
        if source != raw_source:
            self.store.clear_records(filters={"source": source})
        for chunk in chunks:
            self.store.upsert_record(
                record_id=chunk.chunk_id,
                vector=self.embedding_service.embed(chunk.content),
                payload={
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "created_at": chunk.chunk_id,
                    "chunk": chunk.model_dump(mode="json"),
                },
            )
        return len(chunks)

    def search(self, query: str, limit: int | None = None) -> List[RetrievedChunk]:
        """按向量相似度返回最相关的文档切片。"""
        resolved_limit = limit or self.config.rag_top_k
        matches = self.store.search_records(
            vector=self.embedding_service.embed(query),
            limit=resolved_limit,
            score_threshold=0.0,
        )
        return [
            RetrievedChunk(
                chunk=DocumentChunk.model_validate(item["payload"]["chunk"]),
                score=item["score"],
            )
            for item in matches
            if "chunk" in item["payload"]
        ]

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
        self.store.clear_records()

    def list_sources(self) -> List[str]:
        """列出当前已入库的文档来源。"""
        sources = {
            record["payload"].get("source", "")
            for record in self.store.list_recent_records(limit=1000)
        }
        sources.discard("")
        return sorted(sources)

    def run(self, query: str, documents: List[DocumentChunk]) -> str:
        """
        保留旧接口兼容性。

        修改说明：旧版本只接受外部传入的切片列表，
        现在仍允许这样调用，但内部会统一走检索格式输出。
        """
        if documents:
            matches = self._search_inline_documents(query, documents, limit=self.config.rag_top_k)
            return self._format_matches(matches)
        return self.answer(query)

    def _search_inline_documents(
        self,
        query: str,
        documents: List[DocumentChunk],
        *,
        limit: int,
    ) -> List[RetrievedChunk]:
        query_vector = self.embedding_service.embed(query)
        matches: List[RetrievedChunk] = []
        for chunk in documents:
            score = self.embedding_service.cosine_similarity(
                query_vector,
                self.embedding_service.embed(chunk.content),
            )
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
