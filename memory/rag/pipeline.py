from __future__ import annotations

from pathlib import Path
import re
from typing import List
from uuid import NAMESPACE_URL, uuid5

from core.context_engineering import ContextBuilder
from memory.base import MemoryConfig
from memory.embedding import BaseEmbeddingService, EmbeddingServiceFactory
from memory.rag.document import DocumentChunk, DocumentProcessor, RetrievedChunk
from memory.storage.qdrant_store import QdrantVectorStore


class RagPipeline:
    """一个最小可用的本地 RAG 管道。"""

    _QUERY_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_\-+.]+|[\u4e00-\u9fff]{2,16}")

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

        修改说明：这里统一走向量存储适配层；
        有真实 Qdrant 配置时会直连云端，没有时再自动回退到本地 JSON。
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
                record_id=self._build_chunk_record_id(chunk),
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
        """按“向量召回 + 词项重排”的方式返回最相关切片。"""
        resolved_limit = limit or self.config.rag_top_k
        raw_matches = self.store.search_records(
            vector=self.embedding_service.embed(query),
            limit=max(resolved_limit * 3, resolved_limit),
            score_threshold=0.0,
        )
        matches = [
            RetrievedChunk(
                chunk=DocumentChunk.model_validate(item["payload"]["chunk"]),
                score=self._rerank_score(
                    query=query,
                    chunk=DocumentChunk.model_validate(item["payload"]["chunk"]),
                    vector_score=float(item["score"]),
                ),
            )
            for item in raw_matches
            if "chunk" in item["payload"]
        ]
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:resolved_limit]

    def answer(self, query: str, limit: int | None = None) -> str:
        """把检索结果重排后整理成更接近“可直接使用”的参考答案。"""
        matches = self.search(query, limit=limit)
        if not matches:
            return "没有检索到相关文档。"

        return self.build_answer_context(query=query, matches=matches)

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
            vector_score = self.embedding_service.cosine_similarity(
                query_vector,
                self.embedding_service.embed(chunk.content),
            )
            if vector_score <= 0:
                continue
            matches.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=self._rerank_score(query=query, chunk=chunk, vector_score=vector_score),
                )
            )
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:limit]

    def build_answer_context(self, *, query: str, matches: List[RetrievedChunk]) -> str:
        """
        生成适合直接交给 Agent 或模型继续使用的结构化上下文。

        修改说明：相比之前只返回“原始片段列表”，
        这里会额外给出“参考结论 + 证据片段”，让后续答案合成更稳定。
        """
        builder = ContextBuilder()
        builder.add_runtime_rules(
            [
                f"当前检索问题：{query}",
                "优先基于证据片段作答；若证据不足，不要补造事实。",
            ]
        )

        synthesized_lines = ["参考结论："]
        for index, item in enumerate(matches, start=1):
            synthesized_lines.append(
                f"{index}. {self._summarize_chunk(item.chunk.content)}"
            )
        builder.add_notes(
            "检索结论",
            "\n".join(synthesized_lines),
            priority=70,
            source="rag_summary",
        )

        evidence_lines = []
        for index, item in enumerate(matches, start=1):
            evidence_lines.append(
                f"{index}. 来源: {item.chunk.source} | 综合分数: {item.score:.4f}\n{item.chunk.content}"
            )
        builder.add_retrieval("\n\n".join(evidence_lines))
        return builder.build().render()

    def _format_matches(self, matches: List[RetrievedChunk]) -> str:
        if not matches:
            return "没有检索到相关文档。"
        return "\n\n".join(
            f"来源: {item.chunk.source} | 分数: {item.score:.4f}\n{item.chunk.content}"
            for item in matches
        )

    def _rerank_score(self, *, query: str, chunk: DocumentChunk, vector_score: float) -> float:
        """把向量相似度和词项覆盖率合并成一个更稳定的排序分数。"""
        query_tokens = self._extract_query_tokens(query)
        if not query_tokens:
            return vector_score
        content = chunk.content.lower()
        overlap = sum(token in content for token in query_tokens) / len(query_tokens)
        starts_with_bonus = 0.08 if any(chunk.content.startswith(token) for token in query_tokens) else 0.0
        return (vector_score * 0.75) + (overlap * 0.25) + starts_with_bonus

    def _extract_query_tokens(self, query: str) -> List[str]:
        tokens = [
            token.lower().strip()
            for token in self._QUERY_TOKEN_PATTERN.findall(query)
            if token.strip()
        ]
        deduped: List[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
        return deduped[:12]

    @staticmethod
    def _summarize_chunk(content: str, limit: int = 80) -> str:
        compact = " ".join(content.split())
        if len(compact) <= limit:
            return compact
        return f"{compact[:limit]}..."

    @staticmethod
    def _build_chunk_record_id(chunk: DocumentChunk) -> str:
        """
        为文档切片生成稳定的存储 id。

        修改说明：RAG 原来直接把 `chunk_id` 当成向量库主键，
        但云端 Qdrant 只接受 uint / UUID，所以这里统一映射成稳定 UUID，
        同时把原始 `chunk_id` 继续放在 payload 里，便于后续追踪与调试。
        """
        stable_key = f"{chunk.source}:{chunk.chunk_id}"
        return str(uuid5(NAMESPACE_URL, stable_key))
