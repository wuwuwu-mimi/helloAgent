from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import BaseModel


class DocumentChunk(BaseModel):
    """RAG 文档切片。"""

    chunk_id: str
    source: str
    content: str


class RetrievedChunk(BaseModel):
    """表示一次检索命中的文档切片及其相似度分数。"""

    chunk: DocumentChunk
    score: float


class DocumentProcessor:
    """一个最小可用的文档处理器，目前支持 UTF-8 文本和 Markdown。"""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 60) -> None:
        self.chunk_size = max(80, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))

    def load(self, path: str) -> List[DocumentChunk]:
        """按文件类型加载文档，目前统一按 UTF-8 文本处理。"""
        file_path = Path(path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"文档不存在: {file_path}")
        text = file_path.read_text(encoding="utf-8")
        return self.split_text(text=text, source=str(file_path))

    def load_text(self, path: str) -> List[DocumentChunk]:
        """兼容旧名字，内部转到 `load()`。"""
        return self.load(path)

    def split_text(self, *, text: str, source: str) -> List[DocumentChunk]:
        """把长文本切成多个重叠片段，便于后续做向量检索。"""
        normalized = text.strip()
        if not normalized:
            return []

        chunks: List[DocumentChunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for index, start in enumerate(range(0, len(normalized), step), start=1):
            chunk_text = normalized[start : start + self.chunk_size].strip()
            if not chunk_text:
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{Path(source).name}-chunk-{index}",
                    source=source,
                    content=chunk_text,
                )
            )
            if start + self.chunk_size >= len(normalized):
                break
        return chunks
