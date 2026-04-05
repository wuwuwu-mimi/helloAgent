from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import BaseModel


class DocumentChunk(BaseModel):
    """RAG 文档切片。"""

    source: str
    content: str


class DocumentProcessor:
    """一个最小可用的文档处理器，目前只支持纯文本文件。"""

    def load_text(self, path: str) -> List[DocumentChunk]:
        file_path = Path(path)
        text = file_path.read_text(encoding="utf-8")
        return [DocumentChunk(source=str(file_path), content=text)]
