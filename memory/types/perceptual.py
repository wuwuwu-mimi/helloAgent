from __future__ import annotations

from typing import List

from memory.base import BaseMemory, MemoryItem


class PerceptualMemory(BaseMemory):
    """
    感知记忆占位实现。

    当前项目还没有多模态输入链路，这里先把目录和接口留好，
    后面如果接图片 / 音频理解时可以直接往这个文件里扩展。
    """

    def add(self, item: MemoryItem) -> None:
        del item

    def recent(self, session_id: str, limit: int = 10) -> List[MemoryItem]:
        del session_id, limit
        return []

    def search(self, session_id: str, query: str, limit: int = 10) -> List[MemoryItem]:
        del session_id, query, limit
        return []

    def clear(self, session_id: str) -> None:
        del session_id
