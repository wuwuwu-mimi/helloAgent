from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Iterable, List

from memory.base import MemoryConfig


class BaseEmbeddingService(ABC):
    """统一的嵌入服务抽象，便于后续切换到真实 embedding provider。"""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """把文本编码成向量。"""

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        """批量编码文本，默认复用单条编码逻辑。"""
        return [self.embed(text) for text in texts]

    @staticmethod
    def cosine_similarity(left: List[float], right: List[float]) -> float:
        """计算两个向量的余弦相似度。"""
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)


class HashEmbeddingService(BaseEmbeddingService):
    """
    一个完全离线、零依赖的本地嵌入实现。

    修改说明：当前环境还没接入真实向量模型，
    这里先用“token + 概念标签 + 哈希投影”的方式提供最小可用语义检索能力，
    这样后续接 DashScope / 本地 embedding 模型时，只需要替换这一层。
    """

    _TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
    _CONCEPT_GROUPS = {
        "concept:preference": ("喜欢", "偏好", "爱喝", "爱吃", "习惯", "记住"),
        "concept:drink": ("饮品", "喝", "咖啡", "美式", "拿铁", "奶茶", "茶", "饮料"),
        "concept:food": ("食物", "吃", "早餐", "午饭", "晚饭", "面", "米饭"),
        "concept:time": ("时间", "几点", "日期", "今天", "明天", "昨天"),
        "concept:work": ("项目", "任务", "代码", "开发", "工作", "计划"),
        "concept:person": ("我叫", "名字", "联系人", "朋友", "同事", "老师"),
    }

    def __init__(self, dimensions: int = 96) -> None:
        self.dimensions = max(16, dimensions)

    def embed(self, text: str) -> List[float]:
        normalized = text.strip().lower()
        if not normalized:
            return [0.0] * self.dimensions

        vector = [0.0] * self.dimensions
        units = self._extract_units(normalized)
        for unit in units:
            digest = hashlib.md5(unit.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], byteorder="big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.5 if unit.startswith("concept:") else 1.0
            vector[index] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _extract_units(self, text: str) -> List[str]:
        tokens = self._TOKEN_PATTERN.findall(text)
        text_without_spaces = "".join(text.split())
        bigrams = [
            text_without_spaces[index : index + 2]
            for index in range(len(text_without_spaces) - 1)
        ]

        concept_tags: List[str] = []
        for concept, keywords in self._CONCEPT_GROUPS.items():
            if any(keyword in text for keyword in keywords):
                concept_tags.append(concept)

        # 修改说明：把原始 token、双字片段和概念标签一起编码，
        # 让这个离线 embedding 至少具备一点“同类表达可召回”的能力。
        return tokens + bigrams + concept_tags


class EmbeddingServiceFactory:
    """根据 MemoryConfig 构造当前使用的 embedding 服务。"""

    @staticmethod
    def create(config: MemoryConfig) -> BaseEmbeddingService:
        backend = (config.embedding_backend or "hash").strip().lower()
        if backend in {"hash", "local", "simple"}:
            return HashEmbeddingService(dimensions=config.embedding_dimensions)
        raise ValueError(f"暂不支持的 embedding backend: {config.embedding_backend}")


class SimpleEmbeddingService(HashEmbeddingService):
    """兼容旧名字，内部直接复用新的 HashEmbeddingService。"""
