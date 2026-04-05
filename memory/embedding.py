from __future__ import annotations

import hashlib
import json
import math
import re
from urllib import error, request
from abc import ABC, abstractmethod
from typing import Any, Iterable, List

from memory.base import MemoryConfig


class BaseEmbeddingService(ABC):
    """统一的嵌入服务抽象，便于后续切换到真实 embedding provider。"""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """把文本编码成向量。"""

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        """批量编码文本，默认复用单条编码逻辑。"""
        return [self.embed(text) for text in texts]

    def dimension_hint(self) -> int | None:
        """返回当前 embedding 维度提示；未知时返回 None。"""
        return None

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

    def dimension_hint(self) -> int:
        """离线哈希向量的维度固定可知。"""
        return self.dimensions

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


class OllamaEmbeddingService(BaseEmbeddingService):
    """
    基于本地 Ollama 的 embedding 服务。

    修改说明：当本机已经跑了 Ollama embedding 模型时，
    记忆检索和 RAG 就可以直接使用真实向量，而不再局限于离线哈希近似。
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        dimensions_hint: int | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("使用 Ollama embedding 时必须提供模型名。")
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.timeout_seconds = max(1.0, timeout_seconds)
        self._dimensions_hint = dimensions_hint if dimensions_hint and dimensions_hint > 0 else None

    def embed(self, text: str) -> List[float]:
        payload_text = text if text.strip() else " "
        vector = self._request_single_embedding(payload_text)
        if self._dimensions_hint is None:
            self._dimensions_hint = len(vector)
        return vector

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        normalized_texts = [text if str(text).strip() else " " for text in texts]
        if not normalized_texts:
            return []

        payload = {
            "model": self.model,
            "input": normalized_texts,
        }

        try:
            body = self._post_json("/api/embed", payload)
            embeddings = body.get("embeddings", [])
            vectors = [self._normalize_vector(item) for item in embeddings if item]
            if vectors:
                if self._dimensions_hint is None:
                    self._dimensions_hint = len(vectors[0])
                return vectors
        except RuntimeError:
            # 修改说明：有些 Ollama 版本只支持单条 `/api/embeddings`，
            # 这里降级回逐条请求，兼容本地不同版本。
            pass

        return [self.embed(text) for text in normalized_texts]

    def dimension_hint(self) -> int | None:
        return self._dimensions_hint

    def _request_single_embedding(self, text: str) -> List[float]:
        try:
            body = self._post_json("/api/embed", {"model": self.model, "input": text})
            embeddings = body.get("embeddings", [])
            if embeddings:
                return self._normalize_vector(embeddings[0])
        except RuntimeError:
            pass

        body = self._post_json("/api/embeddings", {"model": self.model, "prompt": text})
        embedding = body.get("embedding")
        if embedding:
            return self._normalize_vector(embedding)
        raise RuntimeError("Ollama embeddings 响应里没有找到向量数据。")

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        target_url = f"{self.base_url}{path}"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            target_url,
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                raw_text = response.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(
                f"Ollama embedding 请求失败，请确认本地服务已启动：{exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - 需要把本地服务异常转成更清晰的错误
            raise RuntimeError(f"Ollama embedding 调用失败：{exc}") from exc

        try:
            return json.loads(raw_text or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama embedding 返回了无法解析的 JSON：{raw_text}") from exc

    @staticmethod
    def _normalize_vector(values: Any) -> List[float]:
        if not isinstance(values, list):
            raise RuntimeError("embedding 返回格式不正确，期望为 number 列表。")
        return [float(item) for item in values]


class EmbeddingServiceFactory:
    """根据 MemoryConfig 构造当前使用的 embedding 服务。"""

    @staticmethod
    def create(config: MemoryConfig) -> BaseEmbeddingService:
        backend = (config.embedding_backend or "hash").strip().lower()
        if backend in {"hash", "local", "simple"}:
            return HashEmbeddingService(dimensions=config.embedding_dimensions)
        if backend in {"ollama", "ollama_local"}:
            return OllamaEmbeddingService(
                base_url=config.ollama_embedding_base_url,
                model=config.ollama_embedding_model,
                timeout_seconds=config.ollama_embedding_timeout_seconds,
                dimensions_hint=config.embedding_dimensions,
            )
        raise ValueError(f"暂不支持的 embedding backend: {config.embedding_backend}")


class SimpleEmbeddingService(HashEmbeddingService):
    """兼容旧名字，内部直接复用新的 HashEmbeddingService。"""
