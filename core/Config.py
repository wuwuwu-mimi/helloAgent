from __future__ import annotations

"""集中管理 agent 运行时配置。"""

import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


def _first_env(*names: str) -> str:
    """
    按顺序读取环境变量，返回第一个非空值。

    这样写的目的是把“环境变量优先级”集中在一处，
    避免散落在各个模块里时越来越难维护。
    """

    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return ""


def _read_bool(default: bool, *names: str) -> bool:
    raw = _first_env(*names)
    if not raw:
        return default

    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"无法将环境变量解析为 bool: {raw}")


def _read_float(default: Optional[float], *names: str) -> Optional[float]:
    raw = _first_env(*names)
    if not raw:
        return default
    return float(raw)


def _read_int(default: Optional[int], *names: str) -> Optional[int]:
    raw = _first_env(*names)
    if not raw:
        return default
    return int(raw)


class Config(BaseModel):
    """
    Agent 的统一配置对象。

    这个类只负责“读配置”和“整理配置”，不负责真正调用模型。
    这样 `Config -> Message -> LLM Client` 的职责边界会清楚很多。
    """

    # LLM 默认配置
    default_model: str = "deepseek-chat"
    default_provider: str = "deepseek"
    timeout: float = 60.0
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

    # 运行时配置
    debug: bool = False
    log_level: str = "INFO"
    max_history_length: int = 100
    context_max_chars: int = 3200
    context_max_sections: int = 6
    context_section_max_chars: int = 1200
    auto_rag_context: bool = True
    auto_rag_context_limit: int = 3
    tool_context_observation_limit: int = 4

    @classmethod
    def from_env(cls) -> "Config":
        """
        从环境变量创建配置。

        支持两类命名：
        1. 更通用的 `DEFAULT_MODEL`、`DEFAULT_PROVIDER`
        2. 已经在项目里使用的 `LLM_MODEL_ID`、`LLM_PROVIDER`
        """

        defaults = cls()
        return cls(
            default_model=_first_env("DEFAULT_MODEL", "LLM_MODEL_ID") or defaults.default_model,
            default_provider=_first_env("DEFAULT_PROVIDER", "LLM_PROVIDER") or defaults.default_provider,
            timeout=_read_float(defaults.timeout, "TIMEOUT", "LLM_TIMEOUT"),
            temperature=_read_float(defaults.temperature, "TEMPERATURE", "LLM_TEMPERATURE"),
            max_tokens=_read_int(defaults.max_tokens, "MAX_TOKENS", "LLM_MAX_TOKENS"),
            debug=_read_bool(defaults.debug, "DEBUG"),
            log_level=_first_env("LOG_LEVEL") or defaults.log_level,
            max_history_length=_read_int(defaults.max_history_length, "MAX_HISTORY_LENGTH"),
            context_max_chars=_read_int(defaults.context_max_chars, "CONTEXT_MAX_CHARS"),
            context_max_sections=_read_int(defaults.context_max_sections, "CONTEXT_MAX_SECTIONS"),
            context_section_max_chars=_read_int(
                defaults.context_section_max_chars,
                "CONTEXT_SECTION_MAX_CHARS",
            ),
            auto_rag_context=_read_bool(defaults.auto_rag_context, "AUTO_RAG_CONTEXT"),
            auto_rag_context_limit=_read_int(
                defaults.auto_rag_context_limit,
                "AUTO_RAG_CONTEXT_LIMIT",
            ),
            tool_context_observation_limit=_read_int(
                defaults.tool_context_observation_limit,
                "TOOL_CONTEXT_OBSERVATION_LIMIT",
            ),
        )

    def llm_options(self) -> Dict[str, Any]:
        """
        生成传给 `OpenAICompatibleLLM` 的默认参数。

        这样外层 agent 初始化时不需要自己手写参数映射。
        """

        options: Dict[str, Any] = {
            "provider": self.default_provider,
            "model": self.default_model,
            "timeout": self.timeout,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            options["max_tokens"] = self.max_tokens
        return options

    def trimmed_history(self, messages: list[Any]) -> list[Any]:
        """
        根据配置裁剪历史消息。

        大多数手写 agent 最容易失控的地方，就是上下文历史一直增长。
        先放一个简单且可复用的裁剪函数，后面如果需要再换成更复杂的策略。
        """

        if self.max_history_length <= 0:
            return list(messages)
        return list(messages)[-self.max_history_length :]

    def to_dict(self) -> Dict[str, Any]:
        """导出为普通字典，方便调试和打印。"""

        return self.model_dump()
