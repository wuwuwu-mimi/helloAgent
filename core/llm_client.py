from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from .Config import Config
from .message import ChatMessageLike, normalize_messages

load_dotenv()

TextCallback = Callable[[str], None]


@dataclass(frozen=True)
class ProviderSpec:
    """
    描述一个 OpenAI-compatible provider 的默认配置与环境变量约定。
    """

    name: str
    default_base_url: str
    api_key_envs: Tuple[str, ...]
    base_url_envs: Tuple[str, ...]
    model_envs: Tuple[str, ...]
    requires_api_key: bool = True


@dataclass(frozen=True)
class LLMConfig:
    """
    运行时最终生效的配置。

    单独保留这个对象，有两个好处：
    1. 调试时可以直接打印最终解析结果
    2. 外层 agent 想记录“本次到底用了什么模型配置”会很方便
    """

    provider: str
    model: str
    api_key: str
    base_url: str
    timeout: float
    temperature: Optional[float]
    max_tokens: Optional[int]


@dataclass
class ChatResult:
    """
    统一的聊天结果。

    - text: 最终文本输出
    - reasoning: reasoning 字段的兼容提取结果
    - tool_calls: assistant 返回的工具调用
    - finish_reason: 模型停止原因
    - raw: 非流式时保留原始响应，便于调试
    """

    text: str = ""
    reasoning: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    raw: Any = None


PROVIDER_ALIASES: Dict[str, str] = {
    "dashscope": "qwen",
    "tongyi": "qwen",
    "kimi": "moonshot",
    "glm": "zhipu",
    "bigmodel": "zhipu",
    "ark": "doubao",
    "douban": "doubao",
}


PROVIDER_SPECS: Dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        name="openai",
        default_base_url="https://api.openai.com/v1",
        api_key_envs=("OPENAI_API_KEY",),
        base_url_envs=("OPENAI_BASE_URL",),
        model_envs=("OPENAI_MODEL",),
    ),
    "deepseek": ProviderSpec(
        name="deepseek",
        default_base_url="https://api.deepseek.com/v1",
        api_key_envs=("DEEPSEEK_API_KEY",),
        base_url_envs=("DEEPSEEK_BASE_URL",),
        model_envs=("DEEPSEEK_MODEL",),
    ),
    "qwen": ProviderSpec(
        name="qwen",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_envs=("QWEN_API_KEY", "DASHSCOPE_API_KEY"),
        base_url_envs=("QWEN_BASE_URL", "DASHSCOPE_BASE_URL"),
        model_envs=("QWEN_MODEL", "DASHSCOPE_MODEL"),
    ),
    "zhipu": ProviderSpec(
        name="zhipu",
        default_base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_envs=("ZHIPU_API_KEY",),
        base_url_envs=("ZHIPU_BASE_URL",),
        model_envs=("ZHIPU_MODEL",),
    ),
    "moonshot": ProviderSpec(
        name="moonshot",
        default_base_url="https://api.moonshot.cn/v1",
        api_key_envs=("MOONSHOT_API_KEY",),
        base_url_envs=("MOONSHOT_BASE_URL",),
        model_envs=("MOONSHOT_MODEL",),
    ),
    "doubao": ProviderSpec(
        name="doubao",
        default_base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key_envs=("DOUBAO_API_KEY", "ARK_API_KEY"),
        base_url_envs=("DOUBAO_BASE_URL", "ARK_BASE_URL"),
        model_envs=("DOUBAO_MODEL", "ARK_MODEL"),
    ),
    "minimax": ProviderSpec(
        name="minimax",
        default_base_url="https://api.minimax.io/v1",
        api_key_envs=("MINIMAX_API_KEY",),
        base_url_envs=("MINIMAX_BASE_URL",),
        model_envs=("MINIMAX_MODEL",),
    ),
    "ollama": ProviderSpec(
        name="ollama",
        default_base_url="http://127.0.0.1:11434/v1",
        api_key_envs=("OLLAMA_API_KEY",),
        base_url_envs=("OLLAMA_BASE_URL",),
        model_envs=("OLLAMA_MODEL",),
        requires_api_key=False,
    ),
    "vllm": ProviderSpec(
        name="vllm",
        default_base_url="http://127.0.0.1:8000/v1",
        api_key_envs=("VLLM_API_KEY",),
        base_url_envs=("VLLM_BASE_URL",),
        model_envs=("VLLM_MODEL",),
        requires_api_key=False,
    ),
    "local": ProviderSpec(
        name="local",
        default_base_url="http://127.0.0.1:11434/v1",
        api_key_envs=("LOCAL_API_KEY",),
        base_url_envs=("LOCAL_BASE_URL",),
        model_envs=("LOCAL_MODEL",),
        requires_api_key=False,
    ),
    "custom": ProviderSpec(
        name="custom",
        default_base_url="",
        api_key_envs=("CUSTOM_API_KEY",),
        base_url_envs=("CUSTOM_BASE_URL",),
        model_envs=("CUSTOM_MODEL",),
    ),
}


class HelloAgentsLLM:
    """
    一个不依赖现成 agent 框架的基础 LLM 客户端。

    设计目标：
    1. 配置解析要稳定，不要 provider 串线
    2. 能同时接受 `Message` 和普通 dict
    3. 兼容流式文本、reasoning 和 tool_calls
    """

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        provider_name, provider_source = self._resolve_provider(
            provider=provider,
            base_url=base_url,
        )
        config = self._resolve_config(
            provider=provider_name,
            provider_source=provider_source,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.config = config
        self.provider = config.provider
        self.model = config.model
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.timeout = config.timeout
        self.default_temperature = config.temperature
        self.default_max_tokens = config.max_tokens

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @classmethod
    def from_config(cls, config: Config, **overrides: Any) -> "HelloAgentsLLM":
        """
        从 `Config` 构造 LLM 客户端。

        这是把 `Config.py` 和 `llm_client.py` 串起来的最直接方式。
        外层 agent 一般只需要维护一个 Config 实例即可。
        """

        options = config.llm_options()
        for key, value in overrides.items():
            if value is not None:
                options[key] = value
        return cls(**options)

    @classmethod
    def available_providers(cls) -> List[str]:
        return sorted(PROVIDER_SPECS.keys())

    def think(
        self,
        messages: Sequence[ChatMessageLike],
        *,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        on_text: Optional[TextCallback] = None,
        on_reasoning: Optional[TextCallback] = None,
        **kwargs: Any,
    ) -> str:


        result = self.chat(
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            on_text=on_text,
            on_reasoning=on_reasoning,
            **kwargs,
        )
        return result.text

    def chat(
        self,
        messages: Sequence[ChatMessageLike],
        *,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[Any] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        on_text: Optional[TextCallback] = None,
        on_reasoning: Optional[TextCallback] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        统一聊天入口。

        保留 `**kwargs` 是为了兼容不同 provider 的扩展参数，
        例如部分模型会有自己的采样参数或推理开关。
        """

        request = self._build_request(
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            stop=stop,
            extra_body=extra_body,
            **kwargs,
        )

        response = self.client.chat.completions.create(**request)
        if stream:
            return self._consume_stream(
                response=response,
                on_text=on_text,
                on_reasoning=on_reasoning,
            )
        return self._consume_response(response)

    def _build_request(
        self,
        *,
        messages: Sequence[ChatMessageLike],
        stream: bool,
        temperature: Optional[float],
        max_tokens: Optional[int],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Any],
        response_format: Optional[Dict[str, Any]],
        stop: Optional[Any],
        extra_body: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        只在值存在时才把参数放进请求体，避免把无意义的 null 传给 provider。
        """

        request: Dict[str, Any] = {
            "model": self.model,
            "messages": normalize_messages(messages),
            "stream": stream,
        }

        resolved_temperature = (
            self.default_temperature if temperature is None else temperature
        )
        resolved_max_tokens = self.default_max_tokens if max_tokens is None else max_tokens

        if resolved_temperature is not None:
            request["temperature"] = resolved_temperature
        if resolved_max_tokens is not None:
            request["max_tokens"] = resolved_max_tokens
        if tools is not None:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        if response_format is not None:
            request["response_format"] = response_format
        if stop is not None:
            request["stop"] = stop
        if extra_body is not None:
            request["extra_body"] = extra_body

        request.update(kwargs)
        return request

    def _consume_response(self, response: Any) -> ChatResult:
        """
        处理非流式响应。
        """

        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("模型返回了空响应，未找到 choices。")

        choice = choices[0]
        message = choice.message
        return ChatResult(
            text=self._content_to_text(getattr(message, "content", None)),
            reasoning=self._extract_reasoning(message),
            tool_calls=[
                self._serialize_tool_call(tool_call)
                for tool_call in getattr(message, "tool_calls", None) or []
            ],
            finish_reason=getattr(choice, "finish_reason", None),
            raw=response,
        )

    def _consume_stream(
        self,
        *,
        response: Iterable[Any],
        on_text: Optional[TextCallback],
        on_reasoning: Optional[TextCallback],
    ) -> ChatResult:
        """
        消费流式响应，并把 provider 差异尽量抹平。

        部分 provider 返回“增量文本”，部分返回“截至当前的完整文本”，
        这里统一做一次去重累积，避免外层 agent 收到重复内容。
        """

        text_buffer = ""
        reasoning_buffer = ""
        finish_reason: Optional[str] = None
        tool_calls: Dict[int, Dict[str, Any]] = {}

        for chunk in response:
            if not getattr(chunk, "choices", None):
                continue

            choice = chunk.choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            content_piece = self._content_to_text(getattr(delta, "content", None))
            text_buffer, new_text = self._merge_stream_value(text_buffer, content_piece)
            if new_text and on_text is not None:
                on_text(new_text)

            reasoning_piece = self._extract_reasoning(delta)
            reasoning_buffer, new_reasoning = self._merge_stream_value(
                reasoning_buffer,
                reasoning_piece,
            )
            if new_reasoning and on_reasoning is not None:
                on_reasoning(new_reasoning)

            for tool_call in getattr(delta, "tool_calls", None) or []:
                index = getattr(tool_call, "index", 0)
                merged = tool_calls.setdefault(
                    index,
                    {
                        "id": None,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )

                if getattr(tool_call, "id", None):
                    merged["id"] = tool_call.id
                if getattr(tool_call, "type", None):
                    merged["type"] = tool_call.type

                function = getattr(tool_call, "function", None)
                if function is None:
                    continue

                current_name = getattr(function, "name", None)
                if current_name:
                    merged["function"]["name"], _ = self._merge_stream_value(
                        merged["function"]["name"],
                        current_name,
                    )

                current_arguments = getattr(function, "arguments", None)
                if current_arguments:
                    merged["function"]["arguments"], _ = self._merge_stream_value(
                        merged["function"]["arguments"],
                        current_arguments,
                    )

        ordered_tool_calls = [tool_calls[index] for index in sorted(tool_calls)]
        return ChatResult(
            text=text_buffer,
            reasoning=reasoning_buffer,
            tool_calls=ordered_tool_calls,
            finish_reason=finish_reason,
            raw=None,
        )

    def _resolve_provider(
        self,
        *,
        provider: Optional[str],
        base_url: Optional[str],
    ) -> Tuple[str, str]:
        """
        解析 provider，同时返回来源。

        额外返回 `provider_source` 是为了后面判断：
        当前这个 provider 到底是“用户明确指定的”，还是“从通用环境变量猜出来的”。
        只有把这个来源信息带上，才能彻底避免 provider 串线。
        """

        if self._clean(provider):
            return self._normalize_provider(provider), "explicit_provider"

        if self._clean(base_url):
            return self._infer_provider_from_base_url(base_url), "explicit_base_url"

        env_provider = self._clean(os.getenv("LLM_PROVIDER"))
        if env_provider:
            return self._normalize_provider(env_provider), "env_provider"

        detected_from_env = self._detect_provider_from_env()
        if detected_from_env:
            return detected_from_env, "provider_specific_env"

        generic_base_url = self._clean(os.getenv("LLM_BASE_URL"))
        if generic_base_url:
            return self._infer_provider_from_base_url(generic_base_url), "generic_base_url"

        return "openai", "default"

    def _resolve_config(
        self,
        *,
        provider: str,
        provider_source: str,
        model: Optional[str],
        base_url: Optional[str],
        api_key: Optional[str],
        timeout: float,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> LLMConfig:
        """
        统一处理配置优先级。

        核心原则：
        1. 显式参数永远优先
        2. provider 专属环境变量优先于通用环境变量
        3. 通用 `LLM_BASE_URL` 只在“真的走默认 provider 推断”时才介入

        第 3 条是解决 provider 串线的关键。
        """

        spec = PROVIDER_SPECS[provider]

        resolved_model = self._first_nonempty(
            model,
            *(os.getenv(name) for name in spec.model_envs),
            os.getenv("DEFAULT_MODEL"),
            os.getenv("LLM_MODEL_ID"),
        )
        if not resolved_model:
            model_env_names = list(spec.model_envs) + ["DEFAULT_MODEL", "LLM_MODEL_ID"]
            raise ValueError(
                f"provider={provider} 缺少模型名，请显式传入 model，或设置环境变量: {', '.join(model_env_names)}"
            )

        base_url_candidates: List[Optional[str]] = [
            base_url,
            *(os.getenv(name) for name in spec.base_url_envs),
        ]
        if provider == "custom" or provider_source == "generic_base_url":
            base_url_candidates.append(os.getenv("LLM_BASE_URL"))
        base_url_candidates.append(spec.default_base_url)

        resolved_base_url = self._first_nonempty(*base_url_candidates)
        if not resolved_base_url:
            base_env_names = list(spec.base_url_envs) + ["LLM_BASE_URL"]
            raise ValueError(
                f"provider={provider} 缺少 base_url，请显式传入 base_url，或设置环境变量: {', '.join(base_env_names)}"
            )

        api_key_candidates: List[Optional[str]] = [
            api_key,
            *(os.getenv(name) for name in spec.api_key_envs),
        ]
        if spec.requires_api_key:
            api_key_candidates.append(os.getenv("LLM_API_KEY"))

        resolved_api_key = self._first_nonempty(*api_key_candidates)
        if spec.requires_api_key and not resolved_api_key:
            key_env_names = list(spec.api_key_envs) + ["LLM_API_KEY"]
            raise ValueError(
                f"provider={provider} 缺少 api_key，请显式传入 api_key，或设置环境变量: {', '.join(key_env_names)}"
            )

        if not resolved_api_key:
            # 本地 OpenAI-compatible 服务常常不校验 key，但 OpenAI SDK 仍要求传一个值。
            resolved_api_key = "fake-key"

        return LLMConfig(
            provider=provider,
            model=resolved_model,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            timeout=float(timeout),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _detect_provider_from_env(self) -> Optional[str]:
        """
        只根据 provider 专属环境变量判断，不看通用 `LLM_*`。

        这样可以避免项目里只配置了一个全局 `LLM_BASE_URL`，
        却把所有 provider 都误导向同一个服务地址。
        """

        for name, spec in PROVIDER_SPECS.items():
            if name == "custom":
                continue
            for env_name in (*spec.api_key_envs, *spec.base_url_envs):
                if self._clean(os.getenv(env_name)):
                    return name
        return None

    def _normalize_provider(self, provider: str) -> str:
        normalized = self._clean(provider)
        if not normalized:
            raise ValueError("provider 不能为空。")

        normalized = normalized.lower()
        normalized = PROVIDER_ALIASES.get(normalized, normalized)
        if normalized not in PROVIDER_SPECS:
            supported = ", ".join(self.available_providers())
            raise ValueError(f"不支持的 provider: {provider}。当前支持: {supported}")
        return normalized

    def _infer_provider_from_base_url(self, base_url: str) -> str:
        """
        没有显式 provider 时，尽量从 URL 猜一个最合理的 provider。
        """

        value = (self._clean(base_url) or "").lower()
        if not value:
            return "openai"

        if "api.openai.com" in value:
            return "openai"
        if "api.deepseek.com" in value:
            return "deepseek"
        if "dashscope.aliyuncs.com" in value:
            return "qwen"
        if "open.bigmodel.cn" in value:
            return "zhipu"
        if "api.moonshot.cn" in value:
            return "moonshot"
        if "ark.cn-" in value or "volces.com" in value:
            return "doubao"
        if "api.minimax.io" in value:
            return "minimax"
        if "127.0.0.1" in value or "localhost" in value:
            if ":11434" in value:
                return "ollama"
            if ":8000" in value:
                return "vllm"
            return "local"

        return "custom"

    @staticmethod
    def _serialize_tool_call(tool_call: Any) -> Dict[str, Any]:
        function = getattr(tool_call, "function", None)
        return {
            "id": getattr(tool_call, "id", None),
            "type": getattr(tool_call, "type", "function"),
            "function": {
                "name": getattr(function, "name", "") if function else "",
                "arguments": getattr(function, "arguments", "") if function else "",
            },
        }

    @staticmethod
    def _extract_reasoning(obj: Any) -> str:
        """
        尽量兼容不同 provider 的 reasoning 字段命名。

        注意这里故意不把 `content_text` 视为 reasoning，
        因为很多兼容层会把它当正文文本，而不是思维链。
        """

        if obj is None:
            return ""

        direct_fields = (
            "reasoning_content",
            "reasoning",
            "reasoning_text",
        )
        for field_name in direct_fields:
            value = HelloAgentsLLM._read_value(obj, field_name)
            if isinstance(value, str) and value:
                return value

        details = HelloAgentsLLM._read_value(obj, "reasoning_details")
        if not details:
            return ""

        parts: List[str] = []
        for item in details:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(text)
                continue

            text = getattr(item, "text", None) or getattr(item, "content", None)
            if text:
                parts.append(text)

        return "".join(parts)

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """
        把 SDK 返回的 content 统一转成纯文本。
        """

        if content is None:
            return ""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if isinstance(text, str):
                        parts.append(text)
                    continue

                text = getattr(item, "text", None) or getattr(item, "content", None)
                if isinstance(text, str):
                    parts.append(text)

            return "".join(parts)

        return str(content)

    @staticmethod
    def _merge_stream_value(current: str, incoming: str) -> Tuple[str, str]:
        """
        兼容两类流式输出：
        1. chunk 只给新增内容
        2. chunk 每次都给截至当前的完整内容

        返回：
        - 新的累计值
        - 本次真正新增的部分
        """

        if not incoming:
            return current, ""
        if current and incoming.startswith(current):
            return incoming, incoming[len(current) :]
        return current + incoming, incoming

    @staticmethod
    def _first_nonempty(*values: Optional[str]) -> str:
        for value in values:
            cleaned = HelloAgentsLLM._clean(value)
            if cleaned:
                return cleaned
        return ""

    @staticmethod
    def _clean(value: Optional[str]) -> str:
        if value is None:
            return ""
        return value.strip()

    @staticmethod
    def _read_value(obj: Any, field_name: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(field_name)
        return getattr(obj, field_name, None)
