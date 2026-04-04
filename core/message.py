from __future__ import annotations

"""定义 agent 内部使用的消息结构。"""

from datetime import datetime
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, Field

MessageRole = Literal["system", "user", "assistant", "tool"]


class ToolFunction(BaseModel):
    """
    对齐 OpenAI-compatible 的函数调用结构。

    `arguments` 保持字符串，是为了和大多数模型接口的原始格式一致。
    真正执行工具前，再按需 `json.loads()` 会更稳。
    """

    name: str
    arguments: str = ""


class ToolCall(BaseModel):
    """assistant 发起的单次工具调用。"""

    id: Optional[str] = None
    type: str = "function"
    function: ToolFunction


class Message(BaseModel):
    """
    统一的消息对象。

    之所以不用裸 dict，是因为 agent 后面会涉及：
    - assistant 带 tool_calls
    - tool 结果回填
    - 调试日志和时间戳
    这些信息如果都塞在散乱的 dict 里，后面会越来越难维护。
    """

    role: MessageRole
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def system(cls, content: str, **kwargs: Any) -> "Message":
        return cls(role="system", content=content, **kwargs)

    @classmethod
    def user(cls, content: str, **kwargs: Any) -> "Message":
        return cls(role="user", content=content, **kwargs)

    @classmethod
    def assistant(
        cls,
        content: Optional[str] = None,
        *,
        tool_calls: Optional[List[ToolCall]] = None,
        **kwargs: Any,
    ) -> "Message":
        return cls(
            role="assistant",
            content=content,
            tool_calls=tool_calls or [],
            **kwargs,
        )

    @classmethod
    def tool(
        cls,
        content: str,
        *,
        tool_call_id: str,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> "Message":
        return cls(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=name,
            **kwargs,
        )

    def to_chat_message(self) -> Dict[str, Any]:
        """
        转成发给 OpenAI-compatible 接口的结构。

        这里保留 tool_calls / tool_call_id，避免在进入 LLM 层时丢失 agent 关键状态。
        """

        payload: Dict[str, Any] = {"role": self.role}

        if self.content is not None:
            payload["content"] = self.content
        elif self.role == "assistant" and self.tool_calls:
            payload["content"] = None
        else:
            payload["content"] = ""

        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [
                tool_call.model_dump(exclude_none=True) for tool_call in self.tool_calls
            ]

        return payload

    @classmethod
    def from_chat_message(cls, payload: Dict[str, Any]) -> "Message":
        """
        从 OpenAI-compatible 消息结构反向构造 `Message`。

        这个方法常用于把模型响应重新放回历史记录里。
        """

        raw_tool_calls = payload.get("tool_calls") or []
        tool_calls = [ToolCall.model_validate(item) for item in raw_tool_calls]

        return cls(
            role=payload["role"],
            content=payload.get("content"),
            name=payload.get("name"),
            tool_call_id=payload.get("tool_call_id"),
            tool_calls=tool_calls,
            metadata=payload.get("metadata") or {},
        )

    def short(self) -> str:
        """
        返回适合调试日志的简短表示。
        """

        preview = (self.content or "").replace("\n", " ").strip()
        if len(preview) > 60:
            preview = f"{preview[:57]}..."
        return f"{self.role}: {preview}"

    def __str__(self) -> str:
        return self.short()


ChatMessageLike = Union[Message, Dict[str, Any]]


def normalize_messages(messages: Sequence[ChatMessageLike]) -> List[Dict[str, Any]]:
    """
    把 `Message` 对象或普通 dict 统一转换成可发给模型的消息列表。
    """

    normalized: List[Dict[str, Any]] = []
    for message in messages:
        if isinstance(message, Message):
            normalized.append(message.to_chat_message())
            continue

        if isinstance(message, dict):
            normalized.append(dict(message))
            continue

        raise TypeError(f"不支持的消息类型: {type(message)!r}")

    return normalized


def trim_messages(messages: Iterable[Message], max_length: int) -> List[Message]:
    """
    额外提供一个纯消息层的裁剪函数，便于不依赖 `Config` 时单独使用。
    """

    items = list(messages)
    if max_length <= 0:
        return items
    return items[-max_length:]
