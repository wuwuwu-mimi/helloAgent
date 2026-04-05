from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence


@dataclass
class ContextSection:
    """表示一段可拼装进 prompt 的上下文片段。"""

    title: str
    content: str
    priority: int = 0
    source: str = "runtime"

    def render(self) -> str:
        return f"[{self.title}]\n{self.content.strip()}"


@dataclass
class ContextPacket:
    """
    表示一次请求最终注入给模型的结构化上下文。

    修改说明：把“系统提示词 / 记忆 / 检索结果 / 运行约束”统一抽成 section，
    后面做上下文裁剪、优先级排序、不同来源合并时会更清晰。
    """

    sections: List[ContextSection] = field(default_factory=list)

    def add(
        self,
        title: str,
        content: str,
        *,
        priority: int = 0,
        source: str = "runtime",
    ) -> None:
        cleaned = content.strip()
        if not cleaned:
            return
        self.sections.append(
            ContextSection(
                title=title,
                content=cleaned,
                priority=priority,
                source=source,
            )
        )

    def extend(self, sections: Iterable[ContextSection]) -> None:
        for section in sections:
            self.add(
                section.title,
                section.content,
                priority=section.priority,
                source=section.source,
            )

    def ordered_sections(self) -> List[ContextSection]:
        return sorted(
            self.sections,
            key=lambda item: (item.priority, item.title),
            reverse=True,
        )

    def render(self) -> str:
        ordered = self.ordered_sections()
        if not ordered:
            return ""
        return "\n\n".join(section.render() for section in ordered)


class ContextBuilder:
    """一个轻量的上下文工程构建器。"""

    def __init__(self) -> None:
        self.packet = ContextPacket()

    def add_system_prompt(self, text: str) -> "ContextBuilder":
        self.packet.add("系统提示", text, priority=100, source="system")
        return self

    def add_runtime_rules(self, rules: Sequence[str]) -> "ContextBuilder":
        cleaned_rules = [rule.strip() for rule in rules if rule.strip()]
        if cleaned_rules:
            body = "\n".join(f"- {rule}" for rule in cleaned_rules)
            self.packet.add("运行规则", body, priority=90, source="runtime")
        return self

    def add_memory(self, text: str) -> "ContextBuilder":
        self.packet.add("相关记忆", text, priority=70, source="memory")
        return self

    def add_retrieval(self, text: str) -> "ContextBuilder":
        self.packet.add("检索上下文", text, priority=60, source="retrieval")
        return self

    def add_notes(self, title: str, text: str, *, priority: int = 50, source: str = "runtime") -> "ContextBuilder":
        self.packet.add(title, text, priority=priority, source=source)
        return self

    def build(self) -> ContextPacket:
        return self.packet
