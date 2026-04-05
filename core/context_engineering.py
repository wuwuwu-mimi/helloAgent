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
        ordered = sorted(
            self.sections,
            key=lambda item: (item.priority, item.title),
            reverse=True,
        )
        deduped: List[ContextSection] = []
        seen: set[tuple[str, str]] = set()
        for section in ordered:
            signature = (section.title.strip(), section.content.strip())
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(section)
        return deduped

    def render(
        self,
        *,
        max_chars: int | None = None,
        max_sections: int | None = None,
        section_max_chars: int | None = None,
    ) -> str:
        ordered = self.ordered_sections()
        if not ordered:
            return ""

        rendered_sections: List[str] = []
        remaining_chars = max_chars if max_chars and max_chars > 0 else None
        section_limit = max_sections if max_sections and max_sections > 0 else None
        per_section_limit = section_max_chars if section_max_chars and section_max_chars > 0 else None

        for section in ordered:
            if section_limit is not None and len(rendered_sections) >= section_limit:
                break

            rendered = self._render_section(section, per_section_limit)
            if not rendered:
                continue

            if remaining_chars is None:
                rendered_sections.append(rendered)
                continue

            separator_length = 2 if rendered_sections else 0
            if remaining_chars <= separator_length:
                break
            allowed_length = remaining_chars - separator_length
            if len(rendered) > allowed_length:
                rendered = self._clip_text(rendered, allowed_length)
                if not rendered:
                    break
            rendered_sections.append(rendered)
            remaining_chars -= len(rendered) + separator_length

        return "\n\n".join(rendered_sections)

    @classmethod
    def _render_section(cls, section: ContextSection, section_max_chars: int | None) -> str:
        content = cls._clip_text(section.content.strip(), section_max_chars)
        if not content:
            return ""
        return f"[{section.title}]\n{content}"

    @staticmethod
    def _clip_text(text: str, limit: int | None) -> str:
        if limit is None or limit <= 0 or len(text) <= limit:
            return text
        if limit <= 1:
            return text[:limit]
        return f"{text[: limit - 1].rstrip()}…"


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
