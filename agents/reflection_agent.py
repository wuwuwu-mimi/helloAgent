from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Tuple

from core import Config, HelloAgentsLLM, Message
from memory.manager import MemoryManager
from tools.builtin.toolRegistry import ToolRegistry

from .react_agent import ReactAgent

logger = logging.getLogger(__name__)

# 修改说明：草稿阶段继续复用 ReAct 风格，让模型先借助工具拿到基础信息。
REFLECTION_DRAFT_PROMPT = """你是一个会先行动、再反思的 AI 助手。
请先像 ReAct 一样完成“草稿答案”，必要时可以调用工具，但每一轮只能做一件事。
不要直接暴露完整推理过程，只需严格输出 Thought 和 Action。

可用工具：
{tools}

输出格式：
Thought: 简要说明你当前的判断，以及下一步准备做什么。
Action: 只能是下面两种格式之一：
- tool_name[tool_input]
- Finish[草稿答案]

规则：
1. 如果信息不足，就调用一个合适的工具。
2. 如果你已经可以先给出草稿答案，就输出 Finish[...].
3. 不要省略 Thought 或 Action。
4. 如果工具需要多个参数，请使用 JSON 对象字符串。

当前问题：
{question}

历史记录：
{history}
"""

# 修改说明：审查阶段增加“工具事实不可改写”约束，避免反思阶段无依据地推翻工具结果。
REFLECTION_REVIEW_PROMPT = """你是一个负责审查答案质量的评审助手。
请检查下面的答案是否正确、完整、清晰，并给出是否需要修订的决定。

用户问题：
{question}

当前答案：
{answer}

已确认的工具事实（不可擅自改写）：
{grounded_facts}

请严格按照下面格式输出：
Reflection: 用 1 到 3 句话总结答案的优点、问题和风险。
Decision: 只能填写 revise 或 finish
Suggestions:
- 给出 1 到 3 条具体修改建议；如果不需要修改，也请说明为什么可以直接结束。

额外约束：
1. 不要无依据地质疑或改写“已确认的工具事实”。
2. 只有在当前答案与工具事实明显矛盾时，才能指出事实错误。
"""

# 修改说明：修订阶段显式传入工具事实，防止模型在润色文案时篡改已经落地的观察结果。
REFLECTION_REVISION_PROMPT = """你是一个负责改写答案的 AI 助手。
请根据原问题、当前答案和审查意见，输出一个更完整、更清晰的最终答案。

已确认的工具事实（这些关键信息不允许被改写）：
{grounded_facts}

要求：
1. 直接输出修订后的答案。
2. 不要再输出 Thought / Action。
3. 要真正吸收审查建议，而不是简单重复原答案。
4. 如果原答案已经基本正确，请在保持正确性的基础上补足表达和细节。
5. 不要改写或纠正“已确认的工具事实”中的数值、时间、名称等关键信息。

用户问题：
{question}

当前答案：
{answer}

审查意见：
{reflection}
"""

NATIVE_REFLECTION_DRAFT_PROMPT = """你是一个会先行动、再反思的 AI 助手。
请先借助工具生成一个可靠的草稿答案。

要求：
1. 当信息不足时，直接调用工具。
2. 如果已经有足够信息，请直接给出草稿答案。
3. 回答要优先基于工具结果和显式上下文，不要编造未观察到的信息。

当前问题：
{question}
"""


class ReflectionAgent(ReactAgent):
    """先生成草稿、再做自我审查、最后按需修订的 Reflection Agent。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        max_reflections: int = 2,
        draft_prompt: Optional[str] = None,
        review_prompt: Optional[str] = None,
        revision_prompt: Optional[str] = None,
        memory_manager: Optional[MemoryManager] = None,
        session_id: Optional[str] = None,
    ) -> None:
        # 修改说明：草稿阶段直接复用 ReactAgent 的工具解析与执行能力，避免维护两套相似逻辑。
        super().__init__(
            name=name,
            llm=llm,
            tool_registry=tool_registry,
            system_prompt=system_prompt,
            config=config,
            max_steps=max_steps,
            custom_prompt=draft_prompt or REFLECTION_DRAFT_PROMPT,
            memory_manager=memory_manager,
            session_id=session_id,
        )
        self.max_reflections = max_reflections
        self.review_prompt = review_prompt or REFLECTION_REVIEW_PROMPT
        self.revision_prompt = revision_prompt or REFLECTION_REVISION_PROMPT
        self.last_grounded_observations: List[str] = []
        self.enable_native_tool_calling = True

    def run(self, input_text: str, stream: bool = False, **kwargs: Any) -> str:
        """
        执行 Reflection 主流程。

        流程：
        1. 先通过一个小型 ReAct 循环生成草稿答案。
        2. 再让模型扮演评审，对当前答案做反思。
        3. 如果评审认为需要修改，就根据意见重写答案。
        """
        del stream  # 当前版本要拿到完整文本后再解析，暂不支持流式中间控制。

        self._start_new_run(input_text)
        self.last_grounded_observations = []

        logger.info("[%s] 开始 Reflection 任务: %s", self.name, input_text)

        answer = self._build_draft(input_text, **kwargs)
        self.current_history.append(f"Draft Answer: {answer}")

        for round_index in range(1, self.max_reflections + 1):
            reflection_text, decision, suggestions = self._review_answer(
                question=input_text,
                answer=answer,
                **kwargs,
            )
            self.current_history.append(f"Reflection {round_index}: {reflection_text}")
            self.current_history.append(f"Decision {round_index}: {decision}")
            self.current_history.append(f"Suggestions {round_index}: {suggestions}")

            logger.info("[%s] 第 %s 次反思决定: %s", self.name, round_index, decision)

            if decision == "finish":
                self._remember_assistant_text(answer, metadata={"memory_stage": "reflection_finish"})
                logger.info("[%s] 反思结束，直接采用当前答案。", self.name)
                return answer

            answer = self._revise_answer(
                question=input_text,
                answer=answer,
                reflection=self._format_reflection_text(reflection_text, suggestions),
                **kwargs,
            )
            self.current_history.append(f"Revision {round_index}: {answer}")

        self._remember_assistant_text(answer, metadata={"memory_stage": "reflection_final"})
        logger.info("[%s] 已达到最大反思轮数，返回最后一次修订结果。", self.name)
        return answer

    def _build_draft(self, question: str, **kwargs: Any) -> str:
        """
        生成初始草稿答案。

        这里复用了 ReAct 的核心机制：
        模型先输出 Thought / Action，Agent 再负责执行工具并回填 Observation。
        """
        if self._should_use_native_tool_calling():
            return self._build_draft_with_native_tool_calling(question, **kwargs)

        local_history: List[str] = []
        grounded_observations: List[str] = []

        for step in range(1, self.max_steps + 1):
            prompt = self.prompt_template.format(
                tools=self.tool_registry.describe_tools(),
                question=question,
                history=self._render_history(local_history),
            )
            logger.debug("[%s] 草稿阶段 step %s / %s", self.name, step, self.max_steps)
            reply = self._request_text(prompt, **kwargs)
            logger.debug("[%s] 草稿阶段模型输出: %s", self.name, self._preview(reply))

            if not reply:
                local_history.append("Draft Observation: LLM returned an empty response.")
                continue

            thought, action_text = self.parse_react_response(reply)
            if thought:
                local_history.append(f"Draft Thought: {thought}")

            if not action_text:
                local_history.append("Draft Observation: Invalid response format: missing Action line.")
                continue

            local_history.append(f"Draft Action: {action_text}")
            action_type, action_input = self.parse_action(action_text)

            if action_type == "finish":
                draft_answer = action_input or ""
                local_history.append(f"Draft Finish: {draft_answer}")
                self.last_grounded_observations = grounded_observations
                self.current_history.extend(local_history)
                return draft_answer

            observation = self._handle_action(action_type, action_input)
            if action_type not in {"unknown"}:
                grounded_observations.append(observation)
            local_history.append(f"Draft Observation: {observation}")

        fallback = f"Reached max steps ({self.max_steps}) without producing a draft answer."
        local_history.append(f"Draft Fallback: {fallback}")
        self.last_grounded_observations = grounded_observations
        self.current_history.extend(local_history)
        logger.warning("[%s] %s", self.name, fallback)
        return fallback

    def _build_draft_with_native_tool_calling(self, question: str, **kwargs: Any) -> str:
        """使用原生 tool calling 生成草稿答案。"""
        local_history: List[str] = []
        grounded_observations: List[str] = []
        context_packet = self._build_context_packet()
        rendered_context = context_packet.render(
            max_chars=self.config.context_max_chars,
            max_sections=self.config.context_max_sections,
            section_max_chars=self.config.context_section_max_chars,
        )
        messages: List[Message] = []
        if rendered_context:
            messages.append(
                Message.system(rendered_context, metadata={"source": "context_engineering"})
            )
        messages.append(Message.user(NATIVE_REFLECTION_DRAFT_PROMPT.format(question=question)))

        for step in range(1, self.max_steps + 1):
            logger.debug("[%s] 原生 tool calling 草稿阶段 step %s / %s", self.name, step, self.max_steps)
            result = self.llm.chat(
                messages=messages,
                stream=False,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                tools=self.tool_registry.get_available_tools(),
                tool_choice="auto",
                **kwargs,
            )
            assistant_message = self._build_assistant_message_from_result(
                result,
                round_index=step,
                history_target=local_history,
                append_current_history=False,
            )
            messages.append(assistant_message)

            if result.tool_calls:
                for tool_call in result.tool_calls:
                    observation = self._execute_native_tool_call(
                        tool_call,
                        history_target=local_history,
                        append_current_history=False,
                    )
                    if tool_call.get("function", {}).get("name"):
                        grounded_observations.append(observation)
                    messages.append(
                        Message.tool(
                            observation,
                            tool_call_id=tool_call.get("id") or f"draft_tool_call_{step}",
                            name=tool_call.get("function", {}).get("name"),
                            metadata={"native_tool_calling": True, "stage": "draft"},
                        )
                    )
                continue

            draft_answer = (result.text or "").strip()
            if draft_answer:
                local_history.append(f"Draft Finish: {draft_answer}")
                self.last_grounded_observations = grounded_observations
                self.current_history.extend(local_history)
                return draft_answer

            local_history.append("Draft Observation: Native tool calling returned an empty response.")

        fallback = f"Reached max steps ({self.max_steps}) without producing a draft answer."
        local_history.append(f"Draft Fallback: {fallback}")
        self.last_grounded_observations = grounded_observations
        self.current_history.extend(local_history)
        logger.warning("[%s] %s", self.name, fallback)
        return fallback

    def _review_answer(self, question: str, answer: str, **kwargs: Any) -> Tuple[str, str, str]:
        """
        审查当前答案，并返回反思结果。

        返回值：
        - reflection_text: 对答案优缺点的总结
        - decision: `revise` 或 `finish`
        - suggestions: 具体修改建议
        """
        prompt = self.review_prompt.format(
            question=question,
            answer=answer,
            grounded_facts=self._render_grounded_facts(),
        )
        raw_review = self._request_text(prompt, **kwargs)
        logger.debug("[%s] 原始反思输出: %s", self.name, self._preview(raw_review))
        return self._parse_review(raw_review)

    def _revise_answer(self, question: str, answer: str, reflection: str, **kwargs: Any) -> str:
        """根据反思意见重写答案；如果模型返回空文本，就回退到原答案。"""
        prompt = self.revision_prompt.format(
            question=question,
            answer=answer,
            reflection=reflection,
            grounded_facts=self._render_grounded_facts(),
        )
        revised_answer = self._request_text(prompt, **kwargs)
        logger.debug("[%s] 修订后的答案: %s", self.name, self._preview(revised_answer))

        if revised_answer and self._drops_grounded_facts(answer, revised_answer):
            logger.warning("[%s] 修订稿丢失了工具事实，回退到上一版答案。", self.name)
            return answer

        return revised_answer or answer

    def _parse_review(self, text: str) -> Tuple[str, str, str]:
        """
        解析评审输出。

        目标格式：
        Reflection: ...
        Decision: revise 或 finish
        Suggestions:
        - ...
        """
        reflection_match = re.search(r"Reflection:\s*(.*?)(?=\nDecision:|$)", text, re.DOTALL | re.IGNORECASE)
        decision_match = re.search(r"Decision:\s*(.*?)(?=\nSuggestions:|$)", text, re.DOTALL | re.IGNORECASE)
        suggestions_match = re.search(r"Suggestions:\s*(.*)$", text, re.DOTALL | re.IGNORECASE)

        reflection_text = reflection_match.group(1).strip() if reflection_match else text.strip()
        raw_decision = decision_match.group(1).strip() if decision_match else "revise"
        suggestions = suggestions_match.group(1).strip() if suggestions_match else "- 请补充更明确的修订建议。"

        decision = self._normalize_decision(raw_decision)
        return reflection_text, decision, suggestions

    def _render_grounded_facts(self) -> str:
        """把草稿阶段的工具观察结果整理成可直接写入 Prompt 的文本。"""
        if not self.last_grounded_observations:
            return "- 暂无已确认的工具事实。"
        return "\n".join(f"- {item}" for item in self.last_grounded_observations)

    def _drops_grounded_facts(self, current_answer: str, revised_answer: str) -> bool:
        """如果修订后的答案丢掉了当前答案中已引用的工具事实，则视为违规。"""
        for item in self.last_grounded_observations:
            if item and item in current_answer and item not in revised_answer:
                return True
        return False

    @staticmethod
    def _normalize_decision(raw_decision: str) -> str:
        """兼容中英文决策词，最终只返回 `finish` 或 `revise`。"""
        normalized = raw_decision.strip().lower()
        finish_aliases = {"finish", "done", "complete", "completed", "完成", "结束", "通过"}
        return "finish" if normalized in finish_aliases else "revise"

    @staticmethod
    def _format_reflection_text(reflection_text: str, suggestions: str) -> str:
        """把反思摘要和修改建议拼成统一文本，便于修订阶段直接引用。"""
        return f"Reflection: {reflection_text}\nSuggestions:\n{suggestions}"
