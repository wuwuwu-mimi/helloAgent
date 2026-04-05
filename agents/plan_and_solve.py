from __future__ import annotations

import ast
import logging
import re
from typing import Any, Dict, List, Optional

from core import Config, HelloAgentsLLM
from memory.manager import MemoryManager
from tools.builtin.toolRegistry import ToolRegistry

from .react_agent import ReactAgent

logger = logging.getLogger(__name__)

# 第一步：只负责拆计划，不做求解。
PLAN_PROMPT = """你是一名擅长拆解复杂任务的规划专家。
请把用户问题拆成一个按顺序执行的步骤列表。

要求：
1. 计划尽量简洁，通常 2 到 6 步即可。
2. 每一步都应该是一个清晰、可执行的子任务。
3. 不要输出解释，不要输出 markdown，只输出 Python 列表。

问题：
{question}

请严格按照下面格式输出：
["步骤1", "步骤2", "步骤3"]
"""

# 第二步：按计划逐步求解，每一轮只允许“调用一个工具”或“完成当前步骤”。
STEP_PROMPT = """你是一名严格执行计划的 AI 助手。
你需要根据原始问题、完整计划和已有步骤结果，专注解决“当前步骤”。

可用工具：
{tools}

输出格式：
Thought: 简要说明当前判断。
Action: 只能是下面两种格式之一：
- tool_name[tool_input]
- Finish[当前步骤的结果]

规则：
1. 你当前只解决“当前步骤”，不要直接跳到最终答案。
2. 如果需要更多信息，就调用一个工具。
3. 如果当前步骤已经可以完成，就输出 Finish[...].
4. 一次只允许一个 Action。
5. 如果工具需要多个参数，请使用 JSON 对象字符串。

原始问题：
{question}

完整计划：
{plan}

已完成步骤：
{completed_steps}

当前步骤：
{current_step}

当前步骤内部历史：
{step_history}
"""

# 第三步：把所有子步骤结果合成为对用户的最终回答。
FINAL_PROMPT = """你是一名总结专家。
请根据原始问题、完整计划和各步骤结果，给出最终回答。

要求：
1. 直接回答用户问题。
2. 如果步骤结果里已经包含关键信息，就整合后自然输出。
3. 不要暴露内部推理过程。

原始问题：
{question}

完整计划：
{plan}

步骤结果：
{step_results}
"""


class PlanAndSolveAgent(ReactAgent):
    """先规划、再逐步求解的 Plan-and-Solve Agent。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        max_step_rounds: int = 4,
        plan_prompt: Optional[str] = None,
        step_prompt: Optional[str] = None,
        final_prompt: Optional[str] = None,
        memory_manager: Optional[MemoryManager] = None,
        session_id: Optional[str] = None,
    ) -> None:
        # 这里复用 ReactAgent 的工具执行与解析能力，避免维护两套同类逻辑。
        super().__init__(
            name=name,
            llm=llm,
            tool_registry=tool_registry,
            system_prompt=system_prompt,
            config=config,
            max_steps=max_steps,
            custom_prompt=step_prompt or STEP_PROMPT,
            memory_manager=memory_manager,
            session_id=session_id,
        )
        self.max_step_rounds = max_step_rounds
        self.plan_prompt = plan_prompt or PLAN_PROMPT
        self.final_prompt = final_prompt or FINAL_PROMPT
        self.last_plan: List[str] = []
        self.last_step_results: List[Dict[str, str]] = []

    def run(self, input_text: str, stream: bool = False, **kwargs: Any) -> str:
        """
        执行 Plan-and-Solve 主流程。

        流程：
        1. 先让模型生成步骤计划
        2. 按步骤逐个求解，每个步骤内部允许进行工具调用
        3. 汇总所有步骤结果，生成最终回答
        """
        del stream  # 当前实现统一使用非流式结果，方便做结构化解析。

        self._start_new_run(input_text)
        self.last_plan = []
        self.last_step_results = []

        logger.info("[%s] 开始执行 Plan-and-Solve: %s", self.name, input_text)

        plan = self._generate_plan(input_text, **kwargs)
        self.last_plan = plan
        self.current_history.append(f"Plan: {plan}")

        if not plan:
            fallback = "Planning failed: could not generate a valid plan."
            self._remember_assistant_text(fallback, metadata={"memory_stage": "plan_failed"})
            logger.warning("[%s] %s", self.name, fallback)
            return fallback

        for index, step in enumerate(plan, start=1):
            logger.info("[%s] 开始执行计划步骤 %s: %s", self.name, index, step)
            step_result = self._solve_step(
                question=input_text,
                plan=plan,
                current_step=step,
                completed_steps=self.last_step_results,
                **kwargs,
            )
            self.last_step_results.append({"step": step, "result": step_result})
            self.current_history.append(f"Step {index}: {step}")
            self.current_history.append(f"StepResult {index}: {step_result}")

        final_answer = self._generate_final_answer(input_text, plan, self.last_step_results, **kwargs)
        self._remember_assistant_text(final_answer, metadata={"memory_stage": "plan_final"})
        logger.info("[%s] Plan-and-Solve 完成: %s", self.name, self._preview(final_answer))
        return final_answer

    def _generate_plan(self, question: str, **kwargs: Any) -> List[str]:
        """调用模型生成计划，并尽量把输出解析成字符串列表。"""
        prompt = self.plan_prompt.format(question=question)
        raw_plan = self._request_text(prompt, **kwargs)
        logger.info("规划输出: %s", self._preview(raw_plan))
        return self._parse_plan(raw_plan)

    def _parse_plan(self, raw_plan: str) -> List[str]:
        """
        尽量从模型输出里解析出 Python 列表。

        这里先去掉 markdown code fence，再用 `ast.literal_eval()` 做安全解析。
        """
        if not raw_plan:
            return []

        fenced_match = re.search(r"```(?:python)?\s*(.*?)```", raw_plan, re.DOTALL | re.IGNORECASE)
        if fenced_match:
            raw_plan = fenced_match.group(1).strip()

        try:
            parsed = ast.literal_eval(raw_plan)
        except (SyntaxError, ValueError):
            logger.warning("规划结果无法解析为 Python 列表: %s", raw_plan)
            return []

        if not isinstance(parsed, list):
            logger.warning("规划结果不是列表: %r", parsed)
            return []

        cleaned_steps = [str(item).strip() for item in parsed if str(item).strip()]
        return cleaned_steps[: self.max_steps]

    def _solve_step(
        self,
        *,
        question: str,
        plan: List[str],
        current_step: str,
        completed_steps: List[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """对单个计划步骤执行一个小型 ReAct 循环。"""
        step_history: List[str] = []

        for round_index in range(1, self.max_step_rounds + 1):
            prompt = self.prompt_template.format(
                tools=self.tool_registry.describe_tools(),
                question=question,
                plan=self._render_plan(plan),
                completed_steps=self._render_completed_steps(completed_steps),
                current_step=current_step,
                step_history=self._render_history(step_history, empty_text="暂无步骤历史"),
            )
            logger.info("当前步骤 `%s` 第 %s 轮求解", current_step, round_index)
            logger.debug("步骤 Prompt 预览: %s", self._preview(prompt, limit=800))
            reply = self._request_text(prompt, **kwargs)
            logger.info("步骤 `%s` 模型输出: %s", current_step, self._preview(reply))

            if not reply:
                step_history.append("Observation: LLM returned an empty response.")
                continue

            thought, action_text = self.parse_react_response(reply)
            if thought:
                step_history.append(f"Thought: {thought}")

            if not action_text:
                step_history.append("Observation: Invalid response format: missing Action line.")
                continue

            step_history.append(f"Action: {action_text}")
            action_type, action_input = self.parse_action(action_text)

            if action_type == "finish":
                return action_input or ""

            observation = self._handle_action(action_type, action_input)
            step_history.append(f"Observation: {observation}")

        return f"Step unfinished after {self.max_step_rounds} rounds: {current_step}"

    def _generate_final_answer(
        self,
        question: str,
        plan: List[str],
        step_results: List[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """把所有步骤结果合成为用户可直接阅读的最终回答。"""
        prompt = self.final_prompt.format(
            question=question,
            plan=self._render_plan(plan),
            step_results=self._render_completed_steps(step_results),
        )
        return self._request_text(prompt, **kwargs)

    @staticmethod
    def _render_plan(plan: List[str]) -> str:
        """把步骤列表渲染成便于模型阅读的编号文本。"""
        if not plan:
            return "暂无计划"
        return "\n".join(f"{index}. {step}" for index, step in enumerate(plan, start=1))

    @staticmethod
    def _render_completed_steps(step_results: List[Dict[str, str]]) -> str:
        """把已完成步骤及其结果整理成摘要文本。"""
        if not step_results:
            return "暂无已完成步骤"
        lines: List[str] = []
        for index, item in enumerate(step_results, start=1):
            lines.append(f"{index}. 步骤: {item['step']}")
            lines.append(f"   结果: {item['result']}")
        return "\n".join(lines)
