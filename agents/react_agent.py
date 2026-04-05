from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from core import Config, HelloAgentsLLM, Message
from memory.manager import MemoryManager
from tools.builtin.toolRegistry import ToolRegistry
from tools.builtin.tool_base import Tool, ToolResult, ToolValidationError

from .reasoning_agent_base import ReasoningAgentBase

logger = logging.getLogger(__name__)

# 给模型的核心约束提示词：
# 这个版本是“文本解析型 ReAct”，模型只负责输出 Thought / Action，
# 真正的工具解析与执行仍由 Agent 本身控制。
MY_REACT_PROMPT = """你是一个具备推理和行动能力的 AI 助手。
你必须严格遵守下面的输出格式，并且每一轮只做一件事。

可用工具：
{tools}

输出格式：
Thought: 简要说明你当前的判断，以及下一步准备做什么。
Action: 只能是下面两种格式之一：
- tool_name[tool_input]
- Finish[最终答案]

规则：
1. 如果你还需要外部信息，就调用工具。
2. 如果你已经有足够信息回答问题，就输出 Finish[...].
3. 不要输出多余段落，不要省略 Thought 或 Action。
4. tool_input 尽量简洁；如果需要多个参数，使用 JSON 对象字符串。
5. 一次只允许调用一个工具，不要在一轮里写多个 Action。

当前问题：
{question}

历史记录：
{history}
"""

NATIVE_TOOL_CALLING_PROMPT = """你是一个会使用工具解决问题的 AI 助手。

规则：
1. 当你需要外部信息时，直接使用提供的工具，不要伪造工具结果。
2. 如果已经有足够信息，请直接用自然语言回答用户。
3. 回答时保持简洁、准确，并优先基于工具结果和显式上下文。
4. 如果上下文仍不足以支持结论，请明确说明不确定性。

用户问题：
{question}
"""


class ReactAgent(ReasoningAgentBase):
    """一个最小可运行的文本版 ReAct Agent。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None,
        memory_manager: Optional[MemoryManager] = None,
        session_id: Optional[str] = None,
    ) -> None:
        # 修改说明：工具注册、prompt 模板和运行期 history 现在都交给公共父类管理。
        super().__init__(
            name=name,
            llm=llm,
            tool_registry=tool_registry,
            prompt_template=custom_prompt or MY_REACT_PROMPT,
            system_prompt=system_prompt,
            config=config,
            memory_manager=memory_manager,
            session_id=session_id,
        )
        self.max_steps = max_steps
        self.enable_native_tool_calling = custom_prompt is None

    def run(self, input_text: str, stream: bool = False, **kwargs: Any) -> str:
        """
        运行主循环。

        流程：
        1. 根据用户问题和历史 Observation 组装 prompt
        2. 调用 LLM 输出 Thought / Action
        3. 解析 Action，决定是结束还是继续调用工具
        4. 把工具结果作为 Observation 追加到历史，进入下一轮
        """
        del stream  # 当前版本需要拿到完整回复后再解析 Thought / Action。

        # 修改说明：任务启动阶段统一走公共父类，减少不同 Agent 之间的重复代码。
        self._start_new_run(input_text)

        if self._should_use_native_tool_calling():
            logger.info("[%s] 当前启用原生 tool calling 模式。", self.name)
            return self._run_with_native_tool_calling(input_text, **kwargs)

        logger.info("[%s] 开始处理任务: %s", self.name, input_text)

        for step in range(1, self.max_steps + 1):
            prompt = self._build_prompt(input_text)
            self._log_step_start(step, prompt)
            reply = self._request_text(prompt, **kwargs)
            logger.info("[step %s] 模型原始输出: %s", step, self._preview(reply))

            # 模型空响应时不中断，记录一条 Observation，让下一轮继续修正。
            if not reply:
                self._append_observation("LLM returned an empty response.")
                continue

            self.add_message(Message.assistant(reply, metadata={"step": step}))
            thought, action_text = self.parse_react_response(reply)

            if thought:
                self.current_history.append(f"Thought: {thought}")
                logger.debug("[step %s] Thought: %s", step, self._preview(thought))

            if not action_text:
                self._append_observation("Invalid response format: missing Action line.")
                continue

            self.current_history.append(f"Action: {action_text}")
            logger.debug("[step %s] Action: %s", step, action_text)
            action_type, action_input = self.parse_action(action_text)

            if action_type == "finish":
                final_answer = action_input or ""
                self._remember_assistant_text(final_answer, metadata={"memory_stage": "react_finish"})
                logger.info("[step %s] 任务完成，最终答案: %s", step, self._preview(final_answer))
                return final_answer

            observation = self._handle_action(action_type, action_input)
            self._append_observation(observation)

        fallback = f"Reached max steps ({self.max_steps}) without Finish."
        self._remember_assistant_text(fallback, metadata={"memory_stage": "react_fallback"})
        logger.warning("[%s] %s", self.name, fallback)
        return fallback

    def _run_with_native_tool_calling(self, question: str, **kwargs: Any) -> str:
        """
        运行基于原生 tool calling 的主循环。

        修改说明：这里不再依赖文本 `Thought / Action` 解析，
        而是直接复用工具 schema，让模型通过标准 `tool_calls` 返回调用意图。
        """
        logger.info("[%s] 开始原生 tool calling 循环。", self.name)
        final_answer = self._run_native_tool_calling_loop(
            prompt=NATIVE_TOOL_CALLING_PROMPT.format(question=question),
            max_rounds=self.max_steps,
            on_assistant_message=lambda message, result, round_index: self.add_message(message),
            tool_metadata_factory=lambda round_index, tool_call: {
                "native_tool_calling": True,
                "round": round_index,
            },
            on_tool_message=lambda message, tool_call, observation, round_index: self._remember_message(message),
            **kwargs,
        )
        if final_answer:
            self._remember_assistant_text(
                final_answer,
                metadata={"memory_stage": "native_tool_calling_finish"},
            )
            logger.info("[%s] 原生 tool calling 任务完成。", self.name)
            return final_answer

        fallback = f"Reached max steps ({self.max_steps}) without final answer in native tool calling mode."
        self._remember_assistant_text(
            fallback,
            metadata={"memory_stage": "native_tool_calling_fallback"},
        )
        logger.warning("[%s] %s", self.name, fallback)
        return fallback

    def _build_prompt(self, question: str) -> str:
        """把工具说明、问题和历史 Thought/Action/Observation 拼成最终提示词。"""
        return self.prompt_template.format(
            tools=self.tool_registry.describe_tools(),
            question=question,
            history=self._render_history(self.current_history),
        )

    def _handle_action(self, action_type: str, action_input: Optional[str]) -> str:
        """
        处理模型给出的 Action。

        - unknown: 模型格式不合法
        - tool name: 去注册表里找工具并执行
        返回值统一转成 Observation 文本，方便下一轮继续推理。
        """
        if action_type == "unknown":
            return f"Unknown action format: {action_input or ''}"

        tool = self.tool_registry.get_tool(action_type)
        if tool is None:
            return f"Unknown tool: {action_type}"

        logger.info("开始执行工具 `%s`，原始参数: %s", action_type, action_input or "<empty>")

        try:
            parameters = self._prepare_tool_parameters(tool, action_input)
            logger.debug("工具 `%s` 解析后的参数: %r", action_type, parameters)
            result = self._execute_tool_with_recovery(tool, parameters)
            logger.info("工具 `%s` 执行结果: %s", action_type, self._preview(result.render_for_observation()))
            return self._handle_tool_result(action_type, result)
        except Exception as exc:  # noqa: BLE001 - 参数准备错误应当回传给模型继续推理
            logger.exception("tool %s parameter preparation failed", action_type)
            failure_result = ToolResult.fail(
                str(exc),
                meta={
                    "tool": action_type,
                    "retryable": False,
                    "failure_stage": "parameter_prepare",
                },
            )
            failure_result = self._finalize_failed_tool_result(
                action_type,
                failure_result,
                attempt=1,
                max_attempts=1,
            )
            return self._handle_tool_result(action_type, failure_result)

    def _handle_tool_result(self, tool_name: str, result: ToolResult) -> str:
        """
        统一消费工具结果协议。

        修改说明：Agent 不再假设工具只会返回字符串，而是统一围绕 `ToolResult`
        做 Observation 渲染、错误拼装和成功结果记忆。
        """
        base_observation = result.render_for_observation()
        observation = base_observation if result.success else f"Tool '{tool_name}' failed: {base_observation}"
        logger.debug(
            "工具 `%s` 结构化结果 | success=%s | meta=%s | data=%s",
            tool_name,
            result.success,
            result.meta,
            self._preview(self._stringify_tool_payload(result.data), limit=200),
        )
        self._stash_tool_result_snapshot(tool_name, observation, result)
        if result.success:
            self._remember_tool_observation(tool_name, observation, result=result)
        if not self._native_tool_execution_in_progress:
            self._remember_tool_result_memory(tool_name, observation, result)
        return observation

    def _execute_tool_with_recovery(self, tool: Tool, parameters: Dict[str, Any]) -> ToolResult:
        """
        执行工具，并在失败时按配置做最小可用的重试与降级。

        修改说明：这一步把“失败恢复策略”收敛到一处，后面如果要接更复杂的
        backoff / 熔断 / 降级路由，也只需要沿着这里继续扩展。
        """
        max_attempts = max(1, int(self.config.tool_max_retries or 0) + 1)
        last_result: Optional[ToolResult] = None

        for attempt in range(1, max_attempts + 1):
            result = tool.execute(parameters)
            result = self._annotate_tool_attempt_metadata(
                result=result,
                tool_name=tool.name,
                attempt=attempt,
                max_attempts=max_attempts,
            )
            last_result = result

            if result.success:
                if attempt > 1:
                    logger.info("工具 `%s` 在第 %s 次尝试后恢复成功。", tool.name, attempt)
                return result

            if not self._should_retry_tool_result(result, attempt=attempt, max_attempts=max_attempts):
                return self._finalize_failed_tool_result(
                    tool.name,
                    result,
                    attempt=attempt,
                    max_attempts=max_attempts,
                )

            logger.warning(
                "工具 `%s` 第 %s/%s 次执行失败，准备重试: %s",
                tool.name,
                attempt,
                max_attempts,
                self._preview(result.render_for_observation()),
            )
            self._sleep_before_tool_retry()

        assert last_result is not None  # 理论上不会为空，这里只是给类型系统一个兜底。
        return self._finalize_failed_tool_result(
            tool.name,
            last_result,
            attempt=max_attempts,
            max_attempts=max_attempts,
        )

    def _annotate_tool_attempt_metadata(
        self,
        *,
        result: ToolResult,
        tool_name: str,
        attempt: int,
        max_attempts: int,
    ) -> ToolResult:
        """给工具结果补充当前重试轮次信息。"""
        metadata = dict(result.meta)
        metadata.update(
            {
                "tool": tool_name,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "recovered_after_retry": result.success and attempt > 1,
            }
        )
        return result.model_copy(update={"meta": metadata})

    def _should_retry_tool_result(
        self,
        result: ToolResult,
        *,
        attempt: int,
        max_attempts: int,
    ) -> bool:
        """根据 ToolResult 判断当前失败是否值得自动重试。"""
        if result.success or attempt >= max_attempts:
            return False

        explicit_retryable = result.meta.get("retryable")
        if isinstance(explicit_retryable, bool):
            return explicit_retryable

        combined = " ".join(
            part for part in [result.error.strip(), result.content.strip()] if part
        ).lower()
        transient_keywords = (
            "timeout",
            "timed out",
            "tempor",
            "connection",
            "unavailable",
            "rate limit",
            "429",
            "busy",
            "reset",
            "refused",
            "稍后重试",
            "超时",
            "连接",
        )
        return any(keyword in combined for keyword in transient_keywords)

    def _finalize_failed_tool_result(
        self,
        tool_name: str,
        result: ToolResult,
        *,
        attempt: int,
        max_attempts: int,
    ) -> ToolResult:
        """在最终失败时补一层降级提示，避免模型收到过于生硬的错误。"""
        metadata = dict(result.meta)
        metadata.update(
            {
                "attempt": attempt,
                "max_attempts": max_attempts,
                "recovery_exhausted": True,
            }
        )
        if not self.config.tool_enable_graceful_degradation:
            return result.model_copy(update={"meta": metadata})

        guidance = self._build_tool_degradation_guidance(
            tool_name=tool_name,
            result=result,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        content = result.content.strip()
        if guidance and guidance not in content:
            content = f"{content}\n{guidance}".strip() if content else guidance
        metadata["degraded"] = True
        return result.model_copy(update={"content": content, "meta": metadata})

    def _build_tool_degradation_guidance(
        self,
        *,
        tool_name: str,
        result: ToolResult,
        attempt: int,
        max_attempts: int,
    ) -> str:
        """根据失败类型生成一条适合回填给模型的降级提示。"""
        stage = str(result.meta.get("failure_stage", "") or "").strip().lower()
        if stage == "parameter_prepare":
            return "降级提示：请检查工具参数格式，必要时改用 JSON 对象重新调用。"

        if max_attempts > 1:
            return (
                f"降级提示：工具 `{tool_name}` 已自动重试 {attempt} 次仍失败。"
                "如果问题不是强依赖外部工具，请基于现有上下文说明不确定性，不要伪造工具结果。"
            )

        return (
            f"降级提示：工具 `{tool_name}` 当前不可用。"
            "如果无法再次获取外部信息，请明确说明失败原因。"
        )

    def _sleep_before_tool_retry(self) -> None:
        """按配置在工具重试前等待一小段时间。"""
        backoff_ms = int(self.config.tool_retry_backoff_ms or 0)
        if backoff_ms <= 0:
            return
        time.sleep(backoff_ms / 1000)

    def _prepare_tool_parameters(self, tool: Tool, action_input: Optional[str]) -> Dict[str, Any]:
        """
        把 Action 中的字符串参数转成工具真正需要的字典参数。

        兼容策略：
        - 空输入 + 无参数工具 -> {}
        - JSON 对象 -> 直接作为参数字典
        - 单值输入 + 工具只有一个参数 -> 自动映射到该参数名
        - JSON 数组 + 参数数量匹配 -> 按顺序映射到参数名
        """
        parameters = tool.get_parameters()
        payload = self._parse_tool_input(action_input)

        if payload in (None, ""):
            if parameters:
                raise ValueError(f"Tool '{tool.name}' requires parameters.")
            return tool.normalize_parameters({})

        if isinstance(payload, dict):
            return tool.normalize_parameters(payload)

        if isinstance(payload, list):
            parameter_names = [item.name for item in parameters]
            if len(payload) != len(parameter_names):
                raise ValueError(
                    f"Tool '{tool.name}' expected {len(parameter_names)} parameters, got {len(payload)}."
                )
            return tool.normalize_parameters(dict(zip(parameter_names, payload, strict=True)))

        if len(parameters) == 1:
            return tool.normalize_parameters({parameters[0].name: payload})

        if not parameters:
            raise ValueError(f"Tool '{tool.name}' does not take parameters.")

        try:
            return tool.normalize_parameters({})
        except ToolValidationError as exc:
            raise ValueError(
                f"Tool '{tool.name}' requires structured input. Please pass a JSON object."
            ) from exc

    def _parse_tool_input(self, action_input: Optional[str]) -> Any:
        """
        解析工具参数。

        约定：
        - JSON 对象 -> dict，适合 keyword args
        - JSON 数组 -> list，适合 positional args
        - 其他内容 -> 普通字符串
        """
        if action_input is None:
            return None

        text = action_input.strip()
        if not text:
            return ""

        if text[0] in '[{"' or text in {"true", "false", "null"}:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logger.debug("工具参数不是合法 JSON，按普通字符串处理: %s", text)
                return text

        return text

    def _append_observation(self, observation: str) -> None:
        """统一记录 Observation，并打印日志，方便排查每轮反馈内容。"""
        self.current_history.append(f"Observation: {observation}")
        logger.debug("Observation: %s", self._preview(observation))

    def _log_step_start(self, step: int, prompt: str) -> None:
        """
        输出每一步开始时的调试日志。

        - INFO: 标记当前到了第几轮
        - DEBUG: 打印本轮 prompt 预览，避免日志过长时完全不可读
        """
        logger.info("========== Step %s / %s ==========", step, self.max_steps)
        logger.debug("[step %s] Prompt 预览:\n%s", step, self._preview(prompt, limit=800))

    def parse_react_response(self, text: str) -> Tuple[str, str]:
        """
        从模型原始文本中提取 Thought 和 Action。

        只负责解析整段回复的结构，不判断 Action 是否合法。
        Action 的语义校验放在 `parse_action()` 里处理。
        """
        text = text.strip()

        thought_pattern = r"Thought:\s*(.*?)(?=\nAction:|$)"
        thought_match = re.search(thought_pattern, text, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else ""

        action_pattern = r"Action:\s*(.*)"
        action_match = re.search(action_pattern, text, re.DOTALL | re.IGNORECASE)
        action = action_match.group(1).strip() if action_match else ""

        return thought, action

    def parse_action(self, action: str) -> Tuple[str, Optional[str]]:
        """
        解析 Action 行。

        支持两种格式：
        - Finish[...]
        - tool_name[...]

        返回 `(action_type, action_input)`：
        - `finish` 表示任务结束
        - 其他字符串表示工具名
        - `unknown` 表示格式无法识别
        """
        action = action.strip()

        finish_match = re.fullmatch(r"Finish\[(.*)\]", action, re.DOTALL)
        if finish_match:
            return "finish", finish_match.group(1).strip()

        tool_match = re.fullmatch(r"([A-Za-z_][\w]*)\[(.*)\]", action, re.DOTALL)
        if tool_match:
            return tool_match.group(1), tool_match.group(2).strip()

        return "unknown", action
