from __future__ import annotations

import inspect
import json
import logging
import re
from typing import Any, Callable, List, Optional, Tuple

from agents.agentBase import Agent
from core import Config, HelloAgentsLLM, Message
from tools.builtin.toolRegistry import ToolRegistry

logger = logging.getLogger(__name__)

# 给模型的核心约束提示词：
# 这个版本是“纯文本 ReAct”，不走原生 tool_calls，
# 而是要求模型输出 Thought / Action，再由 agent 自己解析。
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


class ReactAgent(Agent):
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
    ) -> None:
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.prompt_template = custom_prompt or MY_REACT_PROMPT

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

        # 每次 run 都从新的任务上下文开始，避免上一个问题污染当前推理。
        self.current_history = []
        self.clear_history()

        # system_prompt 放进消息历史里，便于后续扩展为真正的多轮消息流。
        if self.system_prompt:
            self.add_message(Message.system(self.system_prompt))
        self.add_message(Message.user(input_text))

        logger.info("[%s] 开始处理任务: %s", self.name, input_text)

        for step in range(1, self.max_steps + 1):
            # ReAct 的关键是把“问题 + 工具描述 + 历史观察”一起喂给模型。
            prompt = self._build_prompt(input_text)
            self._log_step_start(step, prompt)

            messages = self._build_messages(prompt)
            result = self.llm.chat(
                messages=messages,
                stream=False,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **kwargs,
            )

            reply = (result.text or "").strip()
            logger.info("[step %s] 模型原始输出: %s", step, self._preview(reply))

            # 模型空响应时不中断，记录一条 Observation，让下一轮继续修正。
            if not reply:
                observation = "LLM returned an empty response."
                self._append_observation(observation)
                continue

            # 原始 assistant 输出保存在消息历史里，便于后续调试或扩展记忆能力。
            self.add_message(Message.assistant(reply, metadata={"step": step}))
            thought, action_text = self.parse_react_response(reply)

            if thought:
                self.current_history.append(f"Thought: {thought}")
                logger.info("[step %s] Thought: %s", step, self._preview(thought))

            # 如果模型没有按格式给出 Action，就把错误反馈给下一轮。
            if not action_text:
                observation = "Invalid response format: missing Action line."
                self._append_observation(observation)
                continue

            self.current_history.append(f"Action: {action_text}")
            logger.info("[step %s] Action: %s", step, action_text)
            action_type, action_input = self.parse_action(action_text)

            # Finish[...] 代表模型认为信息已经足够，可以直接返回最终答案。
            if action_type == "finish":
                final_answer = action_input or ""
                logger.info("[step %s] 任务完成，最终答案: %s", step, self._preview(final_answer))
                return final_answer

            # 非 Finish 就尝试执行工具，并把结果当作 Observation 反馈给模型。
            observation = self._handle_action(action_type, action_input)
            self._append_observation(observation)

        # 超过最大步数仍未结束时，直接返回兜底信息，防止死循环。
        fallback = f"Reached max steps ({self.max_steps}) without Finish."
        logger.warning("[%s] %s", self.name, fallback)
        return fallback

    def _build_messages(self, prompt: str) -> List[Message]:
        """把本轮 prompt 包装成发给 LLM 的消息列表。"""
        messages: List[Message] = []
        if self.system_prompt:
            messages.append(Message.system(self.system_prompt))
        messages.append(Message.user(prompt))
        return messages

    def _build_prompt(self, question: str) -> str:
        """把工具说明、问题和历史 Thought/Action/Observation 拼成最终提示词。"""
        return self.prompt_template.format(
            tools=self._format_tools(),
            question=question,
            history="\n".join(self.current_history) or "暂无历史记录",
        )

    def _format_tools(self) -> str:
        """把 ToolRegistry 中的工具整理成模型容易阅读的文本列表。"""
        tools = self.tool_registry.getAvailableTools() or []
        if not tools:
            return "- 当前没有可用工具"

        lines: List[str] = []
        for item in tools:
            function = item.get("function", {})
            name = function.get("name", "unknown")
            description = function.get("description", "")
            lines.append(f"- {name}: {description}")
        return "\n".join(lines)

    def _handle_action(self, action_type: str, action_input: Optional[str]) -> str:
        """
        处理模型给出的 Action。

        - unknown: 模型格式不合法
        - tool name: 去注册表里找工具并执行
        返回值统一转成 Observation 文本，方便下一轮继续推理。
        """
        if action_type == "unknown":
            return f"Unknown action format: {action_input or ''}"

        tool = self.tool_registry.getTool(action_type)
        if tool is None:
            return f"Unknown tool: {action_type}"

        logger.info("开始执行工具 `%s`，原始参数: %s", action_type, action_input or "<empty>")

        try:
            # 先把模型传入的字符串尽量解析成结构化参数，再决定怎么调用工具。
            payload = self._parse_tool_input(action_input)
            logger.debug("工具 `%s` 解析后的参数: %r", action_type, payload)
            result = self._invoke_tool(tool, payload, action_input or "")
            output = "" if result is None else str(result)
            logger.info("工具 `%s` 执行结果: %s", action_type, self._preview(output))
            return output or "Tool returned empty output."
        except Exception as exc:  # noqa: BLE001 - tool errors should be surfaced as observations
            logger.exception("tool %s execution failed", action_type)
            return f"Tool '{action_type}' failed: {exc}"

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

        if text[0] in "[{\"" or text in {"true", "false", "null"}:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logger.debug("工具参数不是合法 JSON，按普通字符串处理: %s", text)
                return text

        return text

    def _invoke_tool(self, tool: Callable[..., Any], payload: Any, raw_input: str) -> Any:
        """
        按参数类型执行工具。

        这里做了一层兼容：
        - dict -> tool(**payload)
        - list -> tool(*payload)
        - 其他 -> tool(payload)
        如果因为签名不匹配报 TypeError，再退回原始字符串调用。
        """
        if payload is None:
            return tool()

        try:
            if isinstance(payload, dict):
                return tool(**payload)
            if isinstance(payload, list):
                return tool(*payload)
            return tool(payload)
        except TypeError:
            logger.debug("工具签名与解析后的参数不匹配，回退为原始字符串调用")
            signature = inspect.signature(tool)
            if len(signature.parameters) == 0:
                return tool()
            return tool(raw_input)

    def _append_observation(self, observation: str) -> None:
        """统一记录 Observation，并打印日志，方便排查每轮反馈内容。"""
        self.current_history.append(f"Observation: {observation}")
        logger.info("Observation: %s", self._preview(observation))

    def _log_step_start(self, step: int, prompt: str) -> None:
        """
        输出每一步开始时的调试日志。

        - INFO: 标记当前到了第几轮
        - DEBUG: 打印本轮 prompt 预览，避免日志过长时完全不可读
        """
        logger.info("========== Step %s / %s ==========" , step, self.max_steps)
        logger.debug("[step %s] Prompt 预览:\n%s", step, self._preview(prompt, limit=800))

    @staticmethod
    def _preview(text: str, limit: int = 200) -> str:
        """生成日志预览文本，避免长输出把终端刷满。"""
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return f"{compact[:limit]} ..."

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
