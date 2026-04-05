from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from core import Config, HelloAgentsLLM, Message
from memory.manager import MemoryManager
from tools.builtin.toolRegistry import ToolRegistry
from tools.builtin.tool_base import Tool

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
            output = tool.run(parameters)
            output = output if output is not None else ""
            logger.info("工具 `%s` 执行结果: %s", action_type, self._preview(str(output)))
            return str(output) or "Tool returned empty output."
        except Exception as exc:  # noqa: BLE001 - 工具错误应当回传给模型继续推理
            logger.exception("tool %s execution failed", action_type)
            return f"Tool '{action_type}' failed: {exc}"

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
            return {}

        if isinstance(payload, dict):
            return payload

        if isinstance(payload, list):
            parameter_names = [item.name for item in parameters]
            if len(payload) != len(parameter_names):
                raise ValueError(
                    f"Tool '{tool.name}' expected {len(parameter_names)} parameters, got {len(payload)}."
                )
            return dict(zip(parameter_names, payload, strict=True))

        if len(parameters) == 1:
            return {parameters[0].name: payload}

        if not parameters:
            return {}

        raise ValueError(
            f"Tool '{tool.name}' requires structured input. Please pass a JSON object."
        )

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
