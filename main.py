# 狠狠测试

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from agents.plan_and_solve import PlanAndSolveAgent
from agents.react_agent import ReactAgent
from agents.reflection_agent import ReflectionAgent
from core import ChatResult, Config, HelloAgentsLLM
from memory import MemoryConfig, MemoryManager
from memory.rag import RagPipeline
from tools.builtin.get_time import GetTimeTool
from tools.builtin.memory_tool import MemoryTool
from tools.builtin.rag_tool import RagTool
from tools.builtin.tool_base import Tool, ToolConditionalRule, ToolParameter, ToolResult, ToolValidationError
from tools.builtin.toolRegistry import ToolRegistry


@dataclass
class NativeToolCallingSmokeLLM:
    """
    一个最小的假 LLM，用来验证原生 tool calling 主链路。

    修改说明：这里不依赖真实模型服务，直接返回“先调用工具、再生成答案”的固定流程，
    方便测试 schema / 原生 tool calling 的执行路径是否闭环。
    """

    provider: str = "mock-native"

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        tool_messages = [message for message in messages if getattr(message, "role", None) == "tool"]
        if not tool_messages:
            return ChatResult(
                text="",
                tool_calls=[
                    {
                        "id": "call_get_time_001",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": ""},
                    }
                ],
                finish_reason="tool_calls",
            )

        latest_tool_result = tool_messages[-1].content or ""
        return ChatResult(
            text=f"我已经通过原生 tool calling 调用了 get_time，结果是：{latest_tool_result}",
            tool_calls=[],
            finish_reason="stop",
        )


@dataclass
class NativePlanAndSolveSmokeLLM:
    """用于验证 Plan-and-Solve 混合 tool calling 链路的假 LLM。"""

    provider: str = "mock-plan-native"

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        tools = kwargs.get("tools")
        user_messages = [message for message in messages if getattr(message, "role", None) == "user"]
        last_user = user_messages[-1].content if user_messages else ""
        tool_messages = [message for message in messages if getattr(message, "role", None) == "tool"]

        if not tools:
            if "请严格按照下面格式输出" in (last_user or ""):
                return ChatResult(text='["获取当前时间", "根据已获取的信息组织最终表述"]')
            if "步骤结果：" in (last_user or ""):
                return ChatResult(text="我先通过工具拿到当前时间，再基于步骤结果组织成最终回答。")
            return ChatResult(text="")

        current_step_match = re.search(r"当前步骤：\s*(.+)", last_user or "")
        current_step = current_step_match.group(1).strip() if current_step_match else ""

        if current_step == "获取当前时间":
            if not tool_messages:
                return ChatResult(
                    text="",
                    tool_calls=[
                        {
                            "id": "plan_get_time_001",
                            "type": "function",
                            "function": {"name": "get_time", "arguments": ""},
                        }
                    ],
                    finish_reason="tool_calls",
                )
            latest_tool = tool_messages[-1].content or ""
            return ChatResult(text=f"当前步骤已完成：我已经获取到当前时间，结果为 {latest_tool}")

        return ChatResult(text="当前步骤已完成：我已经基于上一阶段结果整理好了最终表述。")


@dataclass
class NativeReflectionSmokeLLM:
    """用于验证 Reflection 草稿阶段混合 tool calling 的假 LLM。"""

    provider: str = "mock-reflection-native"

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        tools = kwargs.get("tools")
        user_messages = [message for message in messages if getattr(message, "role", None) == "user"]
        last_user = user_messages[-1].content if user_messages else ""
        tool_messages = [message for message in messages if getattr(message, "role", None) == "tool"]

        if tools:
            if not tool_messages:
                return ChatResult(
                    text="",
                    tool_calls=[
                        {
                            "id": "reflection_get_time_001",
                            "type": "function",
                            "function": {"name": "get_time", "arguments": ""},
                        }
                    ],
                    finish_reason="tool_calls",
                )
            latest_tool = tool_messages[-1].content or ""
            return ChatResult(text=f"草稿答案：我已经通过工具确认当前时间，结果是 {latest_tool}。")

        if "Decision:" in (last_user or ""):
            return ChatResult(
                text=(
                    "Reflection: 当前答案已经基于工具结果给出了关键信息，内容完整且清晰。\n"
                    "Decision: finish\n"
                    "Suggestions:\n- 可以直接结束，不需要额外修订。"
                )
            )

        if "审查意见" in (last_user or ""):
            return ChatResult(text="修订后答案：我已经通过工具确认当前时间，并确保表达更清晰。")

        return ChatResult(text="")


class SchemaSmokeTool(Tool):
    """
    专门用于演示第一版工具 schema 校验的测试工具。

    修改说明：这里不依赖真实外部服务，只验证“参数约束 -> 归一化 -> 错误提示”
    这条链路是否已经具备最小可用能力。
    """

    def __init__(self) -> None:
        super().__init__(
            name="schema_smoke_tool",
            description="用于验证工具 schema 校验与参数归一化的演示工具。",
        )

    def run(self, parameters: dict[str, object]) -> ToolResult:
        """把归一化后的参数直接格式化输出，方便在 smoke test 里观察。"""
        content = (
            f"mode={parameters.get('mode')} | "
            f"level={parameters.get('level')} | "
            f"dry_run={parameters.get('dry_run')}"
        )
        return ToolResult.ok(
            content,
            data=dict(parameters),
            meta={"tool": "schema_smoke_tool"},
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="mode",
                type="string",
                description="运行模式。",
                choices=["safe", "fast"],
            ),
            ToolParameter(
                name="level",
                type="integer",
                description="执行级别。",
                required=False,
                default=1,
                minimum=1,
                maximum=5,
            ),
            ToolParameter(
                name="dry_run",
                type="boolean",
                description="是否只做演练，不真正执行。",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="payload",
                type="object",
                description="复杂嵌套对象参数，用于验证 object / array 的递归 schema。",
                required=False,
                object_properties=[
                    ToolParameter(
                        name="query",
                        type="string",
                        description="本次请求的主题。",
                        min_length=2,
                    ),
                    ToolParameter(
                        name="options",
                        type="object",
                        description="额外选项。",
                        required=False,
                        default={},
                        object_properties=[
                            ToolParameter(
                                name="timezone",
                                type="string",
                                description="可选的时区名称。",
                                required=False,
                                default="",
                            ),
                            ToolParameter(
                                name="include_seconds",
                                type="boolean",
                                description="是否输出秒。",
                                required=False,
                                default=False,
                            ),
                        ],
                    ),
                    ToolParameter(
                        name="tags",
                        type="array",
                        description="附加标签列表。",
                        required=False,
                        default=[],
                        items_type="string",
                    ),
                ],
            ),
        ]

    def get_conditional_rules(self) -> list[ToolConditionalRule]:
        """
        演示条件 schema：当 mode=fast 时，payload 必须出现。

        修改说明：这样 smoke test 可以同时看到：
        1. schema 里已经导出了 `if/then`
        2. 本地执行前也会走同一套条件校验
        """
        return [
            ToolConditionalRule(field="mode", equals="fast", required=["payload"]),
        ]

    def validate_normalized_parameters(self, parameters: dict[str, object]) -> dict[str, object]:
        """演示跨字段语义校验。"""
        mode = str(parameters.get("mode", "")).strip().lower()
        level = int(parameters.get("level", 1) or 1)
        if mode == "fast" and level < 3:
            raise ToolValidationError("Tool 'schema_smoke_tool' requires level >= 3 when mode=fast.")
        return parameters


class FlakyRecoveryTool(Tool):
    """
    用于验证 Agent 自动重试能力的演示工具。

    修改说明：第一次调用故意返回可重试失败，第二次再成功，
    这样可以稳定观察“失败恢复”是否真的在 Agent 层生效。
    """

    def __init__(self) -> None:
        super().__init__(
            name="flaky_recovery_tool",
            description="第一次失败、第二次成功的测试工具。",
        )
        self.calls = 0

    def run(self, parameters: dict[str, object]) -> ToolResult:
        del parameters
        self.calls += 1
        if self.calls == 1:
            return ToolResult.fail(
                "模拟临时超时，请稍后重试。",
                meta={
                    "tool": self.name,
                    "retryable": True,
                    "failure_stage": "remote_call",
                },
            )
        return ToolResult.ok(
            "工具在重试后恢复成功。",
            data={"calls": self.calls},
            meta={"tool": self.name, "action": "recover"},
        )

    def get_parameters(self) -> list[ToolParameter]:
        return []


class AlwaysFailTool(Tool):
    """
    用于验证降级提示的演示工具。

    修改说明：这个工具始终失败，便于观察重试耗尽后 Agent 是否会补充降级说明。
    """

    def __init__(self) -> None:
        super().__init__(
            name="always_fail_tool",
            description="始终失败的测试工具。",
        )

    def run(self, parameters: dict[str, object]) -> ToolResult:
        del parameters
        return ToolResult.fail(
            "模拟服务当前不可用。",
            meta={
                "tool": self.name,
                "retryable": True,
                "failure_stage": "service_unavailable",
            },
        )

    def get_parameters(self) -> list[ToolParameter]:
        return []


def configure_logging() -> None:
    """
    配置更干净的控制台日志。

    这里默认优先保证 main.py 的测试输出清晰，
    如果后面你想看更细的 Agent 运行细节，再把相应 logger 调高即可。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    # 修改说明：main 演示时只保留最关键的输出，避免网络和中间调试日志把终端刷乱。
    for logger_name in (
        "openai",
        "openai._base_client",
        "httpx",
        "httpcore",
        "agents",
        "tools",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def build_memory_manager() -> MemoryManager:
    """
    创建默认的记忆管理器。

    修改说明：当前已经包含“工作记忆 + 情景记忆 + 最小语义记忆”，
    并且会优先读取环境变量里的向量存储配置，方便切到真实 Qdrant。
    """
    return MemoryManager(MemoryConfig.from_env())


def describe_embedding_backend(memory_manager: MemoryManager) -> str:
    """
    返回当前 embedding 后端的简要说明。

    修改说明：后面接入本地 Ollama embedding 后，测试输出里需要明确看到
    “现在到底是 hash 还是 ollama，在用哪个模型”，排查会快很多。
    """
    config = memory_manager.config
    backend = (config.embedding_backend or "hash").strip().lower()
    if backend == "ollama":
        return f"ollama | model={config.ollama_embedding_model or '(未配置)'} | base_url={config.ollama_embedding_base_url}"
    return f"{backend} | dimensions={config.embedding_dimensions}"


def build_rag_pipeline(memory_manager: MemoryManager) -> RagPipeline:
    """
    创建默认的 RAG 管道。

    修改说明：RAG 先复用记忆系统里的 embedding 配置，
    这样记忆检索和文档检索在同一个项目里保持一致的向量表示方式。
    """
    return RagPipeline(
        config=memory_manager.config,
        embedding_service=memory_manager.embedding_service,
    )


def build_tool_registry(
    memory_manager: MemoryManager,
    session_id: str,
    rag_pipeline: RagPipeline | None = None,
) -> ToolRegistry:
    """
    创建当前示例使用的工具注册表。

    修改说明：工具注册表现在不仅包含 `get_time`，
    还会注入 `memory_tool` 和 `rag_tool`，方便 Agent 显式读取记忆或检索本地文档。
    """
    registry = ToolRegistry()
    registry.register_tool(GetTimeTool())
    registry.register_tool(MemoryTool(memory_manager=memory_manager, session_id=session_id))
    if rag_pipeline is not None:
        registry.register_tool(RagTool(rag_pipeline=rag_pipeline))
    return registry


def _build_llm_and_config() -> tuple[HelloAgentsLLM, Config]:
    """
    统一创建 LLM 客户端和配置对象。

    把构造逻辑集中在一个函数里，后续切换模型配置时只需要改一处。
    """
    config = Config.from_env()
    llm = HelloAgentsLLM.from_config(config)
    return llm, config


def build_react_agent() -> ReactAgent:
    """构造一个可直接运行的 ReAct Agent。"""
    llm, config = _build_llm_and_config()
    memory_manager = build_memory_manager()
    rag_pipeline = build_rag_pipeline(memory_manager)
    return ReactAgent(
        name="react_agent",
        llm=llm,
        tool_registry=build_tool_registry(memory_manager, "react_agent", rag_pipeline),
        config=config,
        max_steps=5,
        memory_manager=memory_manager,
        session_id="react_agent",
    )


def build_plan_and_solve_agent() -> PlanAndSolveAgent:
    """构造一个可直接运行的 Plan-and-Solve Agent。"""
    llm, config = _build_llm_and_config()
    memory_manager = build_memory_manager()
    rag_pipeline = build_rag_pipeline(memory_manager)
    return PlanAndSolveAgent(
        name="plan_and_solve_agent",
        llm=llm,
        tool_registry=build_tool_registry(memory_manager, "plan_and_solve_agent", rag_pipeline),
        config=config,
        max_steps=5,
        max_step_rounds=4,
        memory_manager=memory_manager,
        session_id="plan_and_solve_agent",
    )


def build_reflection_agent() -> ReflectionAgent:
    """构造一个可直接运行的 Reflection Agent。"""
    llm, config = _build_llm_and_config()
    memory_manager = build_memory_manager()
    rag_pipeline = build_rag_pipeline(memory_manager)
    return ReflectionAgent(
        name="reflection_agent",
        llm=llm,
        tool_registry=build_tool_registry(memory_manager, "reflection_agent", rag_pipeline),
        config=config,
        max_steps=5,
        max_reflections=2,
        memory_manager=memory_manager,
        session_id="reflection_agent",
    )


def print_run_summary(title: str, answer: str, history: list[str]) -> None:
    """
    统一打印测试结果。

    日志负责展示运行阶段的关键节点，
    这里则把最终答案和执行轨迹按固定格式整理出来，便于人工检查。
    """
    print("\n" + "=" * 24)
    print(title)
    print("=" * 24)
    print(answer)

    print("\n" + "=" * 24)
    print("执行轨迹")
    print("=" * 24)
    if not history:
        print("(暂无轨迹)")
        return

    for index, item in enumerate(history, start=1):
        print(f"{index:02d}. {item}")


def print_memory_snapshot(agent: ReactAgent, title: str, query: str = "") -> None:
    """
    打印当前会话的记忆快照，方便直接观察记忆模块是否写入成功。

    修改说明：把记忆验证结果单独打印出来，避免你只能从模型回答里“猜”记忆有没有生效。
    """
    print("\n" + "=" * 24)
    print(title)
    print("=" * 24)

    if agent.memory_manager is None:
        print("(当前 Agent 未启用记忆管理器)")
        return

    memory_text = agent.memory_manager.build_memory_prompt(
        session_id=agent.session_id,
        query=query,
        exclude_text=query or None,
    )
    print(memory_text or "(当前没有可展示的记忆)")


def print_runtime_error(exc: Exception) -> None:
    """
    在演示入口里统一打印精简错误信息。

    修改这个函数的目的，是避免 SDK 的长堆栈直接把终端刷屏，
    让你第一眼就能看到是配置问题、连接问题，还是 Agent 本身的逻辑问题。
    """
    print("\n" + "!" * 24)
    print("运行失败")
    print("!" * 24)
    print(f"{type(exc).__name__}: {exc}")
    print("请检查 .env 里的 provider / base_url / api_key 配置，或确认本地模型服务已经启动。")


def ensure_demo_rag_document() -> Path:
    """
    生成一个本地 RAG 测试文档。

    修改说明：把示例知识写到 `data/` 目录里，方便你直接运行 `main.py`
    就能验证索引、检索和 Agent 使用 `rag_tool` 的完整链路。
    """
    path = Path("data/demo_rag_notes.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "项目笔记：helloAgent 当前已经支持 ReAct、Plan-and-Solve、Reflection 三种 Agent 范式。\n"
        "记忆系统已经包含 WorkingMemory、EpisodicMemory 和 SemanticMemory。\n"
        "用户偏好示例：用户喜欢美式咖啡，不喜欢过甜的饮品。\n"
        "RAG 目标：先完成本地文档切片、向量检索和 `rag_tool`，后续再接真实 Qdrant 与 Neo4j。\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def test_react_agent() -> None:
    """
    运行 ReAct Agent 示例。

    这个问题会引导模型主动调用 `get_time`，
    便于验证最基础的工具调用链路是否正常。
    """
    agent = build_react_agent()
    question = "请调用 get_time 工具，并告诉我当前时间是什么。"
    answer = agent.run(question)
    print_run_summary("ReAct Agent 测试", answer, agent.current_history)


def test_plan_and_solve_agent() -> None:
    """
    运行 Plan-and-Solve Agent 示例。

    这个问题会先拆计划，再逐步求解，
    适合检查规划、步骤执行和最终汇总是否连贯。
    """
    agent = build_plan_and_solve_agent()
    question = "请先规划步骤，再回答现在的本地时间，并解释你是怎么得到它的。"
    answer = agent.run(question)
    print_run_summary("Plan-and-Solve Agent 测试", answer, agent.current_history)


def test_reflection_agent() -> None:
    """
    运行 Reflection Agent 示例。

    这个问题会先生成草稿，再做自我审查，
    最后检查修订后的答案是否比草稿更完整。
    """
    agent = build_reflection_agent()
    question = "请结合 get_time 工具回答当前时间，并确认你的答案是否完整清晰。"
    answer = agent.run(question)
    print_run_summary("Reflection Agent 测试", answer, agent.current_history)


def test_memory_workflow() -> None:
    """
    运行一个两轮对话的记忆测试。

    第一轮让 Agent 记住用户偏好，
    第二轮要求它回忆上一轮信息，并额外打印记忆快照帮助排查。
    """
    agent = build_react_agent()

    if agent.memory_manager is not None:
        # 修改说明：测试开始前先清空同 session 的旧记忆，避免历史脏数据影响本次验证结果。
        agent.memory_manager.clear_session(agent.session_id)

    first_question = "请记住我的偏好：我喜欢美式咖啡。你只需要简短确认。"
    first_answer = agent.run(first_question)
    print_run_summary("记忆测试 - 第一轮", first_answer, agent.current_history)
    print_memory_snapshot(agent, "第一轮后的记忆快照")

    second_question = "你还记得我刚才告诉你的饮品偏好吗？请直接回答。"
    second_answer = agent.run(second_question)
    print_run_summary("记忆测试 - 第二轮", second_answer, agent.current_history)
    print_memory_snapshot(agent, "第二轮后的记忆快照")


def test_memory_closure_smoke() -> None:
    """
    运行一个记忆闭环冒烟测试。

    修改说明：这个测试不依赖真实 LLM，重点验证：
    1. 低价值内容会不会被过滤
    2. 重复内容会不会被跳过
    3. preference / fact / tool_result 会不会得到不同层级的写入策略
    4. 长期保留策略会不会优先留下高价值记忆
    """
    config = MemoryConfig.from_env().model_copy(
        update={
            "vector_store_backend": "json",
            "episodic_retention_max_items": 3,
            "semantic_retention_max_items": 2,
        }
    )
    memory_manager = MemoryManager(config)
    session_id = "memory_closure_smoke"
    memory_manager.clear_session(session_id)

    samples = [
        ("assistant", "好的。", {"memory_stage": "react_finish"}),
        ("user", "我喜欢美式咖啡，不喜欢太甜的饮品。", {}),
        ("user", "我喜欢美式咖啡，不喜欢太甜的饮品。", {}),
        ("user", "我最近希望回答更简洁一些。", {}),
        ("assistant", "helloAgent 当前支持 ReAct、Plan-and-Solve、Reflection。", {"memory_stage": "plan_final"}),
        ("assistant", "昨天我们还讨论了 README 的公开写法。", {}),
        (
            "tool",
            "当前时间为 2026-04-05 20:00:00 中国标准时间+0800",
            {
                "source": "tool_result",
                "tool_name": "get_time",
                "tool_success": True,
                "tool_result_meta": {"tool": "get_time"},
            },
        ),
    ]

    for role, content, metadata in samples:
        memory_manager.record_message(
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata,
            persist=True,
        )

    recalled = memory_manager.build_structured_memory_prompt(
        session_id=session_id,
        limit=6,
    )

    print("\n" + "=" * 24)
    print("记忆闭环冒烟测试")
    print("=" * 24)
    print(memory_manager.build_memory_diagnostics(session_id) or "(没有决策日志)")
    print("\n长期保留裁剪：")
    print(memory_manager.build_retention_diagnostics(session_id) or "(没有触发裁剪)")
    print("\n召回解释：")
    print(
        memory_manager.build_recall_diagnostics(
            session_id=session_id,
            limit=6,
        )
        or "(没有召回解释)"
    )
    print("\n结构化记忆召回：")
    print(recalled or "(没有召回到记忆)")


def test_rag_workflow() -> None:
    """
    运行一个最小可用的 RAG 测试。

    流程：
    1. 先把本地示例文档切片并建立索引
    2. 再让 Agent 通过 `rag_tool` 检索文档回答问题
    """
    agent = build_react_agent()
    rag_file = ensure_demo_rag_document()

    index_question = (
        f"请使用 rag_tool 把本地文档 `{rag_file.as_posix()}` 加入索引，"
        "然后只回复“索引完成”。"
    )
    index_answer = agent.run(index_question)
    print_run_summary("RAG 测试 - 建索引", index_answer, agent.current_history)

    ask_question = "请使用 rag_tool 回答：helloAgent 当前支持哪些 Agent 范式？顺便说出文档里提到的饮品偏好。"
    ask_answer = agent.run(ask_question)
    print_run_summary("RAG 测试 - 检索问答", ask_answer, agent.current_history)


def test_rag_pipeline_smoke() -> None:
    """
    运行一个不依赖 LLM 的 RAG 冒烟测试。

    修改说明：上下文工程和向量库联调时，最常见的问题其实发生在“入库 / 检索 / 上下文拼装”阶段；
    这里单独提供一个直连 `RagPipeline` 的测试入口，方便在模型服务不可用时先验证底层链路。
    """
    memory_manager = build_memory_manager()
    rag_pipeline = build_rag_pipeline(memory_manager)
    rag_file = ensure_demo_rag_document()

    count = rag_pipeline.add_document(str(rag_file))
    matches = rag_pipeline.search("helloAgent 支持哪些 Agent 范式？顺便说出饮品偏好。", limit=3)
    answer_context = rag_pipeline.build_answer_context(
        query="helloAgent 支持哪些 Agent 范式？顺便说出饮品偏好。",
        matches=matches,
    )

    print("\n" + "=" * 24)
    print("RAG Pipeline 冒烟测试")
    print("=" * 24)
    print(f"Embedding 后端: {describe_embedding_backend(memory_manager)}")
    print(f"向量后端: {rag_pipeline.store.backend}")
    print(f"写入切片数: {count}")
    print(f"已索引来源: {', '.join(rag_pipeline.list_sources()) or '(空)'}")
    print("\n结构化上下文：")
    print(answer_context or "(没有生成上下文)")


def test_embedding_smoke() -> None:
    """
    运行一个 embedding 冒烟测试。

    修改说明：本地已有 Ollama embedding 模型时，可以先跑这个最小测试，
    确认向量生成和相似度趋势正常，再去联调记忆和 RAG。
    """
    memory_manager = build_memory_manager()
    embedding_service = memory_manager.embedding_service

    text_a = "用户喜欢美式咖啡，不喜欢过甜饮品。"
    text_b = "这个用户偏好黑咖啡，讨厌太甜的饮料。"
    text_c = "今天上海的天气比较晴朗。"

    vector_a = embedding_service.embed(text_a)
    vector_b = embedding_service.embed(text_b)
    vector_c = embedding_service.embed(text_c)

    similar_score = embedding_service.cosine_similarity(vector_a, vector_b)
    unrelated_score = embedding_service.cosine_similarity(vector_a, vector_c)

    print("\n" + "=" * 24)
    print("Embedding 冒烟测试")
    print("=" * 24)
    print(f"Embedding 后端: {describe_embedding_backend(memory_manager)}")
    print(f"向量维度: {len(vector_a)}")
    print(f"相近文本相似度: {similar_score:.4f}")
    print(f"无关文本相似度: {unrelated_score:.4f}")


def test_context_engineering_smoke() -> None:
    """
    运行一个上下文工程冒烟测试。

    修改说明：这个测试不依赖真实 LLM，只检查 Agent 在发请求前会拼出什么上下文，
    便于验证“记忆 / 自动 RAG / 工具观察”是否真的合并进了同一个 context packet。
    """
    memory_manager = build_memory_manager()
    session_id = "context_engineering_smoke"
    memory_manager.clear_session(session_id)
    memory_manager.record_message(
        session_id=session_id,
        role="user",
        content="我喜欢美式咖啡，不喜欢过甜饮品。",
    )

    rag_pipeline = build_rag_pipeline(memory_manager)
    rag_pipeline.add_document(str(ensure_demo_rag_document()))
    registry = build_tool_registry(memory_manager, session_id, rag_pipeline)

    agent = ReactAgent(
        name="context_engineering_smoke",
        llm=object(),  # type: ignore[arg-type]
        tool_registry=registry,
        config=Config.from_env(),
        max_steps=1,
        memory_manager=memory_manager,
        session_id=session_id,
    )
    agent._start_new_run("helloAgent 当前支持哪些 Agent 范式？顺便告诉我饮品偏好。")
    agent._remember_tool_observation("get_time", "当前本地时间为 2026-04-05 21:00:00。")

    rendered = agent._build_context_packet().render(
        max_chars=agent.config.context_max_chars,
        max_sections=agent.config.context_max_sections,
        section_max_chars=agent.config.context_section_max_chars,
    )

    print("\n" + "=" * 24)
    print("上下文工程冒烟测试")
    print("=" * 24)
    print(rendered or "(没有生成上下文)")


def test_context_routing_smoke() -> None:
    """
    运行一个上下文路由冒烟测试。

    修改说明：分别构造“偏记忆”和“偏知识检索”两类问题，
    方便直接观察路由策略是否会调整 memory / rag 的上下文优先级。
    """
    memory_manager = build_memory_manager()
    session_id = "context_routing_smoke"
    memory_manager.clear_session(session_id)
    memory_manager.record_message(
        session_id=session_id,
        role="user",
        content="我喜欢美式咖啡，不喜欢过甜饮品。",
    )
    memory_manager.record_message(
        session_id=session_id,
        role="assistant",
        content="helloAgent 当前支持 ReAct、Plan-and-Solve、Reflection。",
    )

    rag_pipeline = build_rag_pipeline(memory_manager)
    rag_pipeline.add_document(str(ensure_demo_rag_document()))
    registry = build_tool_registry(memory_manager, session_id, rag_pipeline)

    agent = ReactAgent(
        name="context_routing_smoke",
        llm=object(),  # type: ignore[arg-type]
        tool_registry=registry,
        config=Config.from_env(),
        max_steps=1,
        memory_manager=memory_manager,
        session_id=session_id,
    )

    questions = [
        "你还记得我的饮品偏好吗？",
        "helloAgent 当前支持哪些 Agent 范式？请根据文档回答。",
    ]
    for question in questions:
        agent._start_new_run(question)
        rendered = agent._build_context_packet().render(
            max_chars=agent.config.context_max_chars,
            max_sections=agent.config.context_max_sections,
            section_max_chars=agent.config.context_section_max_chars,
        )
        print("\n" + "=" * 24)
        print(f"上下文路由测试: {question}")
        print("=" * 24)
        print(rendered or "(没有生成上下文)")


def test_context_conflict_smoke() -> None:
    """
    运行一个上下文冲突消解冒烟测试。

    修改说明：这里故意制造“记忆”和“文档”里的相反说法，
    用来检查系统是否能把冲突显式标出来，并给出优先采用哪一侧的规则。
    """
    memory_manager = build_memory_manager()
    session_id = "context_conflict_smoke"
    memory_manager.clear_session(session_id)
    memory_manager.record_message(
        session_id=session_id,
        role="user",
        content="我喜欢美式咖啡，不喜欢过甜饮品。",
    )
    memory_manager.record_message(
        session_id=session_id,
        role="assistant",
        content="helloAgent 当前只支持 ReAct。",
    )

    conflict_path = Path("data/demo_conflict_notes.txt")
    conflict_path.parent.mkdir(parents=True, exist_ok=True)
    conflict_path.write_text(
        (
            "文档记录：用户不喜欢美式咖啡，更偏好拿铁。\n"
            "项目说明：helloAgent 当前支持 ReAct、Plan-and-Solve、Reflection 三种 Agent 范式。\n"
        ),
        encoding="utf-8",
    )

    rag_pipeline = build_rag_pipeline(memory_manager)
    rag_pipeline.clear()
    rag_pipeline.add_document(str(conflict_path))
    registry = build_tool_registry(memory_manager, session_id, rag_pipeline)

    agent = ReactAgent(
        name="context_conflict_smoke",
        llm=object(),  # type: ignore[arg-type]
        tool_registry=registry,
        config=Config.from_env(),
        max_steps=1,
        memory_manager=memory_manager,
        session_id=session_id,
    )
    agent._start_new_run("请根据文档说明核对饮品信息，并判断 helloAgent 支持哪些 Agent 范式。")
    rendered = agent._build_context_packet().render(
        max_chars=agent.config.context_max_chars,
        max_sections=agent.config.context_max_sections,
        section_max_chars=agent.config.context_section_max_chars,
    )

    print("\n" + "=" * 24)
    print("上下文冲突消解测试")
    print("=" * 24)
    print(rendered or "(没有生成上下文)")


def test_summary_smoke() -> None:
    """
    运行一个会话摘要冒烟测试。

    修改说明：这里构造一段更长的对话，确认系统能把多条记忆压缩成摘要层，
    而不是只把原始条目一股脑塞进上下文。
    """
    memory_manager = build_memory_manager()
    session_id = "summary_smoke"
    memory_manager.clear_session(session_id)
    seed_messages = [
        ("user", "我喜欢美式咖啡，不喜欢过甜饮品。"),
        ("assistant", "好的，我记住了你的饮品偏好。"),
        ("user", "helloAgent 目前重点在做记忆系统和上下文工程。"),
        ("assistant", "明白了，当前项目重点是记忆系统和上下文工程。"),
        ("user", "后面你回答时可以优先简洁一点。"),
        ("assistant", "收到，后续我会优先用更简洁的方式回答。"),
    ]
    for role, content in seed_messages:
        memory_manager.record_message(session_id=session_id, role=role, content=content)

    rag_pipeline = build_rag_pipeline(memory_manager)
    registry = build_tool_registry(memory_manager, session_id, rag_pipeline)
    agent = ReactAgent(
        name="summary_smoke",
        llm=object(),  # type: ignore[arg-type]
        tool_registry=registry,
        config=Config.from_env(),
        max_steps=1,
        memory_manager=memory_manager,
        session_id=session_id,
    )
    agent._start_new_run("你总结一下我当前的偏好和项目重点。")
    rendered = agent._build_context_packet().render(
        max_chars=agent.config.context_max_chars,
        max_sections=agent.config.context_max_sections,
        section_max_chars=agent.config.context_section_max_chars,
    )

    print("\n" + "=" * 24)
    print("会话摘要测试")
    print("=" * 24)
    print("单独摘要：")
    print(
        memory_manager.build_session_summary(
            session_id=session_id,
            query="你总结一下我当前的偏好和项目重点。",
            exclude_text="你总结一下我当前的偏好和项目重点。",
        )
        or "(没有生成摘要)"
    )
    print("\n注入后的上下文：")
    print(rendered or "(没有生成上下文)")


def test_native_tool_calling_smoke() -> None:
    """
    运行一个原生 tool calling 冒烟测试。

    修改说明：先用假 LLM 走一遍“schema -> tool_calls -> 工具执行 -> tool message 回填 -> 最终回答”，
    确认现有 Tool 定义已经真正能复用于原生 tool calling 主链路。
    """
    config = Config.from_env().model_copy(update={"tool_calling_mode": "native"})
    registry = ToolRegistry()
    registry.register_tool(GetTimeTool())

    agent = ReactAgent(
        name="native_tool_calling_smoke",
        llm=NativeToolCallingSmokeLLM(),  # type: ignore[arg-type]
        tool_registry=registry,
        config=config,
        max_steps=3,
    )
    answer = agent.run("请告诉我当前时间，并说明你是通过工具得到的。")
    print_run_summary("原生 Tool Calling 测试", answer, agent.current_history)


def test_native_plan_smoke() -> None:
    """
    运行一个 Plan-and-Solve 混合 tool calling 冒烟测试。

    修改说明：规划和最终汇总仍可保持文本输出，
    但步骤求解阶段已经可以走原生 tool calling。
    """
    config = Config.from_env().model_copy(update={"tool_calling_mode": "native"})
    registry = ToolRegistry()
    registry.register_tool(GetTimeTool())
    agent = PlanAndSolveAgent(
        name="native_plan_smoke",
        llm=NativePlanAndSolveSmokeLLM(),  # type: ignore[arg-type]
        tool_registry=registry,
        config=config,
        max_steps=3,
        max_step_rounds=3,
    )
    answer = agent.run("请先规划，再告诉我当前时间，并说明你如何得到它。")
    print_run_summary("Plan-and-Solve 原生 Tool Calling 测试", answer, agent.current_history)


def test_native_reflection_smoke() -> None:
    """
    运行一个 Reflection 混合 tool calling 冒烟测试。

    修改说明：当前先让“草稿生成”阶段走原生 tool calling，
    反思与修订阶段继续保留现有文本链路。
    """
    config = Config.from_env().model_copy(update={"tool_calling_mode": "native"})
    registry = ToolRegistry()
    registry.register_tool(GetTimeTool())
    agent = ReflectionAgent(
        name="native_reflection_smoke",
        llm=NativeReflectionSmokeLLM(),  # type: ignore[arg-type]
        tool_registry=registry,
        config=config,
        max_steps=3,
        max_reflections=2,
    )
    answer = agent.run("请结合工具告诉我当前时间，并确认你的答案是否完整。")
    print_run_summary("Reflection 原生 Tool Calling 测试", answer, agent.current_history)


def test_tool_schema_smoke() -> None:
    """
    运行一个工具 schema 冒烟测试。

    修改说明：先验证第一版 schema 骨架已经支持：
    1. 生成 function calling schema
    2. 把字符串参数归一化成目标类型
    3. 在非法枚举值时返回清晰错误
    """
    tool = SchemaSmokeTool()
    valid_parameters = tool.normalize_parameters(
        {
            "mode": "safe",
            "level": "3",
            "dry_run": "true",
            "payload": {
                "query": "本地时间",
                "options": {
                    "timezone": "Asia/Shanghai",
                    "include_seconds": "true",
                },
                "tags": ["clock", "demo"],
            },
        }
    )

    print("\n" + "=" * 24)
    print("Tool Schema 测试")
    print("=" * 24)
    print("生成的参数 Schema：")
    print(tool.get_parameters_schema())
    print("\n归一化后的参数：")
    print(valid_parameters)
    tool_result = tool.execute(valid_parameters)
    print("工具执行结果协议：")
    print(tool_result.model_dump())
    print("Observation 文本：")
    print(tool_result.render_for_observation())

    print("\n非法参数示例：")
    try:
        tool.normalize_parameters({"mode": "broken"})
    except ToolValidationError as exc:
        print(str(exc))
    else:
        print("未触发预期错误，请检查 schema 校验逻辑。")

    print("\n范围约束示例：")
    try:
        tool.normalize_parameters({"mode": "safe", "level": "9", "dry_run": "false"})
    except ToolValidationError as exc:
        print(str(exc))
    else:
        print("未触发预期错误，请检查范围校验逻辑。")

    print("\n跨字段语义校验示例：")
    try:
        tool.normalize_parameters(
            {
                "mode": "fast",
                "level": "1",
                "dry_run": "false",
                "payload": {"query": "时间"},
            }
        )
    except ToolValidationError as exc:
        print(str(exc))
    else:
        print("未触发预期错误，请检查跨字段校验逻辑。")

    print("\n复杂嵌套结构示例：")
    try:
        tool.normalize_parameters(
            {
                "mode": "safe",
                "level": "3",
                "dry_run": "false",
                "payload": {
                    "query": "x",
                    "options": {"include_seconds": "true"},
                    "tags": ["clock", 3],
                },
            }
        )
    except ToolValidationError as exc:
        print(str(exc))
    else:
        print("未触发预期错误，请检查嵌套 schema 校验逻辑。")

    print("\n条件 schema 示例：")
    try:
        tool.normalize_parameters({"mode": "fast", "level": "3", "dry_run": "false"})
    except ToolValidationError as exc:
        print(str(exc))
    else:
        print("未触发预期错误，请检查条件 schema 校验逻辑。")


def test_tool_recovery_smoke() -> None:
    """
    运行一个工具失败恢复冒烟测试。

    修改说明：这里重点验证两条链路：
    1. 可重试失败是否会被 Agent 自动重试
    2. 重试耗尽后是否会补充降级提示，而不是只返回生硬报错
    """
    config = Config.from_env().model_copy(
        update={
            "tool_calling_mode": "text",
            "tool_max_retries": 1,
            "tool_enable_graceful_degradation": True,
        }
    )
    registry = ToolRegistry()
    registry.register_tool(FlakyRecoveryTool())
    registry.register_tool(AlwaysFailTool())

    agent = ReactAgent(
        name="tool_recovery_smoke",
        llm=NativeToolCallingSmokeLLM(),  # type: ignore[arg-type]
        tool_registry=registry,
        config=config,
        max_steps=1,
    )
    agent._start_new_run("测试工具失败恢复链路。")

    agent.current_history.append("Action: flaky_recovery_tool[]")
    recovered = agent._handle_action("flaky_recovery_tool", "")
    agent._append_observation(recovered)

    agent.current_history.append("Action: always_fail_tool[]")
    degraded = agent._handle_action("always_fail_tool", "")
    agent._append_observation(degraded)

    summary = (
        f"恢复成功示例：{recovered}\n"
        f"降级结果示例：{degraded}\n"
        f"工具上下文：\n{agent._build_tool_observation_context()}"
    )
    print_run_summary("Tool Recovery 测试", summary, agent.current_history)


def run_demo(target: str = "reflection") -> None:
    """根据名称运行指定的示例，方便你在一个入口里切换不同 Agent。"""
    demos = {
        "react": test_react_agent,
        "plan": test_plan_and_solve_agent,
        "reflection": test_reflection_agent,
        "memory": test_memory_workflow,
        "memory_closure_smoke": test_memory_closure_smoke,
        "rag": test_rag_workflow,
        "rag_smoke": test_rag_pipeline_smoke,
        "embedding_smoke": test_embedding_smoke,
        "context_smoke": test_context_engineering_smoke,
        "routing_smoke": test_context_routing_smoke,
        "conflict_smoke": test_context_conflict_smoke,
        "summary_smoke": test_summary_smoke,
        "tool_schema_smoke": test_tool_schema_smoke,
        "tool_recovery_smoke": test_tool_recovery_smoke,
        "native_tool_smoke": test_native_tool_calling_smoke,
        "native_plan_smoke": test_native_plan_smoke,
        "native_reflection_smoke": test_native_reflection_smoke,
    }
    if target not in demos:
        raise ValueError(f"不支持的测试目标: {target}，可选值: {', '.join(demos)}")
    demos[target]()


def main() -> int:
    """main.py 的统一入口。"""
    configure_logging()

    try:
        # 修改说明：当前默认切到记忆测试，方便你直接验证多轮对话里的记忆写入与召回。
        run_demo("memory")
        return 0
    except Exception as exc:  # noqa: BLE001 - 演示入口需要统一捕获并给出简洁提示
        print_runtime_error(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
