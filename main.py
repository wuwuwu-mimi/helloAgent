import logging

from agents.plan_and_solve import PlanAndSolveAgent
from agents.react_agent import ReactAgent
from agents.reflection_agent import ReflectionAgent
from core import Config, HelloAgentsLLM
from memory import MemoryConfig, MemoryManager
from tools.builtin.get_time import GetTimeTool
from tools.builtin.memory_tool import MemoryTool
from tools.builtin.toolRegistry import ToolRegistry


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

    修改说明：当前先使用“工作记忆 + SQLite 情景记忆”的最小实现，
    后续如果要切到向量库或图数据库，可以在这里统一替换。
    """
    return MemoryManager(MemoryConfig())


def build_tool_registry(memory_manager: MemoryManager, session_id: str) -> ToolRegistry:
    """
    创建当前示例使用的工具注册表。

    修改说明：工具注册表现在不仅包含 `get_time`，
    还会注入 `memory_tool`，方便 Agent 显式读取或写入会话记忆。
    """
    registry = ToolRegistry()
    registry.register_tool(GetTimeTool())
    registry.register_tool(MemoryTool(memory_manager=memory_manager, session_id=session_id))
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
    return ReactAgent(
        name="react_agent",
        llm=llm,
        tool_registry=build_tool_registry(memory_manager, "react_agent"),
        config=config,
        max_steps=5,
        memory_manager=memory_manager,
        session_id="react_agent",
    )


def build_plan_and_solve_agent() -> PlanAndSolveAgent:
    """构造一个可直接运行的 Plan-and-Solve Agent。"""
    llm, config = _build_llm_and_config()
    memory_manager = build_memory_manager()
    return PlanAndSolveAgent(
        name="plan_and_solve_agent",
        llm=llm,
        tool_registry=build_tool_registry(memory_manager, "plan_and_solve_agent"),
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
    return ReflectionAgent(
        name="reflection_agent",
        llm=llm,
        tool_registry=build_tool_registry(memory_manager, "reflection_agent"),
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


def run_demo(target: str = "reflection") -> None:
    """根据名称运行指定的示例，方便你在一个入口里切换不同 Agent。"""
    demos = {
        "react": test_react_agent,
        "plan": test_plan_and_solve_agent,
        "reflection": test_reflection_agent,
        "memory": test_memory_workflow,
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
