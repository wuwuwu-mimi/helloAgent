import logging

from agents.plan_and_solve import PlanAndSolveAgent
from agents.react_agent import ReactAgent
from agents.reflection_agent import ReflectionAgent
from core import Config, HelloAgentsLLM
from tools.builtin.get_time import GetTimeTool
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


def build_tool_registry() -> ToolRegistry:
    """
    创建当前示例使用的工具注册表。

    目前三个 Agent 都共用同一套工具，
    这样便于直接横向对比不同范式的行为差异。
    """
    registry = ToolRegistry()
    registry.register_tool(GetTimeTool())
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
    return ReactAgent(
        name="react_agent",
        llm=llm,
        tool_registry=build_tool_registry(),
        config=config,
        max_steps=5,
    )


def build_plan_and_solve_agent() -> PlanAndSolveAgent:
    """构造一个可直接运行的 Plan-and-Solve Agent。"""
    llm, config = _build_llm_and_config()
    return PlanAndSolveAgent(
        name="plan_and_solve_agent",
        llm=llm,
        tool_registry=build_tool_registry(),
        config=config,
        max_steps=5,
        max_step_rounds=4,
    )


def build_reflection_agent() -> ReflectionAgent:
    """构造一个可直接运行的 Reflection Agent。"""
    llm, config = _build_llm_and_config()
    return ReflectionAgent(
        name="reflection_agent",
        llm=llm,
        tool_registry=build_tool_registry(),
        config=config,
        max_steps=5,
        max_reflections=2,
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


def run_demo(target: str = "reflection") -> None:
    """根据名称运行指定的示例，方便你在一个入口里切换不同 Agent。"""
    demos = {
        "react": test_react_agent,
        "plan": test_plan_and_solve_agent,
        "reflection": test_reflection_agent,
    }
    if target not in demos:
        raise ValueError(f"不支持的测试目标: {target}，可选值: {', '.join(demos)}")
    demos[target]()


def main() -> int:
    """main.py 的统一入口。"""
    configure_logging()

    try:
        # 修改说明：默认先跑 ReflectionAgent，切换其他 Agent 只需要改这里的名称。
        run_demo("reflection")
        return 0
    except Exception as exc:  # noqa: BLE001 - 演示入口需要统一捕获并给出简洁提示
        print_runtime_error(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
