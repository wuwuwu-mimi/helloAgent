import logging

from agents.react_agent import ReactAgent
from core import Config, HelloAgentsLLM
from tools.builtin.get_time import get_time
from tools.builtin.toolRegistry import ToolRegistry


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


def build_agent() -> ReactAgent:
    """
    创建一个带 `get_time` 工具的 ReactAgent。

    这里集中做两件事：
    1. 初始化模型客户端
    2. 注册当前 agent 可以使用的工具
    后面你继续加工具时，直接在这里扩展即可。
    """
    config = Config.from_env()
    llm = HelloAgentsLLM.from_config(config)

    # 统一在注册表里声明工具，方便后续继续增加更多工具。
    tool_registry = ToolRegistry()
    tool_registry.registerTool(
        name="get_time",
        description="获取当前本地时间。",
        func=get_time,
        usage="get_time[]，不需要参数",
    )

    # 这里返回一个最小可运行的 ReAct Agent，先保证主链路跑通。
    return ReactAgent(
        name="react_agent",
        llm=llm,
        tool_registry=tool_registry,
        config=config,
        max_steps=5,
    )


def print_run_summary(agent: ReactAgent, question: str, answer: str) -> None:
    """
    统一打印测试结果。

    这样终端输出会更清楚：
    - 先看问题
    - 再看最终答案
    - 最后逐条看 Thought / Action / Observation 历史
    """
    print("\n" + "=" * 24)
    print("测试问题")
    print("=" * 24)
    print(question)

    print("\n" + "=" * 24)
    print("最终回答")
    print("=" * 24)
    print(answer)

    print("\n" + "=" * 24)
    print("执行历史")
    print("=" * 24)
    if not agent.current_history:
        print("(无历史记录)")
        return

    # 给每条历史加序号，便于你快速定位是哪一步出了问题。
    for index, item in enumerate(agent.current_history, start=1):
        print(f"{index:02d}. {item}")


def test_get_time_tool() -> None:
    """
    用真实 LLM 跑一个最小示例。

    预期流程：
    1. 模型识别需要查询当前时间
    2. 模型输出 `Action: get_time[]`
    3. Agent 执行工具，把结果写回 Observation
    4. 模型根据 Observation 输出最终答案
    """
    agent = build_agent()

    # 问题里明确提示模型可以使用 get_time，便于第一次测试时更稳定触发工具。
    question = "请告诉我现在几点。你可以使用 get_time 工具，最后用中文直接回答。"

    print("开始测试 ReactAgent + get_time ...")
    answer = agent.run(question)
    print_run_summary(agent, question, answer)


if __name__ == "__main__":
    test_get_time_tool()
