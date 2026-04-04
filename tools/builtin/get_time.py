from __future__ import annotations

from datetime import datetime


def get_time() -> str:
    """
    返回当前本地时间。

    这里直接读取运行当前 Python 进程所在机器的本地时区时间，
    适合作为最小工具示例，方便测试 ReAct 的“调用工具 -> 观察结果 -> 给出答案”链路。
    """
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
