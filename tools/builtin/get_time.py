from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from tools.builtin.tool_base import Tool, ToolParameter


def get_time() -> str:
    """返回当前机器本地时区对应的时间字符串。"""
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z%z")


class GetTimeTool(Tool):
    """最小示例工具：获取当前本地时间。"""

    def __init__(self) -> None:
        super().__init__(
            name="get_time",
            description="获取当前本地时间。",
        )

    def run(self, parameters: Dict[str, Any]) -> str:
        """
        当前工具不需要参数。

        这里仍然保留 `parameters` 入参，是为了保持所有工具接口一致，
        方便注册器和 Agent 用统一方式调用。
        """
        del parameters
        return get_time()

    def get_parameters(self) -> List[ToolParameter]:
        """当前工具没有参数，因此返回空列表。"""
        return []