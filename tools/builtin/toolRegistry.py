from typing import Any, Callable, Dict, List, Optional


class ToolRegistry:
    """维护工具注册信息，并向 agent 提供统一查询入口。"""

    def __init__(self):
        # 每个工具都保存在一个简单字典里，后面如果要扩展 schema，
        # 可以继续在这里追加字段，而不用改 Agent 主流程。
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(
        self,
        name: str,
        description: str,
        func: Callable,
        usage: Optional[str] = None,
    ) -> None:
        """
        注册工具。

        参数说明：
        - name: 工具名，供模型在 Action 中引用
        - description: 工具用途说明
        - func: 实际执行的 Python 函数
        - usage: 可选，用自然语言补充调用方式，方便直接展示给模型
        """
        if name in self.tools:
            # 当前策略是允许覆盖注册，这样调试工具时不需要手动先删除。
            print(f"Tool {name} already registered")

        self.tools[name] = {
            "description": description,
            "func": func,
            "usage": usage,
        }
        print(f"Tool {name} registered")

    def getTool(self, name: str) -> Optional[Callable]:
        """根据工具名获取对应的可调用函数。"""
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> List[dict[str, Any]] | None:
        """
        获取所有可用工具的简化描述。

        这里先返回一个接近 function calling 的结构，
        但暂时只提供 name / description，便于文本版 ReAct 复用。
        """
        tools = []

        for name, info in self.tools.items():
            # usage 会被拼到描述里，让模型更容易知道该怎么写 Action。
            description = info["description"]
            if info.get("usage"):
                description = f"{description} 用法: {info['usage']}"

            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                }
            })
        return tools
