from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolValidationError(ValueError):
    """工具参数不符合 schema 时抛出的统一异常。"""


class ToolResult(BaseModel):
    """
    统一的工具执行结果协议。

    修改说明：先把工具结果收敛成 `success / content / data / error / meta` 五个核心字段，
    这样 Agent、日志、记忆和后续失败恢复逻辑都能围绕同一份结果对象协作。
    """

    success: bool = True
    content: str = ""
    data: Any = None
    error: str = ""
    meta: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        content: str = "",
        *,
        data: Any = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        """构造一个成功结果。"""
        return cls(success=True, content=content, data=data, meta=meta or {})

    @classmethod
    def fail(
        cls,
        error: str,
        *,
        content: str = "",
        data: Any = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        """构造一个失败结果。"""
        return cls(success=False, content=content, data=data, error=error, meta=meta or {})

    def render_for_observation(self) -> str:
        """把工具结果渲染成适合回填给 Agent 的 Observation 文本。"""
        if self.success:
            return self.content or "Tool returned empty output."
        if self.error and self.content:
            return f"{self.error}\n{self.content}"
        return self.error or self.content or "Tool execution failed."


class ToolParameter(BaseModel):
    """
    定义工具参数，便于统一生成说明文本和参数 Schema。

    修改说明：第一版 schema 骨架除了基础类型外，再补上 choices / items_type，
    这样后面接原生 tool calling 时，模型和本地执行器都能共享同一份约束。
    """

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    choices: List[Any] = Field(default_factory=list)
    items_type: Optional[str] = None


class Tool(ABC):
    """所有工具的抽象基类。"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> Any:
        """执行工具，并返回字符串或 `ToolResult`。"""

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """返回工具参数定义列表。"""

    def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        统一执行工具并返回 `ToolResult`。

        修改说明：保留各工具的 `run()` 简单接口，但真正给 Agent 使用时统一走 `execute()`，
        这样老工具不用一次性全部重写，新老返回格式也能平滑兼容。
        """
        try:
            return self._coerce_result(self.run(parameters))
        except ToolValidationError:
            raise
        except Exception as exc:  # noqa: BLE001 - 工具异常统一折叠到 ToolResult
            return self._build_exception_result(exc)

    def get_parameters_schema(self) -> Dict[str, Any]:
        """
        生成兼容 function calling 的参数 Schema。

        这样注册器和 Agent 不需要自己再拼装一遍参数描述，
        后面如果切到原生 tool calling，也可以直接复用。
        """
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for parameter in self.get_parameters():
            field_schema: Dict[str, Any] = {
                "type": parameter.type,
                "description": parameter.description,
            }
            if parameter.choices:
                field_schema["enum"] = list(parameter.choices)
            if parameter.type == "array" and parameter.items_type:
                field_schema["items"] = {"type": parameter.items_type}
            if parameter.default is not None:
                field_schema["default"] = parameter.default

            properties[parameter.name] = field_schema
            if parameter.required:
                required.append(parameter.name)

        schema: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        return schema

    def normalize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        按工具 schema 校验并归一化参数。

        修改说明：先把“缺参 / 多参 / 类型不匹配 / 枚举不合法”统一拦在工具执行前，
        这样 Agent 层只需要准备参数形状，不需要了解每个工具自己的细节。
        """
        if not isinstance(parameters, dict):
            raise ToolValidationError(f"Tool '{self.name}' expects an object-like parameter mapping.")

        parameter_defs = {item.name: item for item in self.get_parameters()}
        unknown_keys = [key for key in parameters if key not in parameter_defs]
        if unknown_keys:
            joined = ", ".join(sorted(str(key) for key in unknown_keys))
            raise ToolValidationError(f"Tool '{self.name}' got unexpected parameters: {joined}.")

        normalized: Dict[str, Any] = {}
        for item in self.get_parameters():
            has_value = item.name in parameters and parameters[item.name] is not None
            if not has_value:
                if item.default is not None:
                    normalized[item.name] = deepcopy(item.default)
                    continue
                if item.required:
                    raise ToolValidationError(f"Tool '{self.name}' requires parameter '{item.name}'.")
                continue

            normalized[item.name] = self._normalize_parameter_value(item, parameters[item.name])
        return normalized

    def _normalize_parameter_value(self, parameter: ToolParameter, value: Any) -> Any:
        """根据参数 schema 做最小可用的类型归一化。"""
        coerced = self._coerce_value(parameter, value)
        if parameter.choices and coerced not in parameter.choices:
            choices = ", ".join(str(item) for item in parameter.choices)
            raise ToolValidationError(
                f"Tool '{self.name}' parameter '{parameter.name}' must be one of: {choices}."
            )
        return coerced

    def _coerce_value(self, parameter: ToolParameter, value: Any) -> Any:
        """把常见字符串输入尽量转成 schema 约定的 Python 类型。"""
        target_type = parameter.type.strip().lower()

        if target_type == "string":
            if isinstance(value, (dict, list)):
                raise ToolValidationError(
                    f"Tool '{self.name}' parameter '{parameter.name}' expects a string."
                )
            return str(value)

        if target_type == "integer":
            if isinstance(value, bool):
                raise ToolValidationError(
                    f"Tool '{self.name}' parameter '{parameter.name}' expects an integer, got boolean."
                )
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str):
                text = value.strip()
                if text:
                    try:
                        return int(text)
                    except ValueError as exc:
                        raise ToolValidationError(
                            f"Tool '{self.name}' parameter '{parameter.name}' expects an integer."
                        ) from exc
            raise ToolValidationError(
                f"Tool '{self.name}' parameter '{parameter.name}' expects an integer."
            )

        if target_type == "number":
            if isinstance(value, bool):
                raise ToolValidationError(
                    f"Tool '{self.name}' parameter '{parameter.name}' expects a number, got boolean."
                )
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                text = value.strip()
                if text:
                    try:
                        return float(text)
                    except ValueError as exc:
                        raise ToolValidationError(
                            f"Tool '{self.name}' parameter '{parameter.name}' expects a number."
                        ) from exc
            raise ToolValidationError(
                f"Tool '{self.name}' parameter '{parameter.name}' expects a number."
            )

        if target_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes", "y", "on"}:
                    return True
                if normalized in {"false", "0", "no", "n", "off"}:
                    return False
            raise ToolValidationError(
                f"Tool '{self.name}' parameter '{parameter.name}' expects a boolean."
            )

        if target_type == "array":
            if not isinstance(value, list):
                raise ToolValidationError(
                    f"Tool '{self.name}' parameter '{parameter.name}' expects an array."
                )
            if not parameter.items_type:
                return value
            item_parameter = ToolParameter(
                name=f"{parameter.name}[]",
                type=parameter.items_type,
                description=parameter.description,
                required=True,
            )
            return [self._coerce_value(item_parameter, item) for item in value]

        if target_type == "object":
            if not isinstance(value, dict):
                raise ToolValidationError(
                    f"Tool '{self.name}' parameter '{parameter.name}' expects an object."
                )
            return value

        # 修改说明：先保底原样返回，后续如果要扩展 date / enum object 等类型，
        # 可以继续在这里补更细粒度的 schema 适配。
        return value

    def format_for_prompt(self) -> str:
        """
        生成适合直接写进 Prompt 的工具说明文本。

        文本版 ReAct 主要依赖这段说明来决定该怎么写 Action。
        """
        parameters = self.get_parameters()
        if not parameters:
            return f"- {self.name}: {self.description} 用法: {self.name}[]"

        parameter_text = ", ".join(
            (
                f"{item.name}<{item.type}>{' 必填' if item.required else ' 可选'}: {item.description}"
                + (f" 可选值={item.choices}" if item.choices else "")
            )
            for item in parameters
        )
        return f"- {self.name}: {self.description} 参数: {parameter_text}"

    def _coerce_result(self, result: Any) -> ToolResult:
        """把旧工具返回的字符串 / 字典 / ToolResult 统一包装成 `ToolResult`。"""
        if isinstance(result, ToolResult):
            return result

        if result is None:
            return ToolResult.ok("")

        if isinstance(result, str):
            return ToolResult.ok(result)

        if isinstance(result, dict):
            success = bool(result.get("success", True))
            content = str(result.get("content", "") or "")
            error = str(result.get("error", "") or "")
            data = result.get("data")
            meta = result.get("meta")
            if isinstance(meta, dict):
                payload_meta = meta
            else:
                payload_meta = {}
            return ToolResult(
                success=success,
                content=content,
                error=error,
                data=data,
                meta=payload_meta,
            )

        return ToolResult.ok(str(result), data=result)

    def _build_exception_result(self, exc: Exception) -> ToolResult:
        """
        把工具异常包装成失败结果。

        修改说明：这样 Agent 在做重试 / 降级时，不需要同时兼容“异常”和“失败结果”两套分支。
        """
        return ToolResult.fail(
            f"{type(exc).__name__}: {exc}",
            meta={
                "tool": self.name,
                "retryable": self._is_retryable_exception(exc),
                "exception_type": type(exc).__name__,
                "failure_stage": "tool_execute",
            },
        )

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        """粗略判断一个工具异常是否适合自动重试。"""
        return isinstance(exc, (TimeoutError, ConnectionError, OSError))
