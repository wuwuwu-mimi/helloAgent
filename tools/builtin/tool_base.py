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
    object_properties: List["ToolParameter"] = Field(default_factory=list)
    items_properties: List["ToolParameter"] = Field(default_factory=list)
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None


class ToolConditionalRule(BaseModel):
    """
    描述一条条件 schema 规则。

    修改说明：把“当 action=search 时 query 不能为空”这类规则抽成统一结构后，
    同一份约束就可以同时服务于：
    1. 本地参数校验
    2. 原生 tool calling schema 导出
    """

    field: str
    equals: Any
    required: List[str] = Field(default_factory=list)
    non_empty: List[str] = Field(default_factory=list)


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
            properties[parameter.name] = self._build_parameter_schema(parameter)
            if parameter.required:
                required.append(parameter.name)

        schema: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        conditional_rules = self.get_conditional_rules()
        all_of = self._build_conditional_schema_blocks(conditional_rules)
        if all_of:
            schema["allOf"] = all_of
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
        self._validate_conditional_rules(normalized)
        return self.validate_normalized_parameters(normalized)

    def _normalize_parameter_value(self, parameter: ToolParameter, value: Any) -> Any:
        """根据参数 schema 做最小可用的类型归一化。"""
        coerced = self._coerce_value(parameter, value)
        if parameter.choices and coerced not in parameter.choices:
            choices = ", ".join(str(item) for item in parameter.choices)
            raise ToolValidationError(
                f"Tool '{self.name}' parameter '{parameter.name}' must be one of: {choices}."
            )
        self._validate_parameter_constraints(parameter, coerced)
        return coerced

    def validate_normalized_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        提供给子类覆盖的语义校验入口。

        修改说明：基础 schema 只能描述“类型/范围”，像“action=add 时必须提供 path”
        这类跨字段规则更适合在这里集中处理。
        """
        return parameters

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
            if parameter.items_properties:
                item_parameter = ToolParameter(
                    name=f"{parameter.name}[]",
                    type="object",
                    description=parameter.description,
                    required=True,
                    object_properties=parameter.items_properties,
                )
                return [self._coerce_value(item_parameter, item) for item in value]
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
            if parameter.object_properties:
                return self._normalize_object_value(parameter, value)
            return value

        # 修改说明：先保底原样返回，后续如果要扩展 date / enum object 等类型，
        # 可以继续在这里补更细粒度的 schema 适配。
        return value

    def _validate_parameter_constraints(self, parameter: ToolParameter, value: Any) -> None:
        """校验单个参数的范围和长度约束。"""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if parameter.minimum is not None and value < parameter.minimum:
                raise ToolValidationError(
                    f"Tool '{self.name}' parameter '{parameter.name}' must be >= {parameter.minimum}."
                )
            if parameter.maximum is not None and value > parameter.maximum:
                raise ToolValidationError(
                    f"Tool '{self.name}' parameter '{parameter.name}' must be <= {parameter.maximum}."
                )

        if isinstance(value, str):
            if parameter.min_length is not None and len(value) < parameter.min_length:
                raise ToolValidationError(
                    f"Tool '{self.name}' parameter '{parameter.name}' length must be >= {parameter.min_length}."
                )
            if parameter.max_length is not None and len(value) > parameter.max_length:
                raise ToolValidationError(
                    f"Tool '{self.name}' parameter '{parameter.name}' length must be <= {parameter.max_length}."
                )
        if isinstance(value, list):
            if parameter.min_length is not None and len(value) < parameter.min_length:
                raise ToolValidationError(
                    f"Tool '{self.name}' parameter '{parameter.name}' item count must be >= {parameter.min_length}."
                )
            if parameter.max_length is not None and len(value) > parameter.max_length:
                raise ToolValidationError(
                    f"Tool '{self.name}' parameter '{parameter.name}' item count must be <= {parameter.max_length}."
                )

    def get_conditional_rules(self) -> List[ToolConditionalRule]:
        """提供给子类覆盖的条件 schema 规则入口。"""
        return []

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

    def _normalize_object_value(self, parameter: ToolParameter, value: Dict[str, Any]) -> Dict[str, Any]:
        """按嵌套 object schema 递归归一化对象参数。"""
        field_defs = {item.name: item for item in parameter.object_properties}
        unknown_keys = [key for key in value if key not in field_defs]
        if unknown_keys:
            joined = ", ".join(sorted(str(key) for key in unknown_keys))
            raise ToolValidationError(
                f"Tool '{self.name}' parameter '{parameter.name}' got unexpected nested keys: {joined}."
            )

        normalized: Dict[str, Any] = {}
        for item in parameter.object_properties:
            has_value = item.name in value and value[item.name] is not None
            if not has_value:
                if item.default is not None:
                    normalized[item.name] = deepcopy(item.default)
                    continue
                if item.required:
                    raise ToolValidationError(
                        f"Tool '{self.name}' parameter '{parameter.name}.{item.name}' is required."
                    )
                continue
            normalized[item.name] = self._normalize_parameter_value(item, value[item.name])
        return normalized

    def _build_parameter_schema(self, parameter: ToolParameter) -> Dict[str, Any]:
        """递归生成单个参数的 schema。"""
        field_schema: Dict[str, Any] = {
            "type": parameter.type,
            "description": parameter.description,
        }
        if parameter.choices:
            field_schema["enum"] = list(parameter.choices)
        if parameter.type == "array":
            if parameter.items_properties:
                field_schema["items"] = self._build_object_schema(parameter.items_properties)
            elif parameter.items_type:
                field_schema["items"] = {"type": parameter.items_type}
        if parameter.type == "object" and parameter.object_properties:
            field_schema.update(self._build_object_schema(parameter.object_properties))
        if parameter.minimum is not None:
            field_schema["minimum"] = parameter.minimum
        if parameter.maximum is not None:
            field_schema["maximum"] = parameter.maximum
        if parameter.min_length is not None:
            key = "minItems" if parameter.type == "array" else "minLength"
            field_schema[key] = parameter.min_length
        if parameter.max_length is not None:
            key = "maxItems" if parameter.type == "array" else "maxLength"
            field_schema[key] = parameter.max_length
        if parameter.default is not None:
            field_schema["default"] = parameter.default
        return field_schema

    def _build_object_schema(self, properties: List[ToolParameter]) -> Dict[str, Any]:
        """生成 object 类型参数的递归 schema。"""
        property_map: Dict[str, Any] = {}
        required_fields: List[str] = []
        for child in properties:
            property_map[child.name] = self._build_parameter_schema(child)
            if child.required:
                required_fields.append(child.name)

        schema: Dict[str, Any] = {
            "type": "object",
            "properties": property_map,
            "additionalProperties": False,
        }
        if required_fields:
            schema["required"] = required_fields
        return schema

    def _build_conditional_schema_blocks(self, rules: List[ToolConditionalRule]) -> List[Dict[str, Any]]:
        """把条件规则导出成 JSON schema `if/then` 结构。"""
        blocks: List[Dict[str, Any]] = []
        parameter_defs = {item.name: item for item in self.get_parameters()}
        for rule in rules:
            if "." in rule.field:
                continue
            required_fields = [field for field in rule.required if "." not in field]
            non_empty_fields = [field for field in rule.non_empty if "." not in field]
            if not required_fields and not non_empty_fields:
                continue
            then_block: Dict[str, Any] = {}
            merged_required = list(dict.fromkeys(required_fields + non_empty_fields))
            if merged_required:
                then_block["required"] = merged_required

            property_overrides: Dict[str, Any] = {}
            for field in non_empty_fields:
                parameter = parameter_defs.get(field)
                if parameter is None:
                    continue
                if parameter.type == "string":
                    property_overrides[field] = {"minLength": max(parameter.min_length or 0, 1)}
                elif parameter.type == "array":
                    property_overrides[field] = {"minItems": max(parameter.min_length or 0, 1)}
            if property_overrides:
                then_block["properties"] = property_overrides
            blocks.append(
                {
                    "if": {
                        "properties": {
                            rule.field: {
                                "const": rule.equals,
                            }
                        },
                        "required": [rule.field],
                    },
                    "then": then_block,
                }
            )
        return blocks

    def _validate_conditional_rules(self, parameters: Dict[str, Any]) -> None:
        """执行本地条件 schema 校验。"""
        for rule in self.get_conditional_rules():
            matched, actual_value = self._read_field(parameters, rule.field)
            if not matched or actual_value != rule.equals:
                continue

            for field in rule.required:
                exists, _ = self._read_field(parameters, field)
                if not exists:
                    raise ToolValidationError(
                        f"Tool '{self.name}' requires parameter '{field}' when {rule.field}={rule.equals}."
                    )

            for field in rule.non_empty:
                exists, current = self._read_field(parameters, field)
                if not exists or self._is_empty_value(current):
                    raise ToolValidationError(
                        f"Tool '{self.name}' requires non-empty '{field}' when {rule.field}={rule.equals}."
                    )

    @staticmethod
    def _read_field(parameters: Dict[str, Any], field: str) -> tuple[bool, Any]:
        """支持使用 `a.b.c` 形式读取嵌套字段。"""
        current: Any = parameters
        for part in field.split("."):
            if not isinstance(current, dict) or part not in current:
                return False, None
            current = current[part]
        return True, current

    @staticmethod
    def _is_empty_value(value: Any) -> bool:
        """判断一个值是否可视为“空”。"""
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, dict, tuple, set)):
            return len(value) == 0
        return False


ToolParameter.model_rebuild()
