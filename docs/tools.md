# 工具模块说明

修改说明：这个文档聚焦 `tools/` 目录，说明工具基类、注册器以及当前内置工具的职责。

## 1. `tools/__init__.py`
- 作用：工具包标记文件。

## 2. `tools/builtin/tool_base.py`
- 作用：定义统一的工具抽象、参数 schema 和结果协议。

### `ToolValidationError`
- 作用：表示工具参数校验失败的统一异常。

### `ToolResult`
- 作用：统一表示工具执行结果，当前包含 `success / content / data / error / meta` 五个核心字段。
- `ok`：构造成功结果。
- `fail`：构造失败结果。
- `render_for_observation`：把结果渲染成适合回填给 Agent 的 observation 文本。

### `ToolParameter`
- 作用：描述单个工具参数，包括类型、默认值、枚举、范围和长度约束。

### `Tool`
- 作用：所有工具的抽象父类。
- `__init__`：初始化工具名称和描述。
- `run`：工具自身的执行逻辑。
- `get_parameters`：返回参数定义列表。
- `execute`：统一执行工具并折叠成 `ToolResult`。
- `get_parameters_schema`：生成 function calling 兼容 schema。
- `normalize_parameters`：按 schema 对参数做归一化。
- `_normalize_parameter_value`：归一化单个参数值。
- `validate_normalized_parameters`：做更高层的语义校验。
- `_coerce_value`：把输入转成目标类型。
- `_validate_parameter_constraints`：校验枚举、范围、长度等约束。
- `format_for_prompt`：把工具说明渲染给文本版 Agent 使用。
- `_coerce_result`：把工具原始返回值转成 `ToolResult`。
- `_build_exception_result`：把异常转成失败结果。
- `_is_retryable_exception`：判断异常是否可能值得重试。

## 3. `tools/builtin/toolRegistry.py`
- 作用：统一管理工具注册、获取和导出。

### `ToolRegistry`
- 作用：保存当前 Agent 可用的全部工具。
- `__init__`：初始化内部工具表。
- `register_tool`：注册工具。
- `registerTool`：兼容旧命名接口。
- `get_tool`：按名称获取工具。
- `getTool`：兼容旧命名接口。
- `list_tools`：列出全部已注册工具。
- `get_available_tools`：导出给原生 tool calling 使用的 schema 列表。
- `getAvailableTools`：兼容旧命名接口。
- `describe_tools`：把全部工具渲染成文本说明。

## 4. `tools/builtin/get_time.py`
- 作用：当前最简单的内置示例工具，用来获取当前本地时间。

### `GetTimeTool`
- `__init__`：初始化工具名称和说明。
- `run`：返回当前时间文本。
- `get_parameters`：声明该工具不需要参数。

### 函数
- `get_time`：返回当前系统时间字符串。

## 5. `tools/builtin/memory_tool.py`
- 作用：把记忆系统以工具形式暴露给 Agent。

### `MemoryTool`
- 作用：提供显式读写记忆的能力。
- `__init__`：绑定 `MemoryManager` 和当前 session。
- `run`：根据 action 执行不同记忆操作。
- `get_parameters`：定义 `action / query / content / limit` 等参数 schema。
- `validate_normalized_parameters`：检查 action 和参数组合是否合法。
- `_format_items`：把记忆条目渲染成文本。

### 支持的 action
- `recent`：查看最近记忆。
- `search`：按 query 搜索记忆。
- `context`：返回结构化记忆上下文。
- `summary`：返回会话摘要。
- `remember`：手动写入一条记忆。
- `clear`：清空当前会话记忆。

## 6. `tools/builtin/rag_tool.py`
- 作用：把本地 RAG 管道封装成工具能力。

### `RagTool`
- 作用：提供文档入库、检索和上下文生成能力。
- `__init__`：绑定 `RagPipeline`。
- `run`：根据 action 执行不同 RAG 操作。
- `get_parameters`：定义 `action / path / query / limit` 等参数 schema。
- `validate_normalized_parameters`：检查 action 与参数组合是否合法。

### 支持的 action
- `add`：把本地文档加入索引。
- `search`：返回最相关的文档片段。
- `answer`：返回基于检索结果的参考回答上下文。
- `context`：返回更适合 prompt 拼装的结构化检索上下文。
- `clear`：清空当前 RAG 索引。

## 7. 工具系统在项目中的作用

### 对文本版 ReAct
- 工具描述会通过 `ToolRegistry.describe_tools()` 注入 prompt
- 模型输出 `Action: tool_name[...]`
- Agent 再通过工具系统解析、执行、回填 observation

### 对原生 tool calling
- 工具 schema 由 `Tool.get_parameters_schema()` 导出
- `ToolRegistry.get_available_tools()` 直接组装模型所需的 `tools`
- 工具执行结果统一折叠成 `ToolResult`

### 对记忆与调试
- `ToolResult.meta / data` 会进入 observation、日志和工具记忆
- 工具失败时支持自动重试与降级提示
- 工具结果已经成为记忆闭环里的重要输入之一
