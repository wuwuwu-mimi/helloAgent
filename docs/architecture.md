# 架构说明

修改说明：这个文档聚焦项目整体结构、根目录文件、`core/` 基础设施，以及模块之间的依赖关系。

## 1. 项目整体结构

当前仓库可以理解成 5 层：

1. `main.py`
   - 统一演示入口
   - 负责把 Agent、工具、记忆、RAG 和测试样例串起来
2. `agents/`
   - 放各种 Agent 范式实现
   - 当前主要有 `React / Plan-and-Solve / Reflection`
3. `tools/`
   - 放工具基类、注册器和内置工具
4. `memory/`
   - 放记忆系统、embedding、RAG、向量存储和图谱存储
5. `core/`
   - 放配置、消息结构、LLM 客户端和上下文工程抽象

## 2. 根目录文件

### `README.md`
- 作用：项目对外说明，面向 GitHub 读者介绍能力、运行方式和限制。

### `PROJECT_DOC.md`
- 作用：根目录文档索引，帮助快速跳转到 `docs/` 下的细分文档。

### `TODO.md`
- 作用：记录当前版本的剩余事项，当前重点仍是把 `P0` 收完整。

### `LICENSE`
- 作用：开源许可证文件。

### `requestsment.txt`
- 作用：依赖占位文件，后续如果要整理发布，可以改成正式依赖清单。

### `main.py`
- 作用：项目的统一运行入口和演示入口。

#### `main.py` 中的主要类

##### `NativeToolCallingSmokeLLM`
- 作用：用假 LLM 模拟原生 tool calling 主链路。
- `chat`：在没有 tool message 时返回 `tool_calls`，有工具结果后返回最终答案。

##### `NativePlanAndSolveSmokeLLM`
- 作用：模拟 Plan-and-Solve 的规划、步骤求解和汇总过程。
- `chat`：根据上下文返回计划、工具调用或步骤结果。

##### `NativeReflectionSmokeLLM`
- 作用：模拟 Reflection Agent 的草稿、审查与修订阶段。
- `chat`：根据 prompt 所处阶段返回不同结构的假结果。

##### `SchemaSmokeTool`
- 作用：演示工具参数 schema、归一化和跨字段校验。
- `__init__`：初始化测试工具元信息。
- `run`：输出归一化后的参数结果。
- `get_parameters`：定义测试参数。
- `validate_normalized_parameters`：演示更高层语义校验。

##### `FlakyRecoveryTool`
- 作用：演示工具自动重试。
- `__init__`：初始化调用计数。
- `run`：第一次失败、第二次成功。
- `get_parameters`：声明无参数。

##### `AlwaysFailTool`
- 作用：演示失败后的降级提示。
- `__init__`：初始化工具信息。
- `run`：始终失败。
- `get_parameters`：声明无参数。

#### `main.py` 中的主要函数
- `configure_logging`：统一日志配置。
- `build_memory_manager`：创建记忆管理器。
- `describe_embedding_backend`：输出当前 embedding 后端说明。
- `build_rag_pipeline`：构建 RAG 管道。
- `build_tool_registry`：注册默认内置工具。
- `_build_llm_and_config`：统一构建 LLM 客户端与配置。
- `build_react_agent`：构建 ReAct Agent。
- `build_plan_and_solve_agent`：构建 Plan-and-Solve Agent。
- `build_reflection_agent`：构建 Reflection Agent。
- `print_run_summary`：打印最终答案与执行轨迹。
- `print_memory_snapshot`：打印记忆快照。
- `print_runtime_error`：打印简洁错误信息。
- `ensure_demo_rag_document`：生成本地示例文档。
- `test_react_agent`：ReAct 演示。
- `test_plan_and_solve_agent`：Plan-and-Solve 演示。
- `test_reflection_agent`：Reflection 演示。
- `test_memory_workflow`：多轮记忆演示。
- `test_memory_closure_smoke`：记忆闭环冒烟测试。
- `test_rag_workflow`：RAG 全链路演示。
- `test_rag_pipeline_smoke`：RAG 底层冒烟测试。
- `test_embedding_smoke`：embedding 冒烟测试。
- `test_context_engineering_smoke`：上下文工程冒烟测试。
- `test_context_routing_smoke`：上下文路由测试。
- `test_context_conflict_smoke`：冲突消解测试。
- `test_summary_smoke`：摘要测试。
- `test_native_tool_calling_smoke`：原生 tool calling 测试。
- `test_native_plan_smoke`：Plan-and-Solve 混合原生工具测试。
- `test_native_reflection_smoke`：Reflection 混合原生工具测试。
- `test_tool_schema_smoke`：schema 测试。
- `test_tool_recovery_smoke`：重试与降级测试。
- `run_demo`：按名称运行 demo。
- `main`：默认 CLI 入口。

## 3. `core/` 目录

### `core/__init__.py`
- 作用：集中导出核心对象，方便其他模块统一导入。

### `core/Config.py`
- 作用：统一管理运行配置。

#### `Config`
- 作用：项目全局配置对象。
- `from_env`：从环境变量构建配置。
- `llm_options`：导出模型调用所需参数。
- `trimmed_history`：裁剪历史消息。
- `to_dict`：转换为字典。

#### 辅助函数
- `_first_env`：读取首个有效环境变量。
- `_read_bool`：解析布尔值。
- `_read_float`：解析浮点数。
- `_read_int`：解析整数。

### `core/context_engineering.py`
- 作用：统一上下文工程抽象。

#### `ContextSection`
- 作用：描述一个带标题、来源和优先级的上下文片段。
- `render`：渲染为文本。

#### `ContextPacket`
- 作用：收集并渲染多个上下文片段。
- `add`：添加单个 section。
- `extend`：批量添加 section。
- `ordered_sections`：按优先级排序。
- `render`：在预算限制下渲染全部内容。
- `_render_section`：渲染单个 section。
- `_clip_text`：裁剪过长文本。

#### `ContextBuilder`
- 作用：提供更好用的构建接口。
- `__init__`：初始化上下文包。
- `add_system_prompt`：添加系统提示。
- `add_runtime_rules`：添加运行规则。
- `add_memory`：添加记忆 section。
- `add_retrieval`：添加检索 section。
- `add_notes`：添加备注 section。
- `build`：返回最终上下文包。

### `core/llm_client.py`
- 作用：统一封装 OpenAI-compatible 模型调用。

#### 数据类
##### `ProviderSpec`
- 作用：描述 provider 基本信息。

##### `LLMConfig`
- 作用：描述一次模型调用所需的结构化配置。

##### `ChatResult`
- 作用：统一承载模型返回的文本、工具调用和 finish_reason。

#### `HelloAgentsLLM`
- 作用：当前项目的模型客户端封装。
- `__init__`：初始化 provider、模型和底层客户端。
- `from_config`：从 `Config` 创建客户端。
- `available_providers`：返回支持的 provider。
- `think`：保留的思考调用入口。
- `chat`：统一聊天调用入口。
- `_build_request`：构建请求体。
- `_consume_response`：解析非流式响应。
- `_consume_stream`：解析流式响应。
- `_resolve_provider`：解析 provider。
- `_resolve_config`：解析配置。
- `_detect_provider_from_env`：从环境变量推断 provider。
- `_normalize_provider`：标准化 provider 名称。
- `_infer_provider_from_base_url`：从 base URL 推断 provider。
- `_serialize_tool_call`：标准化工具调用结构。
- `_extract_reasoning`：抽取 reasoning 文本。
- `_content_to_text`：把不同内容格式折叠为文本。
- `_merge_stream_value`：合并流式片段。
- `_first_nonempty`：取第一个非空值。
- `_clean`：清理原始值。
- `_read_value`：兼容不同对象结构的字段读取。

### `core/message.py`
- 作用：定义统一消息结构。

#### `ToolFunction`
- 作用：表示工具函数调用。

#### `ToolCall`
- 作用：表示一次原生工具调用。

#### `Message`
- 作用：统一表示 `system / user / assistant / tool` 四类消息。
- `system`：构造 system message。
- `user`：构造 user message。
- `assistant`：构造 assistant message。
- `tool`：构造 tool message。
- `to_chat_message`：转成模型侧消息格式。
- `from_chat_message`：从模型消息恢复内部结构。
- `short`：生成预览文本。
- `__str__`：输出可读字符串。

#### 辅助函数
- `normalize_messages`：标准化消息列表。
- `trim_messages`：裁剪消息列表。

## 4. 模块关系

### 运行主链路
- `main.py` 负责创建对象并触发 demo。
- `agents/` 负责实际的推理流程。
- `core/llm_client.py` 负责和模型服务通信。
- `tools/` 负责执行外部能力。
- `memory/` 负责保存、检索和注入上下文。

### 三种 Agent 的共享基础
- `Agent` 提供最基础的抽象接口。
- `ReasoningAgentBase` 提供共享运行基础设施。
- `ReactAgent` 提供最小行动循环。
- `PlanAndSolveAgent` 和 `ReflectionAgent` 在其上扩展不同范式。

## 5. 推荐阅读顺序

1. `README.md`
2. `main.py`
3. `core/Config.py`
4. `core/message.py`
5. `core/context_engineering.py`
6. `core/llm_client.py`
7. `docs/agents.md`
8. `docs/memory.md`
9. `docs/tools.md`
