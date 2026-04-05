# PROJECT_DOC

修改说明：这是项目当前版本的总说明文档，按“文件 -> 类 -> 方法 / 函数”的顺序整理，方便后续继续开发、重构和对外说明。

## 1. 文档范围

这个文档描述当前仓库里主要源码文件、核心类、关键方法和辅助函数的作用。

说明约定：
- 只描述当前仓库里已经存在的代码。
- 方法说明以“它负责什么”为主，不展开逐行实现细节。
- `__init__` 主要说明对象初始化时建立了哪些运行依赖。
- `__init__.py` 这类文件如果没有复杂逻辑，主要说明它的导出或包组织作用。

## 2. 项目根目录文件

### `README.md`
- 作用：项目对外公开说明，介绍 helloAgent 的目标、当前能力、运行方式、限制和路线图。

### `TODO.md`
- 作用：记录当前版本接下来优先推进的任务，按 `P0 / P1 / P2` 分层管理。

### `LICENSE`
- 作用：项目开源许可证文件。

### `requestsment.txt`
- 作用：当前仓库中的依赖占位文件，名字还不是标准的 `requirements.txt`，后续可以再整理。

### `main.py`
- 作用：项目统一演示入口，负责构建 Agent、记忆系统、工具注册表，并提供一组 smoke test / demo。

#### 类

##### `NativeToolCallingSmokeLLM`
- 作用：模拟原生 tool calling 的假 LLM，用于不依赖真实模型服务的链路验证。
- `chat`：根据消息历史返回固定的 `tool_calls` 或最终答案，验证原生工具调用主循环。

##### `NativePlanAndSolveSmokeLLM`
- 作用：模拟 Plan-and-Solve 在步骤求解阶段的原生 tool calling 行为。
- `chat`：根据当前步骤和是否已有 tool message，返回计划、工具调用或步骤结果。

##### `NativeReflectionSmokeLLM`
- 作用：模拟 Reflection Agent 在草稿、反思和修订阶段的输出。
- `chat`：根据不同阶段的 prompt 返回工具调用、审查意见或修订结果。

##### `SchemaSmokeTool`
- 作用：用于测试工具 schema、参数归一化和跨字段校验的演示工具。
- `__init__`：初始化工具名称和描述。
- `run`：把归一化后的参数回显出来，方便观察 schema 执行效果。
- `get_parameters`：定义 `mode / level / dry_run` 三个测试参数。
- `validate_normalized_parameters`：演示跨字段语义校验规则。

##### `FlakyRecoveryTool`
- 作用：用于测试 Agent 自动重试逻辑的演示工具。
- `__init__`：初始化调用计数器。
- `run`：第一次返回可重试失败，第二次返回成功。
- `get_parameters`：声明该工具没有参数。

##### `AlwaysFailTool`
- 作用：用于测试工具失败后的降级提示逻辑。
- `__init__`：初始化工具元信息。
- `run`：始终返回失败结果。
- `get_parameters`：声明该工具没有参数。

#### 函数
- `configure_logging`：统一配置控制台日志格式和日志等级。
- `build_memory_manager`：创建默认 `MemoryManager`。
- `describe_embedding_backend`：输出当前 embedding 后端的简要说明。
- `build_rag_pipeline`：基于当前记忆配置构建 `RagPipeline`。
- `build_tool_registry`：注册 `get_time / memory_tool / rag_tool` 等示例工具。
- `_build_llm_and_config`：统一创建 LLM 客户端和配置对象。
- `build_react_agent`：构建可直接运行的 `ReactAgent`。
- `build_plan_and_solve_agent`：构建可直接运行的 `PlanAndSolveAgent`。
- `build_reflection_agent`：构建可直接运行的 `ReflectionAgent`。
- `print_run_summary`：统一打印最终答案和执行轨迹。
- `print_memory_snapshot`：输出当前会话的记忆快照。
- `print_runtime_error`：用简洁格式打印运行期异常。
- `ensure_demo_rag_document`：生成本地示例 RAG 文档。
- `test_react_agent`：运行 ReAct Agent 演示。
- `test_plan_and_solve_agent`：运行 Plan-and-Solve Agent 演示。
- `test_reflection_agent`：运行 Reflection Agent 演示。
- `test_memory_workflow`：运行两轮对话的记忆写入与召回演示。
- `test_memory_closure_smoke`：验证记忆闭环，包括低价值过滤、重复过滤、长期保留和召回解释。
- `test_rag_workflow`：运行最小可用的 RAG 建库与检索问答流程。
- `test_rag_pipeline_smoke`：不依赖 LLM，直接验证 RAG 管道本身。
- `test_embedding_smoke`：验证 embedding 向量生成和相似度趋势。
- `test_context_engineering_smoke`：检查上下文工程拼装结果。
- `test_context_routing_smoke`：验证不同问题类型触发的上下文路由。
- `test_context_conflict_smoke`：验证记忆、RAG、工具信息冲突时的提示逻辑。
- `test_summary_smoke`：验证会话摘要生成与注入效果。
- `test_native_tool_calling_smoke`：验证原生 tool calling 的主链路。
- `test_native_plan_smoke`：验证 Plan-and-Solve 的混合原生工具调用。
- `test_native_reflection_smoke`：验证 Reflection 的混合原生工具调用。
- `test_tool_schema_smoke`：验证工具 schema、参数归一化和统一结果协议。
- `test_tool_recovery_smoke`：验证工具失败后的自动重试与降级。
- `run_demo`：根据名称调度不同演示。
- `main`：命令行默认入口。

## 3. `agents/` 目录

### `agents/__init__.py`
- 作用：Agent 包标记文件，当前主要用于包组织。

### `agents/agent_base.py`
- 作用：定义所有 Agent 的最基础抽象接口。

#### 类
##### `Agent`
- 作用：所有 Agent 的抽象父类，统一持有 `name / llm / system_prompt / config / history`。
- `__init__`：初始化 Agent 基本身份、模型客户端和内部消息历史。
- `run`：抽象方法，要求子类实现自己的主流程。
- `add_message`：向内部历史追加一条消息。
- `clear_history`：清空内部历史。
- `get_history`：返回历史消息副本。
- `__str__`：输出一个简洁的 Agent 字符串描述。

### `agents/reasoning_agent_base.py`
- 作用：多种推理型 Agent 的公共父类，统一处理消息构建、上下文注入、记忆写入、原生 tool calling 和冲突提示。

#### 类
##### `ReasoningAgentBase`
- 作用：承接 `React / Plan-and-Solve / Reflection` 共有的运行基础设施。
- `__init__`：初始化工具注册表、prompt 模板、记忆管理器、session、工具观察缓存和原生 tool calling 状态。
- `_start_new_run`：启动一次新任务，重置运行态并注入用户输入。
- `_build_messages`：把当前 prompt 和结构化上下文组装成消息列表。
- `_request_text`：统一请求 LLM 并返回清理后的文本。
- `_request_result`：统一请求 LLM 并保留完整 `ChatResult`。
- `_request_result_with_messages`：允许直接传入消息列表请求 LLM。
- `_render_context_packet`：把上下文包渲染成最终文本。
- `_build_context_system_message`：把渲染好的上下文包装成 system message。
- `_build_native_user_message`：为原生 tool calling 构造 user message。
- `_build_native_tool_message`：把工具执行结果包装成 tool message。
- `_build_native_tool_calling_messages`：组装原生 tool calling 所需的消息序列。
- `_append_history_entry`：向调试历史里追加一条轨迹文本。
- `_should_use_native_tool_calling`：根据配置判断是否启用原生工具调用。
- `_build_assistant_message_from_result`：把 LLM 返回结果构造成 assistant message。
- `_execute_native_tool_call`：执行单个原生 `tool_call`。
- `_run_native_tool_calling_loop`：统一驱动“模型返回工具调用 -> 执行工具 -> 回填 -> 再请求模型”的循环。
- `_handle_action`：留给子类或上层逻辑调用的动作执行入口。
- `_build_memory_context`：构造记忆注入文本。
- `_build_context_packet`：把系统提示、规则、记忆、RAG、工具观察组装成结构化上下文包。
- `_build_session_summary`：获取当前会话摘要文本。
- `_build_auto_memory_sections`：生成自动注入的记忆 section。
- `_build_tool_observation_context`：把当前轮工具观察整理成上下文。
- `_build_auto_rag_context`：生成自动 RAG 注入结果。
- `_should_use_auto_rag`：判断当前问题是否值得自动检索本地知识。
- `_resolve_context_route`：为当前问题决定上下文路由策略。
- `_remember_tool_observation`：缓存工具 observation 供后续轮次使用。
- `_remember_tool_result_memory`：把工具结果写入记忆系统。
- `_build_tool_result_memory_metadata`：构造工具结果写入记忆时的 metadata。
- `_stash_tool_result_snapshot`：暂存最近一次工具结果快照。
- `_consume_tool_result_snapshot`：取出并清空最近一次工具结果快照。
- `_stringify_tool_payload`：把工具 payload 转成便于展示和记忆的文本。
- `_summarize_tool_observation_metadata`：提炼工具 observation 元信息。
- `_build_conflict_resolution_note`：生成上下文冲突消解说明。
- `_extract_claims_from_memory_sections`：从记忆 section 中提取可比较的事实声明。
- `_extract_claims_from_tool_observations`：从工具观察中提取事实声明。
- `_extract_claims_from_rag_evidence`：从 RAG 证据中提取事实声明。
- `_extract_claims_from_text`：从一般文本中抽取声明。
- `_normalize_claim_object`：统一声明对象的归一化表示。
- `_is_negative_preference`：判断某个偏好声明是否是否定型。
- `_detect_conflicts`：检测不同来源声明之间的冲突。
- `_resolve_conflict_winner`：按优先级规则判断冲突时应优先信任哪一侧。
- `_remember_message`：把消息加入内部历史并按策略写入记忆。
- `_remember_assistant_text`：专门写入 assistant 文本结果。
- `_should_persist_role`：判断某个角色消息是否应该持久化。
- `_render_history`：把当前调试历史渲染成字符串。
- `_preview`：生成日志或调试用的短文本预览。

### `agents/react_agent.py`
- 作用：项目里的文本版 ReAct Agent，也是原生 tool calling 主链路的首个落地点。

#### 类
##### `ReactAgent`
- 作用：围绕 `Thought / Action / Observation / Finish` 循环运行任务。
- `__init__`：初始化 ReAct 运行参数、最大步数和是否启用原生工具调用。
- `run`：执行 ReAct 主循环。
- `_run_with_native_tool_calling`：使用原生 tool calling 方式处理当前问题。
- `_build_prompt`：拼接当前轮 prompt。
- `_handle_action`：解析并执行一个文本版 Action。
- `_handle_tool_result`：把工具执行结果转成 Observation 和历史记录。
- `_execute_tool_with_recovery`：带自动重试与降级逻辑执行工具。
- `_annotate_tool_attempt_metadata`：给工具结果补充重试次数等元数据。
- `_should_retry_tool_result`：判断一个工具失败结果是否应重试。
- `_finalize_failed_tool_result`：对失败结果做最终整理。
- `_build_tool_degradation_guidance`：构造降级提示文本。
- `_sleep_before_tool_retry`：在重试前按配置等待。
- `_prepare_tool_parameters`：准备工具参数并做归一化。
- `_parse_tool_input`：把 `tool_input` 字符串解析成参数字典。
- `_append_observation`：把 Observation 写进历史和消息流。
- `_log_step_start`：记录当前步开始时的调试日志。
- `parse_react_response`：从模型输出中解析 `Thought` 和 `Action`。
- `parse_action`：把 `Action: xxx[...]` 解析为动作类型和输入。

### `agents/plan_and_solve.py`
- 作用：实现 Plan-and-Solve 范式，先规划，再按步骤求解，最后汇总答案。

#### 类
##### `PlanAndSolveAgent`
- 作用：在 ReAct 工具能力基础上增加“计划 -> 分步执行 -> 汇总”的流程。
- `__init__`：初始化计划 prompt、最终汇总 prompt 和每步最大轮数。
- `run`：执行完整的 Plan-and-Solve 主流程。
- `_generate_plan`：请求模型先生成计划列表。
- `_parse_plan`：把文本计划解析为步骤列表。
- `_solve_step`：执行单个步骤，必要时可调用工具。
- `_solve_step_with_native_tool_calling`：以原生 tool calling 方式求解单个步骤。
- `_generate_final_answer`：把各步骤结果汇总成最终回答。
- `_render_plan`：把计划列表渲染为文本。
- `_render_completed_steps`：把已完成步骤结果渲染为文本。

### `agents/reflection_agent.py`
- 作用：实现 Reflection 范式，先生成草稿，再审查，再按需修订。

#### 类
##### `ReflectionAgent`
- 作用：通过“草稿 + 审查 + 修订”提高回答质量，并保护工具事实不被无依据改写。
- `__init__`：初始化反思轮数、审查 prompt 和修订 prompt。
- `run`：执行 Reflection 主流程。
- `_build_draft`：生成初始草稿答案。
- `_build_draft_with_native_tool_calling`：通过原生 tool calling 生成草稿。
- `_review_answer`：请求模型审查当前答案质量。
- `_revise_answer`：根据审查意见修订答案。
- `_parse_review`：解析审查结果中的 `Reflection / Decision / Suggestions`。
- `_render_grounded_facts`：渲染工具确认过的事实清单。
- `_drops_grounded_facts`：检查修订答案是否丢失关键工具事实。
- `_normalize_decision`：规范化审查阶段的决策字段。
- `_format_reflection_text`：把审查结构整理为可读文本。

## 4. `core/` 目录

### `core/__init__.py`
- 作用：统一导出 `Config / Message / HelloAgentsLLM / ContextBuilder` 等核心对象。

### `core/Config.py`
- 作用：集中管理项目配置，包括模型、上下文预算、工具调用和记忆策略参数。

#### 类
##### `Config`
- 作用：项目运行时的统一配置对象。
- `from_env`：从环境变量构建配置。
- `llm_options`：导出请求模型时需要的参数字典。
- `trimmed_history`：根据配置裁剪消息历史。
- `to_dict`：把配置转成普通字典。

#### 函数
- `_first_env`：从多个环境变量名里取第一个有效值。
- `_read_bool`：把环境变量解析成布尔值。
- `_read_float`：把环境变量解析成浮点数。
- `_read_int`：把环境变量解析成整数。

### `core/context_engineering.py`
- 作用：提供结构化上下文工程抽象，把系统提示、记忆、检索结果和规则组合成有优先级的上下文。

#### 类
##### `ContextSection`
- 作用：描述一段上下文片段，包括标题、内容、来源和优先级。
- `render`：把当前 section 渲染为可注入模型的文本。

##### `ContextPacket`
- 作用：收集多个 `ContextSection`，并在渲染前完成排序、裁剪和去重。
- `add`：追加单个 section。
- `extend`：批量追加 section。
- `ordered_sections`：按优先级返回排序后的 section 列表。
- `render`：在预算约束下把全部 section 渲染成文本。
- `_render_section`：渲染单个 section。
- `_clip_text`：按长度限制裁剪文本。

##### `ContextBuilder`
- 作用：用更直观的接口构造 `ContextPacket`。
- `__init__`：初始化内部 `ContextPacket`。
- `add_system_prompt`：添加系统提示 section。
- `add_runtime_rules`：添加运行规则 section。
- `add_memory`：添加记忆 section。
- `add_retrieval`：添加检索结果 section。
- `add_notes`：添加补充说明 section。
- `build`：返回最终构建好的 `ContextPacket`。

### `core/llm_client.py`
- 作用：封装 OpenAI-compatible LLM 调用，兼容多种 provider 和原生 tool calling 结果。

#### 类
##### `ProviderSpec`
- 作用：描述某个 provider 的基础信息。

##### `LLMConfig`
- 作用：承载模型调用所需的结构化配置。

##### `ChatResult`
- 作用：承载一次对话请求返回的统一结果，包括文本、工具调用、finish_reason 等。

##### `HelloAgentsLLM`
- 作用：统一对接多家 OpenAI-compatible 模型服务。
- `__init__`：初始化 provider、模型、认证信息和客户端。
- `from_config`：根据 `Config` 创建 LLM 客户端。
- `available_providers`：返回支持的 provider 列表。
- `think`：保留的思考调用入口。
- `chat`：发起一次聊天请求。
- `_build_request`：把内部消息结构转换成模型接口请求体。
- `_consume_response`：解析非流式响应。
- `_consume_stream`：解析流式响应。
- `_resolve_provider`：解析最终应使用的 provider。
- `_resolve_config`：整理模型调用配置。
- `_detect_provider_from_env`：从环境变量推断 provider。
- `_normalize_provider`：统一 provider 名称格式。
- `_infer_provider_from_base_url`：通过 base URL 推断 provider。
- `_serialize_tool_call`：把模型侧工具调用对象标准化。
- `_extract_reasoning`：从响应中提取 reasoning 内容。
- `_content_to_text`：把多种内容结构折叠成文本。
- `_merge_stream_value`：合并流式响应的增量字段。
- `_first_nonempty`：返回多个候选中的第一个有效值。
- `_clean`：清理空白或异常输入。
- `_read_value`：从不同对象结构中安全读取字段。

### `core/message.py`
- 作用：统一定义消息结构，兼容普通消息和原生 `tool_call` 消息。

#### 类
##### `ToolFunction`
- 作用：表示一个工具函数调用信息。

##### `ToolCall`
- 作用：表示一次原生工具调用。

##### `Message`
- 作用：统一描述 system / user / assistant / tool 四类消息。
- `system`：构造 system message。
- `user`：构造 user message。
- `assistant`：构造 assistant message。
- `tool`：构造 tool message。
- `to_chat_message`：转换成发给模型的消息格式。
- `from_chat_message`：从模型消息结构恢复内部 `Message`。
- `short`：生成简短预览。
- `__str__`：输出可读字符串。

#### 函数
- `normalize_messages`：把原始消息列表统一整理成 `Message` 对象。
- `trim_messages`：按数量或策略裁剪消息列表。

## 5. `memory/` 目录

### `memory/__init__.py`
- 作用：统一导出记忆相关核心对象。

### `memory/base.py`
- 作用：定义记忆系统的核心数据结构、基础配置和抽象接口。

#### 类
##### `MemoryItem`
- 作用：统一描述一条记忆数据。
- `is_expired`：判断当前记忆是否过期。

##### `MemoryConfig`
- 作用：集中管理记忆系统、向量存储、图谱存储、RAG 和长期保留策略配置。
- `from_env`：从环境变量构建记忆配置。
- `working_expires_at`：生成工作记忆默认过期时间。

##### `BaseMemory`
- 作用：所有记忆类型的抽象接口。
- `add`：写入一条记忆。
- `recent`：获取最近记忆。
- `search`：按 query 检索记忆。
- `clear`：清空指定 session 的记忆。

#### 函数
- `_first_env`：读取首个有效环境变量值。
- `_read_bool`：解析布尔配置。
- `_read_int`：解析整数配置。
- `_read_float`：解析浮点配置。

### `memory/embedding.py`
- 作用：提供离线 hash embedding 和 Ollama embedding 两套实现，以及统一工厂。

#### 类
##### `BaseEmbeddingService`
- 作用：embedding 服务抽象基类。
- `embed`：为单条文本生成向量。
- `embed_many`：批量生成向量。
- `dimension_hint`：返回向量维度提示。
- `cosine_similarity`：计算两条向量的余弦相似度。

##### `HashEmbeddingService`
- 作用：完全离线的 embedding 实现，用于本地无依赖验证。
- `__init__`：初始化维度和概念分组等参数。
- `embed`：把文本编码成固定维度的哈希向量。
- `dimension_hint`：返回固定维度。
- `_extract_units`：抽取 token、双字片段和概念标签，增强离线召回能力。

##### `OllamaEmbeddingService`
- 作用：调用本地 Ollama embedding 模型生成真实向量。
- `__init__`：初始化模型名、地址、超时和维度提示。
- `embed`：为单条文本生成 embedding。
- `embed_many`：批量生成 embedding。
- `dimension_hint`：返回可推断的维度提示。
- `_request_single_embedding`：请求一次单文本 embedding。
- `_post_json`：发送 HTTP JSON 请求。
- `_normalize_vector`：对返回向量做归一化处理。

##### `EmbeddingServiceFactory`
- 作用：根据 `MemoryConfig` 决定创建哪种 embedding 服务。
- `create`：返回对应的 embedding 实现。

##### `SimpleEmbeddingService`
- 作用：兼容旧代码的别名或轻量占位类型。

### `memory/manager.py`
- 作用：统一协调 working / episodic / semantic 三类记忆的写入、检索、解释、摘要和长期保留。

#### 类
##### `MemoryManager`
- 作用：当前记忆系统的总控制器。
- `__init__`：初始化 embedding、三类记忆、决策日志和 retention 日志。
- `record_message`：按策略把一条消息写入合适的记忆层。
- `recall`：组合 working / episodic / semantic 的召回结果。
- `build_recall_diagnostics`：输出最近一次召回解释。
- `build_memory_prompt`：把召回结果渲染成可直接注入 prompt 的文本。
- `build_structured_memory_sections`：把记忆分成 `用户偏好 / 项目事实 / 近期对话` 等分组。
- `build_structured_memory_prompt`：把结构化记忆渲染成文本块。
- `build_session_summary`：基于最近记忆生成轻量会话摘要。
- `clear_session`：清空某个 session 的全部记忆和调试日志。
- `build_memory_diagnostics`：输出最近的记忆写入决策。
- `build_retention_diagnostics`：输出最近的长期保留裁剪结果。
- `_dedupe_items`：对召回项做去重和截断。
- `_classify_memory_item`：把记忆项分到偏好、事实或近期对话桶中。
- `_trim_summary_line`：裁剪过长的单行文本。
- `_annotate_recall_source`：给召回项补来源信息。
- `_merge_recall_sources`：合并多个召回来源标记。
- `_build_recall_reason`：生成召回原因说明。
- `_render_recall_explanation`：生成适合直接展示的召回短说明。
- `_plan_memory_record`：在写入前决定该消息应进入哪些记忆层。
- `_build_memory_plan`：构造统一的写入计划字典。
- `_record_memory_decision`：记录一次写入决策。
- `_apply_retention`：统一执行长期保留策略。
- `_apply_single_retention`：对单个持久化层执行裁剪。
- `_select_items_to_keep`：从候选记忆中选出应该保留的条目。
- `_score_retention_priority`：为 retention 计算优先级分数。
- `_record_retention_result`：记录 retention 的裁剪结果。
- `_has_recent_duplicate`：检测最近窗口内是否有重复记忆。
- `_classify_content_kind`：判定内容属于偏好、事实、工具结果或普通对话等类别。
- `_score_memory_value`：给消息打高、中、低价值标签。
- `_is_low_value_message`：过滤明显低价值文本。
- `_should_store_in_semantic`：判断一条消息是否值得进入语义记忆。
- `_normalize_memory_text`：归一化文本，方便做重复检测。

### `memory/types/__init__.py`
- 作用：记忆类型子包标记文件。

### `memory/types/working.py`
- 作用：实现纯内存、带 TTL 的工作记忆。

#### 类
##### `WorkingMemory`
- 作用：保存最近的上下文消息。
- `__init__`：初始化按 session 分桶的内存队列。
- `add`：添加一条工作记忆，并按容量上限裁剪。
- `recent`：获取最近的若干条工作记忆。
- `search`：基于简单 token 命中做工作记忆检索。
- `clear`：清空某个 session 的工作记忆。
- `_cleanup`：移除已经过期的工作记忆。

### `memory/types/episodic.py`
- 作用：封装情景记忆，对接持久化文档存储。

#### 类
##### `EpisodicMemory`
- 作用：负责长期保存会话中的关键消息。
- `__init__`：绑定底层 `DocumentStore`。
- `add`：写入一条情景记忆。
- `recent`：读取最近的情景记忆。
- `search`：检索情景记忆。
- `list_all`：返回某个 session 的全部情景记忆。
- `prune`：执行长期保留裁剪。
- `clear`：清空某个 session 的情景记忆。

### `memory/types/semantic.py`
- 作用：封装语义记忆，把向量召回和图谱召回组合起来。

#### 类
##### `SemanticMemory`
- 作用：保存更适合长期复用的偏好、事实和工具结果。
- `__init__`：绑定向量存储、图存储和 embedding 服务。
- `add`：写入一条语义记忆，同时同步到向量和图谱。
- `recent`：获取最近的语义记忆。
- `search`：组合向量检索和图谱检索结果。
- `list_all`：返回某个 session 的全部语义记忆。
- `prune`：同步裁剪向量存储和图存储中的语义记忆。
- `clear`：清空某个 session 的语义记忆。
- `_merge_items`：合并去重向量和图谱两路召回结果。

### `memory/types/perceptual.py`
- 作用：保留中的感知记忆占位实现，当前主要是接口骨架。

#### 类
##### `PerceptualMemory`
- 作用：表示未来可能接入的感知型短时记忆。
- `add`：占位写入接口。
- `recent`：占位最近记忆接口。
- `search`：占位检索接口。
- `clear`：占位清空接口。

### `memory/storage/__init__.py`
- 作用：记忆存储子包标记文件。

### `memory/storage/document_store.py`
- 作用：实现基于 SQLite / JSON fallback 的轻量文档存储。

#### 类
##### `DocumentStore`
- 作用：为情景记忆提供持久化读写能力。
- `__init__`：初始化数据库路径、JSON fallback 路径和后端类型。
- `_connect`：建立 SQLite 连接。
- `_initialize`：初始化表结构和索引。
- `add_item`：写入一条持久化记忆。
- `list_recent`：读取最近若干条记忆。
- `list_session_items`：读取某个 session 的全部持久化记忆。
- `search_items`：按简单文本匹配检索记忆。
- `clear_session`：清空指定 session 的记忆。
- `prune_session`：只保留指定 id 的记忆，用于 retention。
- `_load_json_items`：从 JSON fallback 读取记忆。
- `_save_json_items`：把记忆写回 JSON fallback。
- `_activate_json_fallback`：切换到 JSON fallback。
- `_row_to_item`：把 SQLite 行对象转换成 `MemoryItem`。

### `memory/storage/qdrant_store.py`
- 作用：实现 Qdrant / JSON fallback 的向量存储适配层。

#### 类
##### `QdrantVectorStore`
- 作用：为语义记忆和 RAG 共用统一向量存储能力。
- `__init__`：初始化配置、embedding 服务、collection 和本地 fallback 路径。
- `upsert`：兼容语义记忆接口，写入一条向量化记忆。
- `search`：兼容语义记忆接口，按 query 检索记忆。
- `list_recent`：兼容语义记忆接口，读取最近入库的记忆。
- `list_session_items`：返回某个 session 的全部语义记忆记录。
- `clear_session`：清空某个 session 的向量记忆。
- `prune_session`：按 keep id 裁剪某个 session 的向量记忆。
- `upsert_record`：写入一条通用向量记录。
- `search_records`：检索通用向量记录。
- `list_recent_records`：返回最近的通用向量记录。
- `list_all_records`：返回满足过滤条件的全部记录。
- `clear_records`：清空满足过滤条件的记录。
- `_initialize_backend`：决定使用真实 Qdrant 还是 JSON fallback。
- `_can_use_qdrant`：检查当前环境是否可用 Qdrant。
- `_ensure_qdrant_collection`：创建或检查 collection 配置。
- `_qdrant_upsert`：使用真实 Qdrant 写入记录。
- `_qdrant_search`：使用真实 Qdrant 检索记录。
- `_qdrant_list_recent`：读取 Qdrant 中最近记录。
- `_normalize_qdrant_point_id`：把业务 id 规范成 Qdrant 合法 point id。
- `_resolve_vector_size`：解析当前应使用的向量维度。
- `_extract_collection_vector_size`：从 collection 信息中提取维度。
- `_qdrant_clear`：清空真实 Qdrant 中的记录或 collection。
- `_qdrant_prune_session`：在 Qdrant 中执行 session 级裁剪。
- `_qdrant_scroll_all`：滚动读取满足条件的全部 Qdrant 记录。
- `_build_qdrant_filter`：构造 Qdrant 过滤器。
- `_ensure_json_store`：确保本地 JSON fallback 文件存在。
- `_load_json_records`：从 JSON fallback 读取记录。
- `_save_json_records`：写回 JSON fallback 记录。
- `_normalize_record`：兼容旧格式并规范记录结构。
- `_match_filters`：判断一条 payload 是否命中过滤条件。

### `memory/storage/neo4j_store.py`
- 作用：实现 Neo4j / JSON fallback 的图谱存储适配层。

#### 类
##### `Neo4jGraphStore`
- 作用：为语义记忆提供实体、关系和关联召回能力。
- `__init__`：初始化图存储配置、本地 fallback 路径和后端类型。
- `upsert_memory`：把一条记忆及其实体关系写入图存储。
- `search_related`：根据查询中的实体查找关联记忆。
- `clear_session`：清空某个 session 的图谱数据。
- `list_session_items`：读取某个 session 的全部图谱记忆。
- `list_recent`：读取最近写入图谱的记忆。
- `prune_session`：裁剪某个 session 的图谱记忆。
- `extract_entities`：从文本中抽取实体候选。
- `extract_relations`：从文本中抽取关系三元组。
- `_extract_targets`：从关系尾部文本中抽目标实体。
- `_normalize_entity`：归一化实体文本。
- `_normalize_relation`：归一化关系名称。
- `_dedupe_texts`：对实体或关系文本做去重。
- `_initialize_backend`：决定使用真实 Neo4j 还是 JSON fallback。
- `_can_use_neo4j`：检查当前环境是否可连通 Neo4j。
- `_neo4j_upsert_memory`：写入真实 Neo4j。
- `_neo4j_search_related`：从真实 Neo4j 中检索相关记忆。
- `_neo4j_clear_session`：清空真实 Neo4j 中某个 session 的数据。
- `_neo4j_list_recent`：读取真实 Neo4j 中最近记忆。
- `_neo4j_list_all`：读取真实 Neo4j 中全部记忆。
- `_neo4j_prune_session`：裁剪真实 Neo4j 中的 session 数据。
- `_json_upsert_memory`：把图谱数据写入本地 JSON fallback。
- `_json_search_related`：从本地 JSON fallback 中按实体做关联检索。
- `_ensure_json_store`：确保 JSON fallback 文件存在。
- `_load_json_payload`：读取 JSON fallback 的图谱数据。
- `_save_json_payload`：写回 JSON fallback 的图谱数据。

## 6. `memory/rag/` 目录

### `memory/rag/__init__.py`
- 作用：统一导出 RAG 子模块对象。

### `memory/rag/document.py`
- 作用：定义文档切片数据结构和基础文档处理逻辑。

#### 类
##### `DocumentChunk`
- 作用：表示一段已切好的文档片段。

##### `RetrievedChunk`
- 作用：表示一条已检索到的文档片段及其分数。

##### `DocumentProcessor`
- 作用：负责加载文本和按规则切片。
- `__init__`：初始化 chunk 大小和重叠长度。
- `load`：从文件路径读取文档并切片。
- `load_text`：直接对传入文本切片。
- `split_text`：执行实际的文本切分逻辑。

### `memory/rag/pipeline.py`
- 作用：实现本地最小可用的 RAG 管道。

#### 类
##### `RagPipeline`
- 作用：完成文档入库、向量检索、简单重排和上下文构造。
- `__init__`：初始化配置、文档处理器、embedding 服务和向量存储。
- `add_document`：读取文档、切片并写入向量索引。
- `search`：先向量召回，再按词项和规则重排切片。
- `answer`：返回更适合直接参考的检索上下文答案。
- `clear`：清空当前 RAG 索引。
- `list_sources`：列出当前已索引的文档来源。
- `run`：兼容旧接口的执行入口。
- `_search_inline_documents`：对内联文档集合执行检索。
- `build_answer_context`：把命中结果组织成“参考结论 + 证据片段”的结构化上下文。
- `_format_matches`：格式化命中的文档片段。
- `_rerank_score`：计算重排分数。
- `_extract_query_tokens`：从 query 中抽取关键词。
- `_summarize_chunk`：生成片段摘要。
- `_build_chunk_record_id`：为切片构建稳定 record id。

## 7. `tools/` 目录

### `tools/__init__.py`
- 作用：工具包标记文件。

### `tools/builtin/tool_base.py`
- 作用：定义工具系统的统一抽象、参数 schema 和执行结果协议。

#### 类
##### `ToolValidationError`
- 作用：表示工具参数校验失败的统一异常。

##### `ToolResult`
- 作用：统一表示工具执行结果。
- `ok`：构造成功结果。
- `fail`：构造失败结果。
- `render_for_observation`：把结果渲染成适合写回 Observation 的文本。

##### `ToolParameter`
- 作用：描述单个工具参数，包括类型、默认值、枚举和范围约束。

##### `Tool`
- 作用：所有工具的抽象基类。
- `__init__`：初始化工具名称和描述。
- `run`：工具自身的实际执行逻辑。
- `get_parameters`：返回参数定义列表。
- `execute`：统一执行工具并保证输出 `ToolResult`。
- `get_parameters_schema`：生成兼容 function calling 的 JSON schema。
- `normalize_parameters`：按 schema 对参数做归一化和基础校验。
- `_normalize_parameter_value`：归一化单个参数值。
- `validate_normalized_parameters`：做跨字段或更高层语义校验。
- `_coerce_value`：把字符串等输入转成目标类型。
- `_validate_parameter_constraints`：检查枚举、范围、长度等约束。
- `format_for_prompt`：把工具描述格式化给文本版 Agent 使用。
- `_coerce_result`：把工具返回值统一折叠成 `ToolResult`。
- `_build_exception_result`：把异常转换成失败结果。
- `_is_retryable_exception`：判断某个异常是否可重试。

### `tools/builtin/toolRegistry.py`
- 作用：管理工具注册、获取和 schema 导出。

#### 类
##### `ToolRegistry`
- 作用：作为所有工具的统一注册表。
- `__init__`：初始化内部工具字典。
- `register_tool`：按新命名方式注册工具。
- `registerTool`：旧命名兼容接口。
- `get_tool`：按名称获取工具。
- `getTool`：旧命名兼容接口。
- `list_tools`：返回全部已注册工具。
- `get_available_tools`：导出原生 tool calling 需要的工具 schema 列表。
- `getAvailableTools`：旧命名兼容接口。
- `describe_tools`：生成文本版工具说明，供 ReAct prompt 使用。

### `tools/builtin/get_time.py`
- 作用：提供最简单的示例工具，用来读取当前本地时间。

#### 类
##### `GetTimeTool`
- 作用：为 Agent 暴露 `get_time` 能力。
- `__init__`：初始化工具名称和说明。
- `run`：返回当前时间。
- `get_parameters`：声明这个工具不需要参数。

#### 函数
- `get_time`：返回当前时间字符串。

### `tools/builtin/memory_tool.py`
- 作用：把记忆系统封装成工具接口，供 Agent 显式读写记忆。

#### 类
##### `MemoryTool`
- 作用：提供 `recent / search / context / summary / remember / clear` 六类动作。
- `__init__`：绑定 `MemoryManager` 和当前 session。
- `run`：根据 action 执行不同的记忆读写操作。
- `get_parameters`：定义工具参数 schema。
- `validate_normalized_parameters`：检查 action 与参数组合是否合法。
- `_format_items`：把记忆项渲染成可读文本。

### `tools/builtin/rag_tool.py`
- 作用：把本地 RAG 管道封装成工具接口。

#### 类
##### `RagTool`
- 作用：提供 `add / search / answer / context / clear` 五类动作。
- `__init__`：绑定 `RagPipeline`。
- `run`：根据 action 执行建索引、检索、回答或清空操作。
- `get_parameters`：定义工具参数 schema。
- `validate_normalized_parameters`：检查当前 action 是否具备必要参数。

## 8. 目前模块之间的关系

### Agent 层
- `Agent` 是最基础抽象。
- `ReasoningAgentBase` 在 `Agent` 之上补齐消息、上下文、记忆和工具调用基础设施。
- `ReactAgent` 提供最核心的工具调用循环。
- `PlanAndSolveAgent` 和 `ReflectionAgent` 都是在 `ReactAgent` / `ReasoningAgentBase` 能力之上扩展不同推理范式。

### 工具层
- `Tool` 负责统一参数 schema 和结果协议。
- `ToolRegistry` 管理工具注册和导出。
- `GetTimeTool / MemoryTool / RagTool` 是当前内置工具。

### 记忆层
- `MemoryManager` 统一协调 `WorkingMemory / EpisodicMemory / SemanticMemory`。
- `DocumentStore / QdrantVectorStore / Neo4jGraphStore` 分别负责不同类型的后端存储。
- `MemoryTool` 和 `ReasoningAgentBase` 都会调用 `MemoryManager`。

### 检索层
- `DocumentProcessor` 负责文档切片。
- `RagPipeline` 负责入库、检索、重排和答案上下文生成。
- `RagTool` 是它对 Agent 的工具封装。

### 模型层
- `Config` 和 `MemoryConfig` 提供运行时配置。
- `HelloAgentsLLM` 负责和实际模型服务交互。
- `Message / ToolCall / ToolFunction / ChatResult` 是模型交互的统一数据结构。

## 9. 当前建议的阅读顺序

如果后面继续维护这个项目，推荐按下面顺序读代码：

1. `main.py`
2. `agents/agent_base.py`
3. `agents/reasoning_agent_base.py`
4. `agents/react_agent.py`
5. `agents/plan_and_solve.py`
6. `agents/reflection_agent.py`
7. `tools/builtin/tool_base.py`
8. `tools/builtin/toolRegistry.py`
9. `memory/manager.py`
10. `memory/storage/*`
11. `memory/rag/*`
12. `core/*`

这样会比较容易看清楚整个项目是怎么从“主流程”一步步落到“工具、记忆、上下文和模型调用”上的。

## 10. 后续维护建议

- 如果新增 Agent 范式，优先复用 `ReasoningAgentBase`。
- 如果新增工具，优先继承 `Tool`，把参数 schema 和 `ToolResult` 一起补齐。
- 如果继续增强记忆系统，优先在 `MemoryManager` 里收敛策略，而不是把逻辑分散到 Agent 层。
- 如果继续做上下文工程，优先沿着 `ContextSection / ContextPacket / ContextBuilder` 扩展。
- 如果后续文件继续增多，建议把这个文档拆成：`docs/architecture.md`、`docs/agents.md`、`docs/memory.md`、`docs/tools.md`。
