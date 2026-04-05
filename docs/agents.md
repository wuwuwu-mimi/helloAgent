# Agent 模块说明

修改说明：这个文档聚焦 `agents/` 目录，按文件、类、方法说明当前三种 Agent 范式和公共父类。

## 1. `agents/__init__.py`
- 作用：Agent 包标记文件，当前主要用于包组织。

## 2. `agents/agent_base.py`
- 作用：定义所有 Agent 的最基础抽象接口。

### `Agent`
- 作用：所有 Agent 的抽象父类，统一持有 `name / llm / system_prompt / config / history`。
- `__init__`：初始化 Agent 基础身份、模型客户端和内部消息历史。
- `run`：抽象方法，要求子类实现自己的主流程。
- `add_message`：向内部历史追加一条消息。
- `clear_history`：清空内部历史。
- `get_history`：返回历史消息副本。
- `__str__`：输出一个简洁字符串，便于调试和打印。

## 3. `agents/reasoning_agent_base.py`
- 作用：多种推理型 Agent 的公共父类，负责消息构建、上下文工程、记忆写入、原生 tool calling、冲突提示等共性能力。

### `ReasoningAgentBase`
- 作用：给 `React / Plan-and-Solve / Reflection` 提供共享运行骨架。
- `__init__`：初始化工具注册表、prompt 模板、记忆管理器、session、工具观察缓存和原生调用状态。
- `_start_new_run`：开启一次新任务并重置运行态。
- `_build_messages`：把当前 prompt 和结构化上下文组装成消息列表。
- `_request_text`：请求模型并返回文本。
- `_request_result`：请求模型并返回完整 `ChatResult`。
- `_request_result_with_messages`：允许直接传入消息列表请求模型。
- `_render_context_packet`：渲染上下文包。
- `_build_context_system_message`：把上下文渲染结果包装成 system message。
- `_build_native_user_message`：构造原生工具调用模式下的 user message。
- `_build_native_tool_message`：构造原生工具调用模式下的 tool message。
- `_build_native_tool_calling_messages`：构造原生工具调用所需的消息序列。
- `_append_history_entry`：向调试历史中追加条目。
- `_should_use_native_tool_calling`：判断当前是否启用原生工具调用。
- `_build_assistant_message_from_result`：把 `ChatResult` 转成 assistant message。
- `_execute_native_tool_call`：执行单个 `tool_call`。
- `_run_native_tool_calling_loop`：统一处理原生工具调用循环。
- `_handle_action`：留给子类复用的动作执行入口。
- `_build_memory_context`：构造记忆注入文本。
- `_build_context_packet`：组合系统提示、规则、记忆、RAG、工具观察为结构化上下文。
- `_build_session_summary`：获取会话摘要。
- `_build_auto_memory_sections`：构造自动注入的记忆 section。
- `_build_tool_observation_context`：把工具 observation 组织为上下文。
- `_build_auto_rag_context`：构造自动 RAG 检索上下文。
- `_should_use_auto_rag`：判断是否自动做 RAG 检索。
- `_resolve_context_route`：根据问题决定 `memory_first / rag_first / tool_first / balanced` 路由。
- `_remember_tool_observation`：缓存工具观察结果。
- `_remember_tool_result_memory`：把工具结果写入记忆系统。
- `_build_tool_result_memory_metadata`：构造工具结果记忆 metadata。
- `_stash_tool_result_snapshot`：暂存最近一次工具结果。
- `_consume_tool_result_snapshot`：读取并清空最近一次工具结果。
- `_stringify_tool_payload`：把工具 payload 转成文本。
- `_summarize_tool_observation_metadata`：提炼工具观察元信息。
- `_build_conflict_resolution_note`：生成冲突消解说明。
- `_extract_claims_from_memory_sections`：从记忆 section 抽取事实声明。
- `_extract_claims_from_tool_observations`：从工具观察抽取事实声明。
- `_extract_claims_from_rag_evidence`：从 RAG 证据抽取事实声明。
- `_extract_claims_from_text`：从一般文本抽取声明。
- `_normalize_claim_object`：归一化声明对象。
- `_is_negative_preference`：判断偏好声明是否是否定式。
- `_detect_conflicts`：检测不同来源声明之间的冲突。
- `_resolve_conflict_winner`：在冲突时决定优先相信哪一侧。
- `_remember_message`：把消息加入历史并按策略记忆。
- `_remember_assistant_text`：写入 assistant 文本结果。
- `_should_persist_role`：判断某个角色是否应该持久化。
- `_render_history`：把调试历史渲染成字符串。
- `_preview`：生成日志预览文本。

## 4. `agents/react_agent.py`
- 作用：项目里的核心行动型 Agent，实现文本版 ReAct 主循环，并承接原生 tool calling 主链路。

### `ReactAgent`
- 作用：围绕 `Thought / Action / Observation / Finish` 进行多轮执行。
- `__init__`：初始化最大步数、prompt 模板和是否启用原生工具调用。
- `run`：执行 ReAct 主循环。
- `_run_with_native_tool_calling`：切换到原生 tool calling 模式运行。
- `_build_prompt`：构造当前轮发给模型的 prompt。
- `_handle_action`：根据 `Action` 执行对应工具或结束逻辑。
- `_handle_tool_result`：把工具执行结果转成 observation 和历史。
- `_execute_tool_with_recovery`：带自动重试和降级策略执行工具。
- `_annotate_tool_attempt_metadata`：给工具结果补充 attempt 等元数据。
- `_should_retry_tool_result`：判断失败结果是否值得重试。
- `_finalize_failed_tool_result`：对失败结果做最终整理。
- `_build_tool_degradation_guidance`：构造降级提示文本。
- `_sleep_before_tool_retry`：在重试前等待。
- `_prepare_tool_parameters`：准备、归一化并校验工具参数。
- `_parse_tool_input`：把字符串版 `tool_input` 解析成字典参数。
- `_append_observation`：把 observation 写入调试历史和消息历史。
- `_log_step_start`：记录当前步开始时的日志。
- `parse_react_response`：解析模型输出中的 `Thought` 与 `Action`。
- `parse_action`：解析 `Action: tool_name[...]` 或 `Finish[...]`。

## 5. `agents/plan_and_solve.py`
- 作用：实现 Plan-and-Solve 范式，强调“先规划，再逐步求解，最后汇总”。

### `PlanAndSolveAgent`
- 作用：在 ReAct 工具循环之上增加计划与分步执行能力。
- `__init__`：初始化计划 prompt、最终答案 prompt 和每步最大求解轮数。
- `run`：执行完整的 Plan-and-Solve 主流程。
- `_generate_plan`：先让模型输出步骤计划。
- `_parse_plan`：把文本计划解析成步骤列表。
- `_solve_step`：解决单个步骤，必要时允许工具调用。
- `_solve_step_with_native_tool_calling`：在原生工具调用模式下解决当前步骤。
- `_generate_final_answer`：把所有步骤结果整合成最终答案。
- `_render_plan`：把计划渲染成可读文本。
- `_render_completed_steps`：把已完成步骤结果渲染成文本。

## 6. `agents/reflection_agent.py`
- 作用：实现 Reflection 范式，强调“先草稿，再审查，再修订”。

### `ReflectionAgent`
- 作用：通过自我审查提升答案质量，并尽量保护已确认的工具事实。
- `__init__`：初始化最大反思轮数、相关 prompt 和运行参数。
- `run`：执行 Reflection 主流程。
- `_build_draft`：生成初始草稿答案。
- `_build_draft_with_native_tool_calling`：通过原生工具调用方式生成草稿。
- `_review_answer`：请求模型审查当前答案。
- `_revise_answer`：根据审查意见修订答案。
- `_parse_review`：解析审查结果里的 `Reflection / Decision / Suggestions`。
- `_render_grounded_facts`：渲染已经由工具确认的事实。
- `_drops_grounded_facts`：检查修订结果是否丢失关键工具事实。
- `_normalize_decision`：标准化审查结果里的决策值。
- `_format_reflection_text`：把反思信息整理成可读文本。

## 7. 三种 Agent 的关系

### `ReactAgent`
- 最基础
- 最适合看清 Agent 的主循环
- 也是工具执行和恢复逻辑最核心的入口

### `PlanAndSolveAgent`
- 更偏任务拆解
- 核心增量在“规划”和“步骤求解”
- 底层工具执行能力仍复用 `ReactAgent`

### `ReflectionAgent`
- 更偏答案质量控制
- 核心增量在“草稿 -> 审查 -> 修订”
- 同样复用了基础工具能力和上下文注入能力
