# helloAgent

一个从零手写 Agent 的练习项目。

这个仓库的目标不是直接做成“大而全”的 Agent 框架，而是先把最核心的链路一段一段搭起来：

`Prompt -> 推理 -> 工具调用 -> Observation -> 最终回答`

目前项目更像是一个持续演进中的实验场 / 学习记录，适合用来理解不同 Agent 范式在代码层面到底是怎么工作的。

## 当前已实现

目前仓库里已经有这些内容：

- 一个文本解析版 `ReAct Agent`
- 一个 `Plan-and-Solve Agent`
- 一个 `Reflection Agent`
- 一个分层记忆系统：`Working / Episodic / Semantic`
- 一个公共父类 `ReasoningAgentBase`，用于复用消息构建、LLM 请求和运行期历史管理
- 一套轻量的上下文工程抽象：`ContextSection / ContextPacket / ContextBuilder`
- 一个工具基类 `Tool`
- 一个工具注册器 `ToolRegistry`
- 一层向原生 `tool calling` 迁移的 schema 复用能力
- 一个内置示例工具 `get_time`
- 一个内置记忆工具 `memory_tool`
- 一个内置检索工具 `rag_tool`
- 一个统一的 OpenAI-compatible LLM 调用层
- 一个本地可运行的离线 embedding 实现
- 一个可选的本地 `Ollama embedding` 接入层
- 一个 Qdrant 适配层：有真实 Qdrant 配置时优先接入，否则自动回退到本地 JSON
- 一个 Neo4j 图谱适配层：有真实 Neo4j 配置时优先接入，否则自动回退到本地 JSON
- 一个最小可用的本地 RAG pipeline（文档切片 / 索引 / 检索 / 重排 / 上下文拼装）
- 一个 `main.py` 演示入口

## 当前支持的 Agent 范式

### 1. ReAct

核心思路是让模型输出：

```text
Thought: 我需要先获取当前时间。
Action: get_time[]
```

然后由 Agent 自己负责：

- 解析 `Thought / Action`
- 找到对应工具
- 执行工具
- 把结果写回 `Observation`
- 继续下一轮推理

这是一种非常适合学习 Agent 主循环的最小实现方式。

### 2. Plan-and-Solve

先生成计划，再按步骤逐步求解。

这个版本适合观察：

- 计划生成
- 步骤级求解
- 步骤结果汇总
- 最终答案合成

### 3. Reflection

先生成草稿答案，再做一次自我审查，最后按需修订。

当前实现里还额外加了一层保护：

- 反思阶段不能随意改写已经由工具确认过的事实
- 如果修订结果丢失了关键工具事实，会回退到上一版答案

这部分主要是为了减少“模型润色时把工具结果改错”的问题。

## 当前实现风格

这个项目当前仍然以“文本解析版 Agent”为主，但已经开始往原生 tool calling 方向迁移。

也就是说：

- 模型主要输出文本格式的 `Thought / Action`
- Agent 自己负责解析和控制工具执行
- 工具系统已经抽成了对象化结构，并且已经开始复用到原生 tool calling 链路

这样做的好处是：

- 代码路径清晰
- 容易调试
- 容易理解每一步到底发生了什么
- 适合逐步演进到更完整的 Agent 结构

当前迁移到原生 tool calling 的部分是：

- `Tool` 的参数定义会统一导出成 schema
- `ToolRegistry.get_available_tools()` 会直接生成 OpenAI-compatible `tools` 结构
- `ReactAgent` 已支持一条“原生 tool calling 主链路”
- `Plan-and-Solve` 的“步骤求解阶段”已经支持混合 tool calling
- `Reflection` 的“草稿生成阶段”已经支持混合 tool calling
- 规划、反思审查、修订等环节目前仍主要保留文本链路

## 当前的记忆系统进展

目前记忆能力已经走到“最小可用 + 可继续演进”的阶段：

- `WorkingMemory`：纯内存、带 TTL，用来保存最近上下文
- `EpisodicMemory`：基于 SQLite / JSON fallback 的持久化记忆
- `SemanticMemory`：基于“向量检索 + 图谱检索”的双通道语义记忆
- `MemoryManager`：统一协调写入、召回、去重和 prompt 注入
- 已补第一版长期保留策略：高价值记忆优先保留，普通闲聊优先淘汰
- `memory_tool`：支持 `recent / search / context / summary / remember / clear`

当前这套语义检索已经开始兼容真实 Qdrant + Neo4j，当前阶段可以做到：

- 把长期记忆同步写入向量存储
- 把关键实体和关系同步写入图存储
- 在后续 query 中做最小语义召回
- 在没有额外服务依赖时自动回退到本地 JSON
- 同一套向量存储能力同时复用于 `SemanticMemory` 和 `RAG`
- 通过图谱通道额外保留“谁喜欢什么 / 某系统支持什么”这种关系结构
- 当前已经增强了启发式实体抽取与关系归纳，重点覆盖：
  - 用户偏好：如“用户喜欢美式咖啡 / 不喜欢过甜饮品”
  - 能力支持：如“helloAgent 支持 ReAct / Plan-and-Solve / Reflection”
  - 系统组成：如“记忆系统包含 WorkingMemory / EpisodicMemory / SemanticMemory”
- 记忆召回结果已经开始做轻量结构化整理，当前会优先分成：
  - `用户偏好`
  - `项目事实`
  - `近期对话`
- 在原始记忆条目之上，当前还新增了一层“会话摘要”：
  - 长对话时会先提炼一份压缩摘要
  - 摘要当前会优先覆盖 `用户偏好 / 项目事实 / 最近进展`
  - 这样可以减少每轮都把大量原始记忆直接塞进 prompt
- 长期保留策略当前的规则是：
  - `episodic / semantic` 都有独立容量上限
  - 裁剪时不是单纯保留“最近 N 条”，而是优先保留 `偏好 / 事实 / 工具成功结果 / 最终答案`
  - 同价值候选之间再按新近程度排序
  - 最近裁剪结果会写入调试日志，便于观察哪些内容被淘汰

后续还会继续往这些方向补：

- 更多文档格式解析
- 基于检索结果的自动答案合成 / 重排
- 更稳定的图谱实体抽取与关系归纳
- 更细粒度的关系类型与冲突消解

## 当前的上下文工程进展

目前项目里已经开始把“给模型喂什么上下文”从零散字符串拼接，收敛成一个更容易扩展的结构：

- `ContextSection`：表示一段有标题、有来源、有优先级的上下文片段
- `ContextPacket`：负责收集多个片段，并按优先级统一渲染
- `ContextBuilder`：提供更直观的构建接口，用来组合系统提示、运行规则、记忆、检索结果和额外备注
- `ReasoningAgentBase`：现在会优先把 `system prompt + 运行规则 + 相关记忆` 整理成结构化上下文，再交给模型
- `RagPipeline`：除了返回原始检索结果，还会做一个轻量重排，并生成“参考结论 + 证据片段”的结构化检索上下文
- `rag_tool`：新增 `context` 动作，方便后续做更细的 prompt 组装

这部分还不是最终版本，但已经把后面比较重要的演进点留好了位置，例如：

- 上下文裁剪
- 多来源上下文去重
- 更明确的优先级策略
- 检索结果与历史记忆的合并策略

当前这个版本还补了一层非常实用的“预算控制”：

- `ContextPacket.render(...)` 已支持总字符预算、最大 section 数量、单 section 字符上限
- 重复 section 会在渲染前自动去重
- `ReasoningAgentBase` 会按 `Config` 里的预算参数统一裁剪上下文，避免 prompt 无限膨胀

同时，这一层已经开始和 Agent 主循环真正接起来了：

- 如果当前注册了 `rag_tool`，并且本地已经有可检索文档，`ReasoningAgentBase` 会在发起推理前自动补一层 `RAG context`
- 当前运行中已经执行过的工具 Observation，也会被提炼成高优先级 section，再次注入后续轮次
- 这样 `ReAct / Plan-and-Solve / Reflection` 三种范式都能共享同一套“记忆 + 检索 + 工具事实”的上下文拼装逻辑

现在这层还多了一步“轻量上下文路由”：

- 偏个性化 / 偏回忆的问题，会走 `memory_first`
- 偏文档问答 / 偏知识检索的问题，会走 `rag_first`
- 偏工具执行的问题，会走 `tool_first`
- 其他情况走 `balanced`

在 `memory_first` 路由下，系统会更强调：

- 用户偏好
- 项目事实
- 会话摘要
- 近期对话摘要

在 `rag_first` 路由下，系统会更强调：

- 自动 RAG 检索结果
- 证据片段
- 与当前问题直接相关的少量记忆

同时，这一层现在也开始做“冲突消解提示”：

- 如果记忆、RAG、工具观察之间出现明显冲突，系统会额外生成一个 `冲突消解` section
- 当前默认规则是：
  - 工具观察优先于一切
  - 用户偏好冲突时，记忆优先
  - 项目 / 文档事实冲突时，RAG 证据优先
- 这一步的目标不是自动裁决真相，而是防止模型把相反说法强行揉成一个新事实

现在记忆层的整体结构已经大致变成：

- 原始记忆条目
- 结构化记忆分组
- 会话摘要
- 自动路由后的上下文注入

## 当前的 embedding 进展

现在项目里有两种 embedding 方案：

- `hash`：完全离线、零依赖，适合先把链路跑通
- `ollama`：直接复用本地 Ollama embedding 模型，适合提升记忆召回和 RAG 检索质量

如果你本地已经有 Ollama embedding 模型，比如：

- `bge-m3:latest`
- `mxbai-embed-large:latest`

那么可以直接把记忆系统和 RAG 切到真实 embedding。

需要注意的一点是：

- 如果你之前用的是 `hash` embedding，并且已经把 Qdrant collection 建成了 `96` 维
- 现在切到 Ollama embedding 后，向量维度通常会变化
- 这时最好为新的 embedding 模型单独使用新的 Qdrant collection 名称，避免维度不匹配

当前代码已经补了一层保护：

- `QdrantVectorStore` 会检查 collection 的向量维度是否和当前 embedding 一致
- 如果不一致，会给出明确提示，并自动回退到 JSON fallback，而不是直接异常崩掉

## 项目结构

```text
helloAgent/
├─ agents/
│  ├─ agent_base.py            # Agent 抽象基类
│  ├─ reasoning_agent_base.py  # 多种推理型 Agent 的公共父类
│  ├─ react_agent.py           # 文本版 ReAct Agent
│  ├─ plan_and_solve.py        # Plan-and-Solve Agent
│  └─ reflection_agent.py      # Reflection Agent
├─ core/
│  ├─ Config.py                # 配置读取与默认参数
│  ├─ context_engineering.py   # 上下文工程抽象
│  ├─ llm_client.py            # OpenAI-compatible LLM 封装
│  ├─ message.py               # 统一消息结构
│  └─ __init__.py
├─ memory/
│  ├─ base.py                  # 记忆核心数据结构与配置
│  ├─ manager.py               # 统一记忆协调器
│  ├─ embedding.py             # 离线 embedding / 向量相似度能力
│  ├─ types/                   # 各类记忆实现
│  ├─ storage/                 # SQLite / 向量存储适配
│  └─ rag/                     # 本地 RAG 管道与文档处理
├─ tools/
│  └─ builtin/
│     ├─ tool_base.py          # 工具基类
│     ├─ toolRegistry.py       # 工具注册器
│     ├─ get_time.py           # 示例工具：获取本地时间
│     ├─ memory_tool.py        # 记忆工具
│     └─ rag_tool.py           # 本地检索工具
├─ main.py                     # 当前测试入口
├─ PROJECT_DOC.md              # 项目总文档：文件、类、方法作用说明
├─ README.md
└─ LICENSE
```

## 快速开始

### 1. 创建虚拟环境

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux：

```bash
source .venv/bin/activate
```

### 2. 安装依赖

当前建议直接手动安装这些包：

```bash
pip install openai python-dotenv pydantic
```

如果你准备接真实 Qdrant，再额外安装：

```bash
pip install qdrant-client
```

如果你准备接真实 Neo4j，再额外安装：

```bash
pip install neo4j
```

说明：

- 仓库里的依赖清单还没有整理成正式可发布版本
- 后续会补更规范的 `requirements.txt` 或 `pyproject.toml`

### 3. 配置环境变量

项目会从 `.env` 或系统环境变量中读取模型配置。

例如你可以准备一个本地 `.env`：

```env
DEFAULT_PROVIDER=deepseek
DEFAULT_MODEL=deepseek-chat
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

也可以使用更通用的命名：

```env
LLM_PROVIDER=deepseek
LLM_MODEL_ID=deepseek-chat
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.deepseek.com/v1
```

如果你想启用真实 Qdrant，可以继续加上这些可选配置：

```env
VECTOR_STORE_BACKEND=auto
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_SEMANTIC_COLLECTION=semantic_memory
QDRANT_RAG_COLLECTION=rag_chunks
```

如果你不配置 `QDRANT_URL`，或者本地没有安装 `qdrant-client`，项目会自动回退到 JSON 向量存储。

如果你本地已经装了 Ollama，并希望直接使用本地 embedding 模型，可以额外加上：

```env
EMBEDDING_BACKEND=ollama
OLLAMA_EMBEDDING_MODEL=bge-m3:latest
OLLAMA_EMBEDDING_BASE_URL=http://127.0.0.1:11434
OLLAMA_EMBEDDING_TIMEOUT_SECONDS=30
```

如果你同时还在用 Qdrant，建议把 collection 分开，例如：

```env
QDRANT_SEMANTIC_COLLECTION=semantic_memory_bge_m3
QDRANT_RAG_COLLECTION=rag_chunks_bge_m3
```

这样可以避免不同 embedding 维度共用同一个 collection。

如果你想调整上下文工程的裁剪和自动检索行为，也可以继续加上这些可选配置：

```env
CONTEXT_MAX_CHARS=3200
CONTEXT_MAX_SECTIONS=6
CONTEXT_SECTION_MAX_CHARS=1200
AUTO_MEMORY_CONTEXT=true
AUTO_MEMORY_CONTEXT_LIMIT=5
AUTO_RAG_CONTEXT=true
AUTO_RAG_CONTEXT_LIMIT=3
TOOL_CONTEXT_OBSERVATION_LIMIT=4
ENABLE_CONTEXT_CONFLICT_RESOLUTION=true
```

如果你想切换原生 tool calling 模式，也可以增加：

```env
TOOL_CALLING_MODE=text
```

当前可选值可以理解为：

- `text`：继续走原来的文本 `Thought / Action`
- `native`：在已接入的 Agent 阶段上优先启用原生 tool calling
- `auto`：当前行为与 `native` 接近，作为后续扩展保留

如果你想调整会话摘要行为，也可以继续加上这些可选配置：

```env
ENABLE_SESSION_SUMMARY=true
SESSION_SUMMARY_RECENT_ITEMS=12
SESSION_SUMMARY_MIN_MESSAGES=4
SESSION_SUMMARY_MAX_LINES=6
```

如果你想调整长期保留策略，也可以继续加上这些可选配置：

```env
ENABLE_MEMORY_RETENTION=true
EPISODIC_RETENTION_MAX_ITEMS=48
SEMANTIC_RETENTION_MAX_ITEMS=24
RETENTION_KEEP_HIGH_VALUE=true
```

如果你想启用真实 Neo4j，可以继续加上这些可选配置：

```env
GRAPH_STORE_BACKEND=auto
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_DATABASE=neo4j
```

如果你不配置 `NEO4J_URL`，或者本地没有安装 `neo4j` Python 驱动，项目会自动回退到 JSON 图存储。

注意：

- 不要把真实 API Key 提交到 GitHub
- `.env` 只应该保留在本地
- 如果你使用本地模型服务，也可以把 `provider` 和 `base_url` 指向本地兼容接口

### 4. 运行示例

```bash
python main.py
```

当前 `main.py` 默认运行的是“记忆测试工作流”。

如果你想验证本地 RAG 链路，可以在 Python 里手动执行：

```python
import main

main.configure_logging()
main.run_demo("rag")
```

如果你想在“不依赖 LLM 服务”的情况下，单独验证 RAG 入库、检索重排和结构化上下文拼装，可以执行：

```python
import main

main.configure_logging()
main.run_demo("rag_smoke")
```

如果你想先检查当前 embedding 后端本身是否工作正常，可以执行：

```python
import main

main.configure_logging()
main.run_demo("embedding_smoke")
```

如果你想直接看 Agent 在发请求前会拼出什么上下文，可以执行：

```python
import main

main.configure_logging()
main.run_demo("context_smoke")
```

如果你想进一步看“不同问题类型会如何切换上下文优先级”，可以执行：

```python
import main

main.configure_logging()
main.run_demo("routing_smoke")
```

如果你想看“记忆和文档互相矛盾时，系统怎么提示冲突与优先规则”，可以执行：

```python
import main

main.configure_logging()
main.run_demo("conflict_smoke")
```

如果你想看“长对话时会话摘要是怎么生成并注入上下文的”，可以执行：

```python
import main

main.configure_logging()
main.run_demo("summary_smoke")
```

如果你想看“现有 Tool schema 如何复用到原生 tool calling 主链路”，可以执行：

```python
import main

main.configure_logging()
main.run_demo("native_tool_smoke")
```

如果你想看 `Plan-and-Solve` 在“步骤求解阶段”如何混合使用原生 tool calling，可以执行：

```python
import main

main.configure_logging()
main.run_demo("native_plan_smoke")
```

如果你想看 `Reflection` 在“草稿阶段”如何混合使用原生 tool calling，可以执行：

```python
import main

main.configure_logging()
main.run_demo("native_reflection_smoke")
```

如果配置正确，终端会输出：

- 最终答案
- 执行轨迹
- 记忆快照
- 更干净的运行日志

在 `rag` 演示里，还会额外看到：

- 本地示例文档建索引
- `rag_tool` 的检索结果
- 基于检索上下文生成的最终回答

在 `rag_smoke` 演示里，会直接看到：

- 当前启用的向量后端
- 当前启用的 embedding 后端
- 写入的切片数量
- 已索引的来源列表
- `ContextBuilder` 生成的结构化检索上下文

在 `embedding_smoke` 演示里，会直接看到：

- 当前 embedding 后端
- 当前向量维度
- 相近文本相似度
- 无关文本相似度

在 `context_smoke` 演示里，会直接看到：

- 运行规则 section
- 当前轮已确认的工具观察
- 召回到的历史记忆
- 自动注入的 RAG 检索上下文

在 `routing_smoke` 演示里，会直接看到：

- 当前命中的上下文路由类型
- memory / rag / tool 三类上下文的优先级
- 同一个 Agent 在“偏回忆问题”和“偏文档问题”下的不同拼装结果

在 `conflict_smoke` 演示里，会直接看到：

- 当前命中的上下文路由
- `冲突消解` section
- 哪些说法被判定为潜在冲突
- 当前应该优先采用哪一侧来源

在 `summary_smoke` 演示里，会直接看到：

- 单独生成的会话摘要
- 摘要被注入后的完整上下文
- 摘要层与原始结构化记忆层是如何同时存在的

在 `memory_closure_smoke` 演示里，会直接看到：

- 哪些消息会被判定为低价值并跳过
- 哪些重复消息会被拦截
- 哪些内容会进入 working / episodic / semantic
- 最近的记忆写入决策日志
- 长期保留策略裁剪前后的差异
- 最近一次召回结果为什么会被选中

在 `tool_schema_smoke` 演示里，会直接看到：

- 工具自动生成的参数 Schema
- 字符串参数被归一化为 `integer / boolean`
- `minimum / maximum / minLength` 等约束会进入 schema
- action 级跨字段规则也可以走本地语义校验
- 工具执行结果会被包装成统一的 `ToolResult`
- ToolResult 里的 `meta / data` 已开始接入日志、工具观察和工具记忆
- 非法枚举值时的本地校验错误提示

在 `tool_recovery_smoke` 演示里，会直接看到：

- 可重试失败在 Agent 层被自动重试
- 重试成功后的恢复结果
- 重试耗尽后的降级提示
- 工具上下文里如何保留 `attempt / max_attempts / degraded`

在 `native_tool_smoke` 演示里，会直接看到：

- 模型返回的 `tool_calls`
- 工具被执行后的 Observation
- tool message 回填后生成的最终答案
- 公共 native tool calling 执行层是如何驱动整段循环的

在 `native_plan_smoke` 演示里，会直接看到：

- 文本规划输出
- 原生 tool calling 驱动的步骤求解
- 步骤结果再被汇总成最终答案
- `PlanAndSolveAgent` 复用公共 assistant/tool 回填链路后的执行结果

在 `native_reflection_smoke` 演示里，会直接看到：

- 草稿阶段的原生 `tool_calls`
- 工具结果如何成为草稿答案依据
- 反思阶段继续沿用现有文本审查链路
- `ReflectionAgent` 复用公共 native tool loop 后的草稿执行轨迹

如果模型服务不可用或网络配置有问题，`main.py` 会尽量输出简洁错误，而不是直接刷一大段 SDK 堆栈。

## 当前支持的 Provider 方向

`core/llm_client.py` 当前按 OpenAI-compatible 方式封装，已经考虑了多种 provider 的兼容入口，例如：

- OpenAI
- DeepSeek
- Qwen / DashScope
- Zhipu
- Moonshot
- Doubao / Ark
- MiniMax
- Ollama
- vLLM
- 自定义 OpenAI-compatible Base URL

这里的目标不是一次性做深度适配，而是先把消息格式、配置解析和调用接口统一起来。

## 当前限制

这个仓库还在快速迭代，目前有这些限制：

- 仍然偏学习 / 实验用途，不是生产框架
- schema 和原生 tool calling 还没有完全接上
- 工具 schema 当前已经支持类型、默认值、枚举、范围和跨字段语义校验
- 但还没有补到复杂嵌套结构、条件 schema 和更细的约束
- 工具执行结果当前已经有第一版 `ToolResult` 协议，并已开始接入日志与工具记忆
- 失败恢复、自动重试和基础降级提示已经有第一版实现
- 但还没有把熔断、统计、策略化重试等能力全部围绕它重构完
- 原生 tool calling 已覆盖 `ReactAgent / Plan-and-Solve / Reflection`，并抽成了公共执行层
- 但 schema 校验、并行工具调用、失败恢复策略还没有补完整
- 记忆系统当前已经有第一版写入策略、低价值过滤、重复跳过、长期保留和决策日志
- 记忆召回当前已经开始带来源和分数解释
- 但更细粒度的淘汰权重、跨 session 整理和更强的检索解释还没有完全补齐
- 工具系统目前比较轻量
- 测试还不够系统化
- 上下文工程目前还是轻量版，虽然已经有字符预算与去重，但还没有做真正的 token 预算
- 自动 RAG 注入目前还是启发式触发，不是完整的 query planner
- memory / rag / tool 的上下文路由目前也是启发式规则，不是训练得到的策略模型
- 冲突检测目前主要覆盖“用户偏好 / 支持关系 / 包含关系”等常见模式，还不是通用事实校验器
- 会话摘要目前还是规则式摘要，不是基于 LLM 的抽象总结器
- 文档会继续补充，目录结构也可能继续调整

## 接下来准备继续做的事情

后续大概率会继续往这些方向补：

- 工具参数 schema
- 更多内置工具
- 更完整的公共 Agent 抽象
- 更稳定的多轮消息管理
- 更完整的上下文工程策略
- 更完善的测试和示例
- 更真实的向量库 / 图数据库接入

当前阶段性的收尾优先级，我也单独整理到了仓库根目录的 `TODO.md`。

如果你想快速了解整个仓库里“每个文件 / 类 / 方法分别负责什么”，
可以直接先看根目录的 `PROJECT_DOC.md`。

## 为什么公开这个仓库

这个仓库公开的目的，主要是记录“自己手写一个 Agent”的过程。

它目前不是一个已经打磨完成的产品，而更像是：

- 学习记录
- 最小实现样板
- 后续持续迭代的实验场

如果你也在自己手写 Agent，希望这个仓库能给你一些实现层面的参考。

## 安全说明

为了适合公开到 GitHub：

- README 不包含任何真实密钥
- 示例环境变量全部使用占位符
- 当前文档只描述仓库中已经公开的代码与能力

如果你 fork / clone 这个仓库，请务必自行管理好本地 `.env` 和 API Key。

---

README 会随着项目继续迭代而持续更新。
