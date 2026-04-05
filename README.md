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
- 一个工具基类 `Tool`
- 一个工具注册器 `ToolRegistry`
- 一个内置示例工具 `get_time`
- 一个内置记忆工具 `memory_tool`
- 一个内置检索工具 `rag_tool`
- 一个统一的 OpenAI-compatible LLM 调用层
- 一个本地可运行的离线 embedding 实现
- 一个 Qdrant 适配层：有真实 Qdrant 配置时优先接入，否则自动回退到本地 JSON
- 一个 Neo4j 图谱适配层：有真实 Neo4j 配置时优先接入，否则自动回退到本地 JSON
- 一个最小可用的本地 RAG pipeline（文档切片 / 索引 / 检索）
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

这个项目当前采用的是“文本解析版 Agent”，而不是完全依赖原生 function calling。

也就是说：

- 模型主要输出文本格式的 `Thought / Action`
- Agent 自己负责解析和控制工具执行
- 工具系统已经抽成了对象化结构，但 schema / 原生 tool calling 还没有完全接上

这样做的好处是：

- 代码路径清晰
- 容易调试
- 容易理解每一步到底发生了什么
- 适合逐步演进到更完整的 Agent 结构

## 当前的记忆系统进展

目前记忆能力已经走到“最小可用 + 可继续演进”的阶段：

- `WorkingMemory`：纯内存、带 TTL，用来保存最近上下文
- `EpisodicMemory`：基于 SQLite / JSON fallback 的持久化记忆
- `SemanticMemory`：基于“向量检索 + 图谱检索”的双通道语义记忆
- `MemoryManager`：统一协调写入、召回、去重和 prompt 注入
- `memory_tool`：支持 `recent / search / remember / clear`

当前这套语义检索已经开始兼容真实 Qdrant + Neo4j，当前阶段可以做到：

- 把长期记忆同步写入向量存储
- 把关键实体和关系同步写入图存储
- 在后续 query 中做最小语义召回
- 在没有额外服务依赖时自动回退到本地 JSON
- 同一套向量存储能力同时复用于 `SemanticMemory` 和 `RAG`
- 通过图谱通道额外保留“谁喜欢什么 / 某系统支持什么”这种关系结构

后续还会继续往这些方向补：

- 更多文档格式解析
- 基于检索结果的自动答案合成 / 重排
- 更稳定的图谱实体抽取与关系归纳

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

如果配置正确，终端会输出：

- 最终答案
- 执行轨迹
- 记忆快照
- 更干净的运行日志

在 `rag` 演示里，还会额外看到：

- 本地示例文档建索引
- `rag_tool` 的检索结果
- 基于检索上下文生成的最终回答

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
- 工具系统目前比较轻量
- 测试还不够系统化
- 文档会继续补充，目录结构也可能继续调整

## 接下来准备继续做的事情

后续大概率会继续往这些方向补：

- 工具参数 schema
- 更多内置工具
- 更完整的公共 Agent 抽象
- 更稳定的多轮消息管理
- 更完善的测试和示例
- 更真实的向量库 / 图数据库接入

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
