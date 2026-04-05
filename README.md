# helloAgent

一个从零手写 Agent 的练习项目。

这个仓库的目标不是做一个“大而全”的框架，而是把 Agent 最核心的链路一步一步手写出来，方便理解：

`Prompt -> 推理 -> 工具调用 -> Observation -> 最终回答`

当前这个版本已经完成了核心闭环，可以作为一个适合学习和参考的阶段性最终版。

## 这个仓库是什么

helloAgent 是一个面向学习和实验的 Agent 项目，重点不是“封装得多完整”，而是“代码路径足够清楚”。

你可以用它来理解：

- 一个 Agent 是怎么跑起来的
- 工具调用是怎么接进去的
- 记忆系统是怎么分层的
- RAG 是怎么接进上下文的
- 不同 Agent 范式在代码里到底有什么差异

## 当前功能

目前仓库已经有这些能力：

- 文本解析版 `ReAct Agent`
- `Plan-and-Solve Agent`
- `Reflection Agent`
- 公共推理父类 `ReasoningAgentBase`
- 工具基类 `Tool` 和工具注册器 `ToolRegistry`
- 原生 `tool calling` 主链路
- 统一工具结果协议 `ToolResult`
- 工具自动重试与降级提示
- 分层记忆系统：`Working / Episodic / Semantic`
- 记忆写入策略、低价值过滤、重复过滤、长期保留策略
- 记忆召回解释与会话摘要
- 本地最小可用 RAG pipeline
- 离线 hash embedding
- 可选的本地 `Ollama embedding`
- Qdrant 适配层（不可用时自动回退本地 JSON）
- Neo4j 适配层（不可用时自动回退本地 JSON）
- 一套轻量上下文工程抽象：`ContextSection / ContextPacket / ContextBuilder`
- 多组 smoke test / demo 入口

## 文档导航

详细说明已经拆到 `docs/`：

- `PROJECT_DOC.md`：根目录文档索引
- `docs/architecture.md`：项目结构、core 模块、阅读顺序
- `docs/agents.md`：各类 Agent 与方法说明
- `docs/memory.md`：记忆、RAG、存储层说明
- `docs/tools.md`：工具系统说明
- `TODO.md`：当前任务收尾情况

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

当前最小依赖：

```bash
pip install openai python-dotenv pydantic
```

如果你要接真实 Qdrant：

```bash
pip install qdrant-client
```

如果你要接真实 Neo4j：

```bash
pip install neo4j
```

### 3. 配置环境变量

最小示例：

```env
LLM_PROVIDER=deepseek
LLM_MODEL_ID=deepseek-chat
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.deepseek.com/v1
```

如果你使用本地 Ollama embedding，也可以增加：

```env
EMBEDDING_BACKEND=ollama
OLLAMA_EMBEDDING_MODEL=bge-m3:latest
OLLAMA_EMBEDDING_BASE_URL=http://127.0.0.1:11434
```

## 运行

默认入口：

```bash
python main.py
```

也可以在 Python 里手动运行不同 demo：

```python
import main

main.configure_logging()
main.run_demo("tool_schema_smoke")
```

常用 demo：

- `react`
- `plan`
- `reflection`
- `memory`
- `memory_closure_smoke`
- `rag_smoke`
- `context_smoke`
- `summary_smoke`
- `tool_schema_smoke`
- `tool_recovery_smoke`
- `native_tool_smoke`
- `native_plan_smoke`
- `native_reflection_smoke`

## 当前状态

- `P0` 已经收完
- 当前版本已经可以作为阶段性最终版
- 后续更偏向文档整理和展示，而不是继续扩功能

## 适合谁看

这个仓库比较适合：

- 想手写一个自己的 Agent 的人
- 想看清 Agent 主循环而不是直接用黑盒框架的人
- 想把工具、记忆、RAG、上下文工程串起来理解的人

## 安全说明

为了适合公开到 GitHub：

- README 不包含任何真实密钥
- 示例环境变量全部使用占位符
- 请不要把本地 `.env` 和真实 API Key 提交到仓库
