# helloAgent

**一个Agent练习项目** 帮助理解Agent的工作方式

这个仓库当前有
从零搭一个最小可运行的 Agent 骨架，先把 `prompt -> 推理 -> 工具调用 -> 观察结果 -> 最终回答` 这条链路跑通，再逐步往上加能力。

目前它还不是一个完整框架，更像是一个持续演进中的实验项目 / 学习记录。

## 当前状态

当前已经有的内容：

- 一个最小可运行的文本版 ReAct Agent
- 一个简单的工具注册器 `ToolRegistry`
- 一个内置示例工具 `get_time`
- 一个统一的 LLM 调用层，兼容 OpenAI-compatible 接口
- 一个 `main.py` 测试入口，用来验证 Agent 是否能正确调用工具

当前示例流程是：

1. 用户提问“现在几点”
2. Agent 让模型输出 `Thought` / `Action`
3. 模型选择调用 `get_time[]`
4. Agent 执行工具，把结果写回 `Observation`
5. 模型根据观察结果生成最终答案

## 目前的实现思路

这个项目当前采用的是“文本解析版 ReAct”方式，而不是原生 function calling。

也就是说，模型不会直接返回结构化 `tool_calls`，而是被要求输出类似下面的文本：

```text
Thought: 我需要先获取当前时间。
Action: get_time[]
```

然后由 Agent 自己完成：

- 解析 `Thought` / `Action`
- 识别工具名和参数
- 执行工具
- 把工具结果追加为 `Observation`
- 继续下一轮推理

这样做的好处是结构简单、便于学习，也更容易看清 Agent 的主循环到底在做什么。

## 项目结构

当前目录大致如下：

```text
helloAgent/
├─ agents/
│  ├─ agentBase.py          # Agent 基类
│  └─ react_agent.py        # 文本版 ReAct Agent 主逻辑
├─ core/
│  ├─ Config.py             # 配置读取与默认参数
│  ├─ llm_client.py         # OpenAI-compatible LLM 封装
│  ├─ message.py            # 统一消息结构
│  └─ __init__.py
├─ tools/
│  └─ builtin/
│     ├─ get_time.py        # 示例工具：获取本地时间
│     └─ toolRegistry.py    # 工具注册器
├─ main.py                  # 当前测试入口
└─ README.md
```

## 快速开始

### 1. 创建虚拟环境

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 2. 安装依赖

当前代码至少依赖这些包：

```bash
pip install openai python-dotenv pydantic
```

说明：

- 仓库里目前还没有完整、稳定的公开依赖清单
- 后续我会再整理正式的 `requirements.txt` 或 `pyproject.toml`

### 3. 配置环境变量

项目会从 `.env` 或系统环境变量中读取模型配置。

最简单的方式是准备一个本地 `.env` 文件，例如：

```env
DEFAULT_PROVIDER=deepseek
DEFAULT_MODEL=deepseek-chat
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

也可以使用更通用的命名，例如：

```env
LLM_PROVIDER=deepseek
LLM_MODEL_ID=deepseek-chat
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.deepseek.com
```

注意：

- 不要把真实的 API Key 提交到 GitHub
- `.env` 只应该留在本地开发环境中

### 4. 运行当前示例

```bash
python main.py
```

运行后，你应该能看到：

- 当前测试问题
- Agent 的最终回答
- 每一轮的 `Thought / Action / Observation` 历史
- 终端中的调试日志

## 当前支持的 Provider 方向

`core/llm_client.py` 当前按 OpenAI-compatible 方式封装，代码里已经考虑了多种 provider 的兼容入口，例如：

- OpenAI
- DeepSeek
- Qwen / DashScope
- Zhipu
- Moonshot
- Doubao / Ark
- MiniMax
- Ollama
- vLLM
- 以及自定义 OpenAI-compatible Base URL

这里的目标不是“每家都做深度适配”，而是先统一消息格式和调用接口。

## 当前限制

这是一个还在继续补功能的仓库，目前有这些限制：

- 主要还是学习 /实验性质，还不算生产可用
- 工具系统目前比较轻，只支持最基础的注册和调用
- 目前主流程是文本版 ReAct，还没有全面切到原生 schema / tool_calls
- 没有系统化测试
- 文档会继续补充，代码结构也还会调整

## 接下来准备继续做的事情

后续大概率会逐步补这些方向：

- 工具参数 schema
- 更多内置工具
- 更清晰的日志和调试输出
- 更稳定的多轮消息管理
- 更完善的 README / 示例 / 测试
- 可能加入 memory / planning / RAG 等能力

## 为什么公开这个仓库

这个仓库公开的目的，主要是记录“自己手写一个 Agent”这件事的过程。

它不是一个已经打磨完成的成品，而是一个：

- 学习记录
- 最小实现样板
- 后续持续迭代的实验场

如果你也在手写自己的 Agent，希望这个仓库未来能给你一点参考。

## 安全说明

为了适合公开到 GitHub：

- README 不包含任何真实密钥
- 示例环境变量全部使用占位符
- 当前文档只描述公开代码中的内容

如果你 fork / clone 这个仓库，请务必自己管理好本地 `.env` 和 API Key。

## License

当前仓库使用项目内已有的许可证文件，请参考 `LICENSE`。

---

后续这个 README 会随着项目演进继续更新。
