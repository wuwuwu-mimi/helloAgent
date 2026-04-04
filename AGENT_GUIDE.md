# core 包怎么串起来做一个基础 Agent

这份文档只讲 `core` 目录下这几个基础文件怎么协作：

- `Config.py`: 负责读取配置
- `message.py`: 负责管理消息结构
- `llm_client.py`: 负责调用 OpenAI-compatible 模型接口
- `__init__.py`: 负责统一导出常用对象

目标不是一开始就做复杂框架，而是先把一个“可跑、可调、可扩”的基础 agent 跑通。

## 1. 先理解每个文件的职责

### `Config.py`

它负责两件事：

1. 从环境变量读取默认配置
2. 把这些配置整理成 LLM 客户端可直接使用的参数

你一般会这样用：

```python
from core.Config import Config

config = Config.from_env()
print(config.to_dict())
```

常用环境变量：

```env
DEFAULT_PROVIDER=deepseek
DEFAULT_MODEL=deepseek-chat
TEMPERATURE=0.7
MAX_TOKENS=2048
TIMEOUT=60
MAX_HISTORY_LENGTH=20
```

如果你已经在用这套命名，也兼容：

```env
LLM_PROVIDER=deepseek
LLM_MODEL_ID=deepseek-chat
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.deepseek.com/v1
```

### `Config.py` 里的类和方法作用

#### `Config`

这是整个 agent 的“默认运行配置”对象。  
你可以把它理解成“启动参数集合”，外层 agent 和 LLM client 都会依赖它。

主要字段作用：

- `default_model`: 默认使用哪个模型
- `default_provider`: 默认使用哪个 provider
- `timeout`: 单次请求超时时间
- `temperature`: 默认采样温度
- `max_tokens`: 单次输出 token 上限
- `debug`: 是否开启调试模式
- `log_level`: 日志等级
- `max_history_length`: 历史消息最多保留多少条

#### `Config.from_env()`

作用：从环境变量读取配置，生成一个 `Config` 实例。  
这是最常用的入口，通常项目启动时就会先调它。

#### `Config.llm_options()`

作用：把 `Config` 中和模型调用相关的字段整理成一个字典。  
这个字典可以直接传给 `OpenAICompatibleLLM`，省得你手写映射关系。

#### `Config.trimmed_history(messages)`

作用：裁剪历史消息，避免上下文无限增长。  
这是手写 agent 很重要的一个基础方法，因为消息历史一旦不控长，成本和延迟都会越来越高。

#### `Config.to_dict()`

作用：把配置导出成普通字典，方便打印、调试、记录日志。

#### `_first_env()`

作用：按优先级读取多个环境变量名，返回第一个非空值。  
这属于内部辅助函数，主要是为了把“环境变量优先级”集中管理。

#### `_read_bool()` / `_read_float()` / `_read_int()`

作用：把字符串环境变量解析成目标类型。  
这些也是内部辅助函数，用来避免在 `from_env()` 里堆很多重复代码。

## 2. `message.py` 负责消息

不要一开始就全程传裸 dict。

原因很简单：一旦开始做 agent，你很快就会碰到这些情况：

- system / user / assistant / tool 四种角色
- assistant 发起 `tool_calls`
- tool 返回结果时需要 `tool_call_id`
- 后面想打印日志、保存历史、做裁剪

这些信息如果全靠散乱 dict 去管理，很快就会乱。

推荐用法：

```python
from core.message import Message

history = [
    Message.system("你是一个乐于助人的助手。"),
    Message.user("帮我解释一下什么是 agent。"),
]
```

真正发给模型前，不需要你自己手动转字典，`llm_client.py` 会自动调用 `normalize_messages()`。

### `message.py` 里的类和方法作用

#### `ToolFunction`

作用：表示一次工具调用中的函数部分。  
它只关心两件事：

- `name`: 工具名
- `arguments`: 工具参数，保持字符串格式

这里把 `arguments` 保持为字符串，是为了和 OpenAI-compatible 接口的原始格式一致。

#### `ToolCall`

作用：表示 assistant 发起的一次完整工具调用。  
它比 `ToolFunction` 多一层包装，包含：

- `id`: 这次工具调用的唯一标识
- `type`: 通常是 `function`
- `function`: 真正的工具函数定义

#### `Message`

作用：表示一条标准消息。  
这是 agent 维护历史记录时最核心的对象。

主要字段作用：

- `role`: 消息角色，支持 `system` / `user` / `assistant` / `tool`
- `content`: 文本内容
- `name`: 工具名或额外名称
- `tool_call_id`: tool 消息要回填给哪一次工具调用
- `tool_calls`: assistant 想调用哪些工具
- `timestamp`: 时间戳
- `metadata`: 额外元数据

#### `Message.system(content)`

作用：快速创建 system 消息。  
适合写系统提示词，例如“你是一个严谨的中文助手”。

#### `Message.user(content)`

作用：快速创建 user 消息。  
一般每次用户输入都会包装成它。

#### `Message.assistant(content=None, tool_calls=None)`

作用：快速创建 assistant 消息。  
它既可以表示普通文本回复，也可以表示“模型决定调用工具”的消息。

#### `Message.tool(content, tool_call_id, name=None)`

作用：快速创建 tool 消息。  
当你的 Python 工具函数执行完成后，应该把工具结果包装成这种消息再送回历史中。

#### `Message.to_chat_message()`

作用：把 `Message` 对象转成真正发给模型接口的字典结构。  
这一步很关键，因为模型接口需要的是标准 JSON 结构，而不是 Python 对象。

#### `Message.from_chat_message(payload)`

作用：把模型接口风格的字典反向转成 `Message`。  
适合在你收到模型响应后，重新塞回自己的消息历史。

#### `Message.short()`

作用：生成一段适合日志打印的短文本。  
当你后面开始调 agent 行为时，这种简短摘要会非常有用。

#### `normalize_messages(messages)`

作用：把 `Message` 列表或 dict 列表统一转换成模型可接受的消息列表。  
这是 `Message` 层和 `LLM` 层之间的桥。

#### `trim_messages(messages, max_length)`

作用：单独在消息层做历史裁剪。  
如果你不想依赖 `Config`，也可以直接用它来控制消息条数。

## 3. `llm_client.py` 负责和模型通信

推荐直接通过 `Config` 来初始化：

```python
from core.Config import Config
from core.llm_client import HelloAgentsLLM

config = Config.from_env()
llm = HelloAgentsLLM.from_config(config)
```

也可以直接手动传：

```python
llm = OpenAICompatibleLLM(
    provider="deepseek",
    model="deepseek-chat",
    api_key="your-api-key",
    base_url="https://api.deepseek.com/v1",
    temperature=0.7,
)
```

`chat()` 返回的是 `ChatResult`，里面会带：

- `text`
- `reasoning`
- `tool_calls`
- `finish_reason`

如果你只想拿字符串，也可以继续用 `think()`。

### `llm_client.py` 里的类和方法作用

这一部分是 `core` 里最值得慢慢读的文件，因为它决定了：

- 配置怎么落到真实请求上
- 不同 provider 怎么切换
- 模型返回结果怎么被统一处理

#### `ProviderSpec`

作用：描述一个 provider 的“静态信息”。  
例如：

- 默认 `base_url`
- 这个 provider 对应哪些环境变量
- 是否必须提供 `api_key`

你可以把它理解成 provider 的说明书。

#### `LLMConfig`

作用：表示“最终解析完成后的模型配置”。  
它和 `Config` 的区别是：

- `Config` 更像项目默认配置
- `LLMConfig` 更像一次实际请求真正使用的配置

#### `ChatResult`

作用：统一封装一次模型调用的结果。  
这样外层 agent 不用关心底层 SDK 细节，只用看：

- `text`
- `reasoning`
- `tool_calls`
- `finish_reason`
- `raw`

#### `OpenAICompatibleLLM`

作用：这是整个模型调用层的核心类。  
只要目标服务兼容 OpenAI 风格接口，就可以尽量复用这一层。

#### `OpenAICompatibleLLM.from_config(config, **overrides)`

作用：从 `Config` 构造 LLM 客户端。  
这是把 `Config.py` 和 `llm_client.py` 串起来的推荐入口。

#### `OpenAICompatibleLLM.available_providers()`

作用：返回当前内置支持的 provider 名称列表。  
适合调试、校验用户输入、或者做命令行提示。

#### `OpenAICompatibleLLM.think(messages, ...)`

作用：快速拿文本结果。  
它适合最简单的聊天场景，本质上是 `chat()` 的轻量封装。

#### `OpenAICompatibleLLM.chat(messages, ...)`

作用：统一聊天入口。  
如果你后面自己写 agent，大多数时候应该直接用这个方法，因为它会返回完整的 `ChatResult`。

#### `OpenAICompatibleLLM._build_request(...)`

作用：把 Python 侧参数组装成发送给模型接口的请求体。  
这是“模型请求长什么样”的核心方法。

重点理解它做了什么：

- 把 `Message` 转成 dict
- 合并默认参数和显式参数
- 只在参数存在时才放进请求体

#### `OpenAICompatibleLLM._consume_response(response)`

作用：处理非流式响应。  
如果你传的是 `stream=False`，模型返回结果会走这里。

#### `OpenAICompatibleLLM._consume_stream(response, ...)`

作用：处理流式响应。  
它会把分块返回的文本、reasoning、tool_calls 重新拼成一份完整结果。

#### `OpenAICompatibleLLM._resolve_provider(...)`

作用：判断本次到底该用哪个 provider。  
这个方法是解决“provider 串线”问题的关键之一。

#### `OpenAICompatibleLLM._resolve_config(...)`

作用：根据 provider、环境变量和显式参数，生成最终的 `LLMConfig`。  
这是“配置优先级”真正落地的地方。

理解它非常重要，因为很多多模型 bug 都出在这里。

#### `OpenAICompatibleLLM._detect_provider_from_env()`

作用：根据 provider 专属环境变量猜测 provider。  
它故意不直接依赖通用 `LLM_*`，就是为了减少串线。

#### `OpenAICompatibleLLM._normalize_provider(provider)`

作用：把别名统一成内部标准 provider 名称。  
比如 `ark` 会归一成 `doubao`。

#### `OpenAICompatibleLLM._infer_provider_from_base_url(base_url)`

作用：当你没有显式传 `provider` 时，从 URL 猜一个最可能的 provider。

#### `OpenAICompatibleLLM._extract_reasoning(obj)`

作用：从不同 provider 的返回结构里提取 reasoning 字段。  
因为不同模型厂商的命名不统一，所以需要这一层兼容。

#### `OpenAICompatibleLLM._content_to_text(content)`

作用：把模型返回的各种 `content` 格式统一转成纯文本。  
有些 provider 返回字符串，有些返回列表，这个方法负责抹平差异。

#### `OpenAICompatibleLLM._merge_stream_value(current, incoming)`

作用：兼容两种流式模式：

1. 每次 chunk 只给新增内容
2. 每次 chunk 给截至当前的完整内容

这个方法的目的就是避免流式输出出现重复拼接。

#### `_serialize_tool_call()` / `_first_nonempty()` / `_clean()` / `_read_value()`

作用：这些是内部辅助方法。  
它们分别负责：

- 序列化工具调用
- 取第一个非空值
- 清理字符串
- 同时兼容对象和字典取值

如果你是学习阶段，很建议把这些小方法也看一遍，因为好的基础层代码往往就是靠这些“小工具函数”保持可读性的。

## 3.5 `__init__.py` 的作用

`core/__init__.py` 的作用很简单：统一导出常用对象。  
这样你在外层代码里可以直接写：

```python
from core import Config, Message, HelloAgentsLLM
```

而不需要每次都分别从不同文件导入。

## 4. 最小可运行版本：无工具 Agent

这是最简单的一版，只维护消息历史，不做工具调用。

```python
from core.Config import Config
from core.llm_client import HelloAgentsLLM
from core.message import Message


class SimpleAgent:
    def __init__(self) -> None:
        self.config = Config.from_env()
        self.llm = HelloAgentsLLM.from_config(self.config)
        self.history = [
            Message.system("你是一个简洁、准确的中文助手。")
        ]

    def reply(self, user_input: str) -> str:
        self.history.append(Message.user(user_input))
        self.history = self.config.trimmed_history(self.history)

        result = self.llm.chat(self.history)

        self.history.append(Message.assistant(result.text))
        return result.text


if __name__ == "__main__":
    agent = SimpleAgent()
    print(agent.reply("请解释一下什么是手写 agent。"))
```

这个版本已经能满足：

- 读配置
- 维护消息历史
- 调用模型
- 把回复继续放回历史里

## 5. 进阶版本：带工具调用的 Agent

如果你后面想自己写工具系统，核心循环通常长这样：

1. 用户消息进入历史
2. 调用模型
3. 如果模型返回 `tool_calls`
4. 执行工具
5. 把工具结果作为 `tool` 消息放回历史
6. 再次调用模型
7. 直到模型返回普通文本答案

一个简化示例：

```python
import json

from core.Config import Config
from core.llm_client import HelloAgentsLLM
from core.message import Message, ToolCall, ToolFunction


def get_weather(city: str) -> str:
    return f"{city} 今天晴，25 度。"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"}
                },
                "required": ["city"],
            },
        },
    }
]


class ToolAgent:
    def __init__(self) -> None:
        self.config = Config.from_env()
        self.llm = HelloAgentsLLM.from_config(self.config)
        self.history = [
            Message.system("你可以在需要时调用工具来回答问题。")
        ]

    def run(self, user_input: str) -> str:
        self.history.append(Message.user(user_input))

        while True:
            self.history = self.config.trimmed_history(self.history)
            result = self.llm.chat(self.history, tools=TOOLS)

            if not result.tool_calls:
                self.history.append(Message.assistant(result.text))
                return result.text

            assistant_calls = []
            for call in result.tool_calls:
                assistant_calls.append(
                    ToolCall(
                        id=call["id"],
                        function=ToolFunction(
                            name=call["function"]["name"],
                            arguments=call["function"]["arguments"],
                        ),
                    )
                )

            self.history.append(
                Message.assistant(content=result.text or None, tool_calls=assistant_calls)
            )

            for call in result.tool_calls:
                name = call["function"]["name"]
                arguments = json.loads(call["function"]["arguments"] or "{}")

                if name == "get_weather":
                    tool_output = get_weather(arguments["city"])
                else:
                    tool_output = f"未知工具: {name}"

                self.history.append(
                    Message.tool(
                        content=tool_output,
                        tool_call_id=call["id"],
                        name=name,
                    )
                )
```

这就是一个最基础的“手写 tool-calling agent”骨架。

## 6. 推荐的数据流

建议你后面都按这个方向组织：

```text
用户输入
  -> Message.user(...)
  -> history
  -> OpenAICompatibleLLM.chat(history, tools=...)
  -> ChatResult
  -> 如果有 tool_calls，就执行工具并追加 Message.tool(...)
  -> 如果没有 tool_calls，就追加 Message.assistant(...)
  -> 返回最终答案
```

这样设计有三个好处：

1. 配置、消息、模型调用三层职责清楚
2. 后面要加日志、记忆、工具、RAG 都有扩展点
3. 你不会很快被一堆裸 dict 和 if/else 淹没

## 7. 你现在最适合继续补的东西

如果你接下来准备继续手写 agent，我建议优先做这三个模块：

### A. `tools/` 里的工具注册器

你需要一个统一的工具入口，比如：

```python
TOOL_REGISTRY = {
    "get_weather": get_weather,
    "search_docs": search_docs,
}
```

然后把“根据 `tool_calls` 找到函数并执行”的逻辑抽出去。

### B. agent 主循环

把“历史管理 + tool loop + 最终返回”封成一个 `Agent` 类。

### C. 日志和调试

至少打印这些信息：

- 当前 provider / model
- 本轮请求消息数
- 本轮是否触发工具
- 工具输入输出
- 模型最终文本

这样你后面排查问题会轻松很多。

## 8. 常见坑

### provider 串线

如果你同时尝试多个模型厂商，尽量优先使用：

- 显式 `provider=...`
- 显式 `model=...`
- provider 专属环境变量

不要一开始就完全依赖一套通用 `LLM_*`，否则很容易在切 provider 时串线。

### tool 参数不是合法 JSON

模型返回的 `arguments` 只是“看起来像 JSON”的字符串，执行前最好先：

```python
json.loads(arguments)
```

必要时自己做异常处理和容错。

### 历史无限增长

上下文历史如果一直不裁剪，成本和延迟会越来越高。  
先用 `Config.max_history_length` 做简单限制就够用了。

## 9. 一个推荐的起步顺序

如果你现在是边学边写，我建议按这个顺序推进：

1. 先让 `SimpleAgent` 跑通
2. 再给它加 `tool_calls`
3. 再抽出工具注册器
4. 再做记忆、RAG、规划等高级能力

不要一开始就把“多工具、多轮规划、长期记忆、反思”全堆上去，不然调试会非常痛苦。

---

如果你愿意，我下一步可以继续直接帮你补一个 `agents/simple_agent.py`，把这份文档里的最小 agent 真正落成代码。
