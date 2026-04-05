# PROJECT_DOC

修改说明：根目录保留这份总索引文档，方便第一次进入仓库时快速找到各模块说明；更详细的内容已经拆分到 `docs/` 目录。

## 文档导航

- `docs/architecture.md`
  - 项目整体结构
  - 根目录文件说明
  - `core/` 模块说明
  - 模块之间的依赖关系与推荐阅读顺序
- `docs/agents.md`
  - `agents/` 目录下各文件、类、方法作用
  - `React / Plan-and-Solve / Reflection` 三种范式说明
- `docs/memory.md`
  - `memory/`、`memory/rag/`、`memory/storage/` 下的文件、类、方法作用
  - 记忆分层、RAG、向量库和图谱适配说明
- `docs/tools.md`
  - `tools/` 下的工具基类、注册器和内置工具说明

## 当前推荐阅读顺序

1. `README.md`
2. `main.py`
3. `docs/architecture.md`
4. `docs/agents.md`
5. `docs/memory.md`
6. `docs/tools.md`

## 文档目标

这些文档主要服务三类场景：

- 自己后续继续维护时，快速定位某个模块应该从哪里改
- 对外公开时，让读者更快看懂项目分层
- 在准备收最终版之前，保留一份较稳定的结构说明
