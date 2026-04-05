# 记忆与检索模块说明

修改说明：这个文档聚焦 `memory/` 目录，包括分层记忆、embedding、RAG、持久化存储以及长期保留策略。

## 1. 记忆系统整体结构

当前记忆系统分成三层：

- `WorkingMemory`
  - 纯内存
  - 带 TTL
  - 保存最近上下文
- `EpisodicMemory`
  - 持久化情景记忆
  - 当前基于 SQLite / JSON fallback
- `SemanticMemory`
  - 语义记忆
  - 当前由“向量检索 + 图谱检索”双通道组成

最上层由 `MemoryManager` 统一协调：
- 写入策略
- 低价值过滤
- 重复过滤
- 召回解释
- 会话摘要
- 长期保留裁剪

## 2. `memory/__init__.py`
- 作用：统一导出记忆相关核心对象。

## 3. `memory/base.py`
- 作用：定义记忆系统的核心数据结构、配置和抽象接口。

### `MemoryItem`
- 作用：表示一条统一的记忆数据。
- `is_expired`：判断当前记忆是否已过期。

### `MemoryConfig`
- 作用：集中管理记忆系统配置。
- `from_env`：从环境变量构建配置。
- `working_expires_at`：计算工作记忆的默认过期时间。

### `BaseMemory`
- 作用：各种记忆类型的统一抽象接口。
- `add`：写入记忆。
- `recent`：获取最近记忆。
- `search`：检索相关记忆。
- `clear`：清空 session 记忆。

### 辅助函数
- `_first_env`：读取首个有效环境变量值。
- `_read_bool`：解析布尔值。
- `_read_int`：解析整数值。
- `_read_float`：解析浮点值。

## 4. `memory/embedding.py`
- 作用：提供 embedding 能力。

### `BaseEmbeddingService`
- 作用：embedding 服务抽象基类。
- `embed`：单条文本转向量。
- `embed_many`：批量转向量。
- `dimension_hint`：返回向量维度提示。
- `cosine_similarity`：计算余弦相似度。

### `HashEmbeddingService`
- 作用：离线哈希 embedding，实现零依赖的本地向量能力。
- `__init__`：初始化维度和概念标签信息。
- `embed`：生成哈希向量。
- `dimension_hint`：返回固定维度。
- `_extract_units`：抽取 token、双字片段和概念标签。

### `OllamaEmbeddingService`
- 作用：调用本地 Ollama embedding 模型生成真实向量。
- `__init__`：初始化模型名、base_url、超时等。
- `embed`：为单条文本生成向量。
- `embed_many`：批量生成向量。
- `dimension_hint`：返回维度提示。
- `_request_single_embedding`：请求一次单文本 embedding。
- `_post_json`：发送 JSON 请求。
- `_normalize_vector`：归一化返回向量。

### `EmbeddingServiceFactory`
- 作用：根据配置创建对应的 embedding 服务。
- `create`：返回实际 embedding 实现。

### `SimpleEmbeddingService`
- 作用：兼容旧结构的轻量别名或占位类型。

## 5. `memory/manager.py`
- 作用：当前记忆系统的核心控制器。

### `MemoryManager`
- 作用：统一协调写入、检索、解释、摘要和长期保留。
- `__init__`：初始化三类记忆、embedding 服务和调试日志。
- `record_message`：按策略把消息写入 working / episodic / semantic。
- `recall`：组合多层记忆进行召回。
- `build_recall_diagnostics`：输出召回解释。
- `build_memory_prompt`：把召回结果渲染成 prompt 文本。
- `build_structured_memory_sections`：把召回结果整理成分组结构。
- `build_structured_memory_prompt`：把结构化记忆渲染成文本。
- `build_session_summary`：生成规则式会话摘要。
- `clear_session`：清空某个 session 的全部记忆与日志。
- `build_memory_diagnostics`：输出写入决策日志。
- `build_retention_diagnostics`：输出长期保留裁剪日志。
- `_dedupe_items`：对召回结果去重。
- `_classify_memory_item`：把记忆项分到偏好、事实、近期对话等桶。
- `_trim_summary_line`：裁剪过长文本。
- `_annotate_recall_source`：给召回项补来源信息。
- `_merge_recall_sources`：合并召回来源。
- `_build_recall_reason`：生成召回原因说明。
- `_render_recall_explanation`：生成短的召回解释文本。
- `_plan_memory_record`：在写入前决定应进入哪些记忆层。
- `_build_memory_plan`：构造标准写入计划结构。
- `_record_memory_decision`：记录写入决策。
- `_apply_retention`：统一执行长期保留裁剪。
- `_apply_single_retention`：对单个存储层执行裁剪。
- `_select_items_to_keep`：选择应该保留的条目。
- `_score_retention_priority`：给 retention 计算优先级。
- `_record_retention_result`：记录 retention 结果。
- `_has_recent_duplicate`：检查最近窗口内是否有重复。
- `_classify_content_kind`：判定内容类型。
- `_score_memory_value`：给消息打高、中、低价值标签。
- `_is_low_value_message`：过滤低价值文本。
- `_should_store_in_semantic`：判断是否应进入语义记忆。
- `_normalize_memory_text`：归一化文本，便于查重。

## 6. `memory/types/` 目录

### `memory/types/__init__.py`
- 作用：记忆类型子包标记文件。

### `memory/types/working.py`
- 作用：实现纯内存工作记忆。

#### `WorkingMemory`
- `__init__`：初始化按 session 分组的内存队列。
- `add`：添加工作记忆，并按容量裁剪。
- `recent`：获取最近若干条记忆。
- `search`：做轻量文本检索。
- `clear`：清空某个 session 的工作记忆。
- `_cleanup`：移除过期记忆。

### `memory/types/episodic.py`
- 作用：封装情景记忆。

#### `EpisodicMemory`
- `__init__`：绑定底层 `DocumentStore`。
- `add`：写入情景记忆。
- `recent`：读取最近情景记忆。
- `search`：检索情景记忆。
- `list_all`：返回某个 session 的全部情景记忆。
- `prune`：执行 retention 裁剪。
- `clear`：清空 session 情景记忆。

### `memory/types/semantic.py`
- 作用：封装语义记忆。

#### `SemanticMemory`
- `__init__`：绑定向量存储、图存储和 embedding 服务。
- `add`：写入语义记忆，同时同步向量和图谱。
- `recent`：读取最近语义记忆。
- `search`：组合向量和图谱结果做检索。
- `list_all`：返回某个 session 的全部语义记忆。
- `prune`：同时裁剪向量层和图谱层。
- `clear`：清空某个 session 的语义记忆。
- `_merge_items`：合并去重多通道召回结果。

### `memory/types/perceptual.py`
- 作用：当前仍是感知记忆占位骨架。

#### `PerceptualMemory`
- `add`：占位写入。
- `recent`：占位读取最近内容。
- `search`：占位检索。
- `clear`：占位清空。

## 7. `memory/storage/` 目录

### `memory/storage/__init__.py`
- 作用：存储子包标记文件。

### `memory/storage/document_store.py`
- 作用：基于 SQLite / JSON fallback 的持久化文档存储。

#### `DocumentStore`
- `__init__`：初始化数据库与 fallback 文件路径。
- `_connect`：建立 SQLite 连接。
- `_initialize`：创建表和索引。
- `add_item`：写入持久化记忆。
- `list_recent`：读取最近若干条记忆。
- `list_session_items`：读取某个 session 的全部持久化记忆。
- `search_items`：按简单文本匹配检索记忆。
- `clear_session`：清空指定 session。
- `prune_session`：按 keep id 裁剪记录。
- `_load_json_items`：从 JSON fallback 读数据。
- `_save_json_items`：写回 JSON fallback。
- `_activate_json_fallback`：切换到 fallback 模式。
- `_row_to_item`：把数据库行转成 `MemoryItem`。

### `memory/storage/qdrant_store.py`
- 作用：Qdrant / JSON fallback 向量存储适配层。

#### `QdrantVectorStore`
- `__init__`：初始化配置、embedding、collection 和 fallback 路径。
- `upsert`：兼容语义记忆接口写入记录。
- `search`：兼容语义记忆接口检索记录。
- `list_recent`：读取最近向量记忆。
- `list_session_items`：读取某个 session 的全部向量记忆。
- `clear_session`：清空某个 session 的向量记录。
- `prune_session`：裁剪某个 session 的向量记录。
- `upsert_record`：写入一条通用向量记录。
- `search_records`：检索通用向量记录。
- `list_recent_records`：读取最近通用向量记录。
- `list_all_records`：读取满足条件的全部记录。
- `clear_records`：清空满足过滤条件的记录。
- `_initialize_backend`：选择真实 Qdrant 或 JSON fallback。
- `_can_use_qdrant`：检查是否可用真实 Qdrant。
- `_ensure_qdrant_collection`：创建或检查 collection。
- `_qdrant_upsert`：写入真实 Qdrant。
- `_qdrant_search`：检索真实 Qdrant。
- `_qdrant_list_recent`：读取最近 Qdrant 记录。
- `_normalize_qdrant_point_id`：规范化 point id。
- `_resolve_vector_size`：解析向量维度。
- `_extract_collection_vector_size`：读取已有 collection 维度。
- `_qdrant_clear`：清空真实 Qdrant 记录。
- `_qdrant_prune_session`：执行 session 级裁剪。
- `_qdrant_scroll_all`：滚动读取全部命中记录。
- `_build_qdrant_filter`：构造 Qdrant 过滤器。
- `_ensure_json_store`：确保 JSON fallback 文件存在。
- `_load_json_records`：读取 JSON fallback 记录。
- `_save_json_records`：写回 JSON fallback。
- `_normalize_record`：兼容旧记录结构。
- `_match_filters`：判断 payload 是否匹配过滤条件。

### `memory/storage/neo4j_store.py`
- 作用：Neo4j / JSON fallback 图谱存储适配层。

#### `Neo4jGraphStore`
- `__init__`：初始化图存储配置、fallback 路径和后端类型。
- `upsert_memory`：把记忆及其实体关系写入图存储。
- `search_related`：按查询中的实体做关联召回。
- `clear_session`：清空某个 session 的图谱数据。
- `list_session_items`：读取某个 session 的全部图谱记忆。
- `list_recent`：读取最近图谱记忆。
- `prune_session`：裁剪某个 session 的图谱数据。
- `extract_entities`：从文本中抽实体。
- `extract_relations`：从文本中抽关系。
- `_extract_targets`：抽关系目标。
- `_normalize_entity`：归一化实体名称。
- `_normalize_relation`：归一化关系名称。
- `_dedupe_texts`：对文本去重。
- `_initialize_backend`：选择真实 Neo4j 或 JSON fallback。
- `_can_use_neo4j`：检查是否可用真实 Neo4j。
- `_neo4j_upsert_memory`：写入真实 Neo4j。
- `_neo4j_search_related`：从真实 Neo4j 中做关联检索。
- `_neo4j_clear_session`：清空真实 Neo4j 中某个 session 的图数据。
- `_neo4j_list_recent`：读取最近图谱记忆。
- `_neo4j_list_all`：读取全部图谱记忆。
- `_neo4j_prune_session`：执行 session 级裁剪。
- `_json_upsert_memory`：写入本地 JSON fallback。
- `_json_search_related`：从本地 JSON fallback 中检索。
- `_ensure_json_store`：确保 JSON fallback 文件存在。
- `_load_json_payload`：读取图谱 fallback 内容。
- `_save_json_payload`：写回图谱 fallback 内容。

## 8. `memory/rag/` 目录

### `memory/rag/__init__.py`
- 作用：统一导出 RAG 子模块对象。

### `memory/rag/document.py`
- 作用：定义文档切片数据结构和基础文档处理逻辑。

#### `DocumentChunk`
- 作用：表示一段切片后的文档。

#### `RetrievedChunk`
- 作用：表示一条检索命中及其分数。

#### `DocumentProcessor`
- 作用：负责读取文本并切片。
- `__init__`：初始化切片大小和重叠长度。
- `load`：从文件读取并切片。
- `load_text`：对传入文本直接切片。
- `split_text`：执行切片逻辑。

### `memory/rag/pipeline.py`
- 作用：实现本地最小可用 RAG 管道。

#### `RagPipeline`
- `__init__`：初始化处理器、embedding 服务和向量存储。
- `add_document`：读取文档、切片并写入索引。
- `search`：做向量召回和轻量重排。
- `answer`：生成“参考结论 + 证据片段”式回答上下文。
- `clear`：清空当前 RAG 索引。
- `list_sources`：列出已入库文档来源。
- `run`：兼容旧接口。
- `_search_inline_documents`：对内联文档做检索。
- `build_answer_context`：把检索结果组织成结构化上下文。
- `_format_matches`：格式化命中文档。
- `_rerank_score`：计算重排分数。
- `_extract_query_tokens`：抽取查询关键词。
- `_summarize_chunk`：生成片段摘要。
- `_build_chunk_record_id`：构造稳定的切片 id。

## 9. 当前关键设计点

### 写入闭环
- 不是所有消息都会进长期记忆
- 会先经过：内容分类、价值评估、低价值过滤、重复过滤、语义写入判断

### 检索闭环
- working、episodic、semantic 可以共同参与召回
- 召回结果会补来源解释和原因说明

### 长期保留
- `episodic` 和 `semantic` 都有独立上限
- retention 不是简单保留最近 N 条，而是优先保留偏好、事实、工具成功结果和最终答案
- retention 日志可以直接用于调试
