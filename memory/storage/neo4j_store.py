from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory.base import MemoryConfig, MemoryItem

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - 当前环境可能未安装 neo4j
    GraphDatabase = None


class Neo4jGraphStore:
    """
    一个支持“真实 Neo4j / 本地 JSON fallback”的图存储骨架。

    修改说明：当前先把图谱接口和本地回退路径打通，
    这样语义记忆可以同时拥有“向量召回 + 关系召回”的双通道；
    后续只要本地装好 `neo4j` 驱动并配置连接信息，就能直接切到真实图数据库。
    """

    _ENTITY_PATTERN = re.compile(r"`[^`]+`|[A-Za-z][A-Za-z0-9_\-+.]{1,31}|[\u4e00-\u9fff]{2,16}")
    _SPLIT_PATTERN = re.compile(r"[，,、/]|(?:\s+and\s+)|(?:\s+or\s+)|和|以及|并且|还有|及")
    _STOPWORDS = {
        "当前",
        "然后",
        "用户问题",
        "回答问题",
        "直接回答",
        "这个问题",
        "这个答案",
        "一下",
        "什么",
        "哪些",
        "一下子",
        "时候",
        "进行",
        "需要",
        "可以",
        "已经",
        "目前",
        "当前支持",
        "支持",
        "包含",
        "具备",
        "提供",
        "采用",
        "三种",
        "范式",
        "饮品",
    }
    _RELATION_VERBS = ("喜欢", "不喜欢", "偏好", "爱喝", "爱吃", "讨厌", "避免", "支持", "包含", "具备", "提供", "采用", "记住", "记录", "是")
    _CANONICAL_ENTITY_MAP = {
        "我": "用户",
        "我的": "用户",
        "我喜欢": "用户",
        "用户偏好": "用户",
    }

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.store_path = Path(config.graph_store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = "json"
        self.driver: Any = None
        self._initialize_backend()

    def upsert_memory(self, item: MemoryItem, entities: List[str], relations: List[Dict[str, str]]) -> None:
        """把一条记忆同步写入图存储。"""
        if self.backend == "neo4j":
            self._neo4j_upsert_memory(item, entities, relations)
            return
        self._json_upsert_memory(item, entities, relations)

    def search_related(
        self,
        session_id: str,
        query: str,
        *,
        limit: int = 5,
    ) -> List[MemoryItem]:
        """根据查询中的实体关键词查找关联记忆。"""
        query_entities = self.extract_entities(query)
        if not query_entities:
            return []
        if self.backend == "neo4j":
            return self._neo4j_search_related(session_id, query_entities, limit=limit)
        return self._json_search_related(session_id, query_entities, limit=limit)

    def clear_session(self, session_id: str) -> None:
        """清空某个 session 的图谱记忆。"""
        if self.backend == "neo4j":
            self._neo4j_clear_session(session_id)
            return
        payload = self._load_json_payload()
        payload["facts"] = [
            fact for fact in payload["facts"] if fact.get("session_id") != session_id
        ]
        payload["nodes"] = [
            node for node in payload["nodes"] if node.get("session_id") != session_id
        ]
        payload["edges"] = [
            edge for edge in payload["edges"] if edge.get("session_id") != session_id
        ]
        self._save_json_payload(payload)

    def list_session_items(self, session_id: str) -> List[MemoryItem]:
        """读取某个 session 的全部图谱记忆，供长期保留策略裁剪。"""
        if self.backend == "neo4j":
            return self._neo4j_list_all(session_id)
        payload = self._load_json_payload()
        facts = [
            MemoryItem.model_validate(fact["memory_item"])
            for fact in payload["facts"]
            if fact.get("session_id") == session_id
        ]
        facts.sort(key=lambda item: item.created_at)
        return facts

    def list_recent(self, session_id: str, limit: int = 10) -> List[MemoryItem]:
        """返回最近写入图谱的若干记忆。"""
        if self.backend == "neo4j":
            return self._neo4j_list_recent(session_id, limit=limit)
        payload = self._load_json_payload()
        facts = [
            MemoryItem.model_validate(fact["memory_item"])
            for fact in payload["facts"]
            if fact.get("session_id") == session_id
        ]
        facts.sort(key=lambda item: item.created_at, reverse=True)
        return facts[:limit]

    def prune_session(self, session_id: str, keep_ids: List[str]) -> int:
        """只保留指定 id 的图谱记忆，返回本次裁剪掉的条数。"""
        keep_id_set = set(keep_ids)
        if self.backend == "neo4j":
            return self._neo4j_prune_session(session_id=session_id, keep_ids=keep_id_set)

        payload = self._load_json_payload()
        facts = payload["facts"]
        kept_facts = [
            fact
            for fact in facts
            if fact.get("session_id") != session_id or fact.get("id") in keep_id_set
        ]
        pruned_count = len(facts) - len(kept_facts)
        if pruned_count <= 0:
            return 0

        keep_fact_ids = {fact["id"] for fact in kept_facts if fact.get("session_id") == session_id}
        payload["facts"] = kept_facts
        payload["edges"] = [
            edge
            for edge in payload["edges"]
            if edge.get("session_id") != session_id or edge.get("item_id") in keep_fact_ids
        ]
        referenced_entities = {
            edge.get("source")
            for edge in payload["edges"]
            if edge.get("session_id") == session_id
        } | {
            edge.get("target")
            for edge in payload["edges"]
            if edge.get("session_id") == session_id
        }
        referenced_entities |= {
            entity
            for fact in payload["facts"]
            if fact.get("session_id") == session_id
            for entity in fact.get("entities", [])
        }
        payload["nodes"] = [
            node
            for node in payload["nodes"]
            if node.get("session_id") != session_id or node.get("name") in referenced_entities
        ]
        self._save_json_payload(payload)
        return pruned_count

    @classmethod
    def extract_entities(cls, text: str) -> List[str]:
        """从文本中抽取更稳一些的实体候选。"""
        candidates: List[str] = []
        normalized_text = text.replace("（", "(").replace("）", ")")

        if "我" in normalized_text or "用户" in normalized_text:
            candidates.append("用户")

        for token in cls._ENTITY_PATTERN.findall(normalized_text):
            entity = cls._normalize_entity(token)
            if not entity:
                continue
            candidates.append(entity)

        # 修改说明：除了直接扫 token，再补一层“关系主语/宾语候选”切分，
        # 让 “ReAct、Plan-and-Solve、Reflection” 这种枚举结构更容易完整保留下来。
        for fragment in cls._SPLIT_PATTERN.split(normalized_text):
            entity = cls._normalize_entity(fragment)
            if not entity:
                continue
            candidates.append(entity)

        return cls._dedupe_texts(candidates, limit=16)

    @classmethod
    def extract_relations(cls, text: str, entities: List[str]) -> List[Dict[str, str]]:
        """从文本中抽取更丰富的最小关系结构。"""
        relations: List[Dict[str, str]] = []
        normalized = " ".join(text.strip().split())

        relation_patterns = [
            (r"(我|用户)(?:[^。！？\n，,]{0,8})?(喜欢|偏好|爱喝|爱吃)\s*([^，,。！？\n]{1,40})", "喜欢"),
            (r"(我|用户)(?:[^。！？\n，,]{0,8})?(不喜欢|讨厌|避免)\s*([^，,。！？\n]{1,40})", "不喜欢"),
            (r"([A-Za-z][A-Za-z0-9_\-+.]{1,31}|[\u4e00-\u9fff]{2,16})[^。！？\n]{0,10}(支持|包含|具备|提供|采用)([^。！？\n]{1,80})", None),
            (r"([A-Za-z][A-Za-z0-9_\-+.]{1,31}|[\u4e00-\u9fff]{2,16})[^。！？\n]{0,10}(记住|记录)([^。！？\n]{1,40})", None),
            (r"([A-Za-z][A-Za-z0-9_\-+.]{1,31}|[\u4e00-\u9fff]{2,16})\s*是\s*([^。！？\n]{1,40})", "是"),
        ]

        for pattern, forced_relation in relation_patterns:
            for match in re.finditer(pattern, normalized):
                source = cls._normalize_entity(match.group(1))
                relation = forced_relation or cls._normalize_relation(match.group(2))
                tail = match.group(3) if match.lastindex and match.lastindex >= 3 else ""
                targets = cls._extract_targets(tail, entities)
                for target in targets:
                    if not source or not target or source == target:
                        continue
                    relations.append(
                        {
                            "source": source,
                            "relation": relation,
                            "target": target,
                        }
                    )

        if "我" in normalized or "用户" in normalized:
            for match in re.finditer(r"(?:^|[，,；;])\s*(不喜欢|讨厌|避免)\s*([^，,。！？\n]{1,40})", normalized):
                targets = cls._extract_targets(match.group(2), entities)
                for target in targets:
                    relations.append(
                        {
                            "source": "用户",
                            "relation": "不喜欢",
                            "target": target,
                        }
                    )

        if not relations and len(entities) >= 2:
            anchor = entities[0]
            for target in entities[1:5]:
                relations.append(
                    {
                        "source": anchor,
                        "relation": "related_to",
                        "target": target,
                    }
                )

        deduped: List[Dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for relation in relations:
            key = (relation["source"], relation["relation"], relation["target"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(relation)
        return deduped[:12]

    @classmethod
    def _extract_targets(cls, tail: str, entities: List[str]) -> List[str]:
        targets: List[str] = []
        normalized_tail = tail.strip("：:，,。.；;!?！？ ")
        if not normalized_tail:
            return []

        for fragment in cls._SPLIT_PATTERN.split(normalized_tail):
            entity = cls._normalize_entity(fragment)
            if entity:
                targets.append(entity)

        if not targets:
            for entity in entities:
                if entity and entity in normalized_tail:
                    targets.append(entity)
        return cls._dedupe_texts(targets, limit=8)

    @classmethod
    def _normalize_entity(cls, text: str) -> str:
        normalized = text.strip().strip("`").strip("，。！？,.!?：:；;()[]{}<>\"'")
        normalized = re.sub(r"\s+", " ", normalized)
        if not normalized:
            return ""
        normalized = re.sub(r"^([A-Za-z][A-Za-z0-9_\-+.]{1,31})\s+[一二三四五六七八九十0-9]+种.*$", r"\1", normalized)
        normalized = cls._CANONICAL_ENTITY_MAP.get(normalized, normalized)
        if len(normalized) < 2 and normalized != "用户":
            return ""
        if normalized.lower() in {"thought", "action", "finish", "observation"}:
            return ""
        if any(verb in normalized for verb in cls._RELATION_VERBS) and len(normalized) > 2:
            return ""
        if normalized in cls._STOPWORDS:
            return ""
        if normalized.startswith("请") and len(normalized) > 2:
            normalized = normalized[1:]
        if normalized in cls._STOPWORDS:
            return ""
        normalized = normalized.strip("的")
        return normalized

    @staticmethod
    def _normalize_relation(text: str) -> str:
        relation = text.strip("：:，,。.；;!?！？ ")
        relation = relation.replace("爱喝", "喜欢").replace("爱吃", "喜欢")
        return relation or "related_to"

    @staticmethod
    def _dedupe_texts(items: List[str], *, limit: int) -> List[str]:
        deduped: List[str] = []
        seen: set[str] = set()
        for item in items:
            if not item or item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped[:limit]

    def _initialize_backend(self) -> None:
        requested_backend = (self.config.graph_store_backend or "auto").strip().lower()
        if requested_backend in {"json", "file"}:
            self.backend = "json"
            self._ensure_json_store()
            return

        if self._can_use_neo4j():
            self.backend = "neo4j"
            return

        self.backend = "json"
        self._ensure_json_store()

    def _can_use_neo4j(self) -> bool:
        if GraphDatabase is None:
            return False
        if not (self.config.neo4j_url or "").strip():
            return False
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_url,
                auth=(
                    (self.config.neo4j_user or "").strip(),
                    (self.config.neo4j_password or "").strip(),
                ),
            )
            self.driver.verify_connectivity()
            return True
        except Exception:
            self.driver = None
            return False

    def _neo4j_upsert_memory(
        self,
        item: MemoryItem,
        entities: List[str],
        relations: List[Dict[str, str]],
    ) -> None:
        if self.driver is None:
            return
        with self.driver.session(database=self.config.neo4j_database) as session:
            session.run(
                """
                MERGE (m:MemoryFact {id: $id})
                SET m.session_id = $session_id,
                    m.role = $role,
                    m.content = $content,
                    m.created_at = $created_at,
                    m.payload = $payload
                """,
                id=item.id,
                session_id=item.session_id,
                role=item.role,
                content=item.content,
                created_at=item.created_at.isoformat(),
                payload=json.dumps(item.model_dump(mode="json"), ensure_ascii=False),
            )
            for entity in entities:
                session.run(
                    """
                    MERGE (e:Entity {session_id: $session_id, name: $name})
                    ON CREATE SET e.created_at = $created_at
                    MERGE (m:MemoryFact {id: $item_id})
                    MERGE (m)-[:MENTIONS]->(e)
                    """,
                    session_id=item.session_id,
                    name=entity,
                    created_at=item.created_at.isoformat(),
                    item_id=item.id,
                )
            for relation in relations:
                session.run(
                    """
                    MERGE (source:Entity {session_id: $session_id, name: $source})
                    MERGE (target:Entity {session_id: $session_id, name: $target})
                    MERGE (source)-[r:RELATED {session_id: $session_id, relation: $relation, item_id: $item_id}]->(target)
                    SET r.created_at = $created_at,
                        r.content = $content
                    """,
                    session_id=item.session_id,
                    source=relation["source"],
                    target=relation["target"],
                    relation=relation["relation"],
                    item_id=item.id,
                    created_at=item.created_at.isoformat(),
                    content=item.content,
                )

    def _neo4j_search_related(
        self,
        session_id: str,
        query_entities: List[str],
        *,
        limit: int,
    ) -> List[MemoryItem]:
        if self.driver is None:
            return []
        with self.driver.session(database=self.config.neo4j_database) as session:
            records = session.run(
                """
                MATCH (m:MemoryFact {session_id: $session_id})-[:MENTIONS]->(e:Entity {session_id: $session_id})
                WHERE e.name IN $entities
                RETURN m.payload AS payload, m.created_at AS created_at, count(DISTINCT e) AS score
                ORDER BY score DESC, created_at DESC
                LIMIT $limit
                """,
                session_id=session_id,
                entities=query_entities,
                limit=limit,
            )
            items: List[MemoryItem] = []
            for record in records:
                payload = json.loads(record["payload"])
                item = MemoryItem.model_validate(payload)
                item.metadata = {**item.metadata, "graph_score": int(record["score"]), "graph_backend": self.backend}
                items.append(item)
            return items

    def _neo4j_clear_session(self, session_id: str) -> None:
        if self.driver is None:
            return
        with self.driver.session(database=self.config.neo4j_database) as session:
            session.run(
                """
                MATCH (m:MemoryFact {session_id: $session_id})
                DETACH DELETE m
                """,
                session_id=session_id,
            )
            session.run(
                """
                MATCH (e:Entity {session_id: $session_id})
                DETACH DELETE e
                """,
                session_id=session_id,
            )

    def _neo4j_list_recent(self, session_id: str, *, limit: int) -> List[MemoryItem]:
        if self.driver is None:
            return []
        with self.driver.session(database=self.config.neo4j_database) as session:
            records = session.run(
                """
                MATCH (m:MemoryFact {session_id: $session_id})
                RETURN m.payload AS payload
                ORDER BY m.created_at DESC
                LIMIT $limit
                """,
                session_id=session_id,
                limit=limit,
            )
            return [MemoryItem.model_validate(json.loads(record["payload"])) for record in records]

    def _neo4j_list_all(self, session_id: str) -> List[MemoryItem]:
        if self.driver is None:
            return []
        with self.driver.session(database=self.config.neo4j_database) as session:
            records = session.run(
                """
                MATCH (m:MemoryFact {session_id: $session_id})
                RETURN m.payload AS payload
                ORDER BY m.created_at ASC
                """,
                session_id=session_id,
            )
            return [MemoryItem.model_validate(json.loads(record["payload"])) for record in records]

    def _neo4j_prune_session(self, *, session_id: str, keep_ids: set[str]) -> int:
        if self.driver is None:
            return 0
        with self.driver.session(database=self.config.neo4j_database) as session:
            records = session.run(
                """
                MATCH (m:MemoryFact {session_id: $session_id})
                RETURN m.id AS id
                """,
                session_id=session_id,
            )
            existing_ids = [str(record["id"]) for record in records]
            drop_ids = [item_id for item_id in existing_ids if item_id not in keep_ids]
            if not drop_ids:
                return 0
            session.run(
                """
                MATCH (m:MemoryFact {session_id: $session_id})
                WHERE m.id IN $drop_ids
                DETACH DELETE m
                """,
                session_id=session_id,
                drop_ids=drop_ids,
            )
            session.run(
                """
                MATCH (e:Entity {session_id: $session_id})
                WHERE NOT (e)<-[:MENTIONS]-(:MemoryFact {session_id: $session_id})
                  AND NOT (e)-[:RELATED {session_id: $session_id}]-()
                DETACH DELETE e
                """,
                session_id=session_id,
            )
            return len(drop_ids)

    def _json_upsert_memory(
        self,
        item: MemoryItem,
        entities: List[str],
        relations: List[Dict[str, str]],
    ) -> None:
        payload = self._load_json_payload()
        payload["facts"] = [fact for fact in payload["facts"] if fact["id"] != item.id]
        payload["facts"].append(
            {
                "id": item.id,
                "session_id": item.session_id,
                "memory_item": item.model_dump(mode="json"),
                "entities": entities,
                "relations": relations,
                "created_at": item.created_at.isoformat(),
            }
        )

        for entity in entities:
            node_id = f"{item.session_id}:{entity}"
            existing = next((node for node in payload["nodes"] if node["id"] == node_id), None)
            if existing is None:
                payload["nodes"].append(
                    {
                        "id": node_id,
                        "session_id": item.session_id,
                        "name": entity,
                        "created_at": item.created_at.isoformat(),
                    }
                )

        for relation in relations:
            edge_id = f"{item.id}:{relation['source']}:{relation['relation']}:{relation['target']}"
            payload["edges"] = [edge for edge in payload["edges"] if edge["id"] != edge_id]
            payload["edges"].append(
                {
                    "id": edge_id,
                    "item_id": item.id,
                    "session_id": item.session_id,
                    "source": relation["source"],
                    "relation": relation["relation"],
                    "target": relation["target"],
                    "created_at": item.created_at.isoformat(),
                }
            )

        self._save_json_payload(payload)

    def _json_search_related(
        self,
        session_id: str,
        query_entities: List[str],
        *,
        limit: int,
    ) -> List[MemoryItem]:
        payload = self._load_json_payload()
        query_set = set(query_entities)
        scored: List[tuple[int, MemoryItem]] = []
        for fact in payload["facts"]:
            if fact.get("session_id") != session_id:
                continue
            entities = set(fact.get("entities", []))
            score = len(query_set & entities)
            if score <= 0:
                continue
            item = MemoryItem.model_validate(fact["memory_item"])
            item.metadata = {
                **item.metadata,
                "graph_score": score,
                "graph_backend": self.backend,
            }
            scored.append((score, item))

        scored.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
        return [item for _, item in scored[:limit]]

    def _ensure_json_store(self) -> None:
        if not self.store_path.exists():
            self.store_path.write_text(
                json.dumps({"facts": [], "nodes": [], "edges": []}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _load_json_payload(self) -> Dict[str, List[Dict[str, Any]]]:
        raw_text = self.store_path.read_text(encoding="utf-8") if self.store_path.exists() else ""
        if not raw_text.strip():
            return {"facts": [], "nodes": [], "edges": []}
        payload = json.loads(raw_text)
        if isinstance(payload, list):
            # 修改说明：兼容未来可能误写成列表的旧格式，避免直接报错。
            return {"facts": payload, "nodes": [], "edges": []}
        return {
            "facts": payload.get("facts", []),
            "nodes": payload.get("nodes", []),
            "edges": payload.get("edges", []),
        }

    def _save_json_payload(self, payload: Dict[str, List[Dict[str, Any]]]) -> None:
        self.store_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
