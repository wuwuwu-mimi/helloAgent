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

    _ENTITY_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+|[\u4e00-\u9fff]{2,12}")

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

    @classmethod
    def extract_entities(cls, text: str) -> List[str]:
        """从文本中抽取最小可用的实体候选。"""
        candidates: List[str] = []
        for token in cls._ENTITY_PATTERN.findall(text):
            normalized = token.strip("，。！？,.!?：:；;()[]{}<>\"'")
            if len(normalized) < 2:
                continue
            if normalized in {"当前", "然后", "用户问题", "回答问题", "直接回答"}:
                continue
            candidates.append(normalized)

        deduped: List[str] = []
        seen: set[str] = set()
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped[:12]

    @staticmethod
    def extract_relations(text: str, entities: List[str]) -> List[Dict[str, str]]:
        """从文本中抽取最小关系结构。"""
        relations: List[Dict[str, str]] = []
        lowered = text.strip()

        preference_match = re.search(r"(我|用户)[^。！？\n]{0,12}(喜欢|偏好)([^。！？\n]{1,24})", lowered)
        if preference_match:
            subject = preference_match.group(1)
            relation = "喜欢"
            target = preference_match.group(3).strip("：:，,。. ")
            if target:
                relations.append(
                    {
                        "source": subject,
                        "relation": relation,
                        "target": target,
                    }
                )

        support_match = re.search(r"([A-Za-z][A-Za-z0-9_\-]+|[\u4e00-\u9fff]{2,12})[^。！？\n]{0,8}(支持|包含)([^。！？\n]+)", lowered)
        if support_match:
            source = support_match.group(1)
            relation = support_match.group(2)
            tail = support_match.group(3)
            targets = [item.strip("，,、。 ") for item in re.split(r"[，,、]", tail) if item.strip()]
            for target in targets[:6]:
                relations.append(
                    {
                        "source": source,
                        "relation": relation,
                        "target": target,
                    }
                )

        if not relations and len(entities) >= 2:
            anchor = entities[0]
            for target in entities[1:4]:
                relations.append(
                    {
                        "source": anchor,
                        "relation": "related_to",
                        "target": target,
                    }
                )
        return relations

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
                RETURN m.payload AS payload, count(DISTINCT e) AS score
                ORDER BY score DESC, m.created_at DESC
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
