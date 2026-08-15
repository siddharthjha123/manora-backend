"""
MANORA Neo4j Graph Database Adapter.
Represents relationships between Students, Goals, Memories, Emotions, Behaviors, and Decisions.
Supports graceful offline operation when Neo4j is disabled.
"""

import logging
from typing import Any, Dict, List, Optional

from config.settings import get_settings

logger = logging.getLogger("manora.graph_db")


class Neo4jAdapter:
    """Adapter for Neo4j Knowledge Graph storing mental health context and relationships."""

    def __init__(self):
        self.settings = get_settings()
        self.enabled = self.settings.NEO4J_ENABLED
        self._driver = None
        self._in_memory_graph: Dict[str, Dict[str, Any]] = {}

        if self.enabled:
            self._connect()

    def _connect(self):
        """Connects to Neo4j database."""
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.settings.NEO4J_URI,
                auth=(self.settings.NEO4J_USERNAME, self.settings.NEO4J_PASSWORD),
            )
            logger.info("Connected to Neo4j graph database successfully.")
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j ({e}). Falling back to in-memory graph representation.")
            self._driver = None

    def close(self):
        """Closes Neo4j driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def create_memory_relationships(
        self,
        user_id: str,
        memory_id: str,
        data: Dict[str, Any],
    ) -> bool:
        """
        Creates graph entities and relationships for a stored memory:
        Student -> HAS_MEMORY -> Memory
        Memory -> REFLECTS_EMOTION -> Emotion
        Memory -> SHOWED_BEHAVIOR -> Behavior
        Memory -> INVOLVED_DECISION -> Decision
        Student -> HAS_GOAL -> Goal
        """
        user_id = str(user_id)
        memory_id = str(memory_id)

        content = data.get("content", "")
        emotions = data.get("emotional_state", [])
        behavior = data.get("behavior", {})
        decision = data.get("decision", {})
        goal_rel = data.get("goal_relevance", {})

        if self._driver:
            cypher = """
            MERGE (s:Student {id: $user_id})
            MERGE (m:Memory {id: $memory_id})
            ON CREATE SET m.content = $content, m.created_at = datetime()
            MERGE (s)-[:HAS_MEMORY]->(m)

            WITH s, m
            UNWIND $emotions AS emo
            MERGE (e:Emotion {name: emo.emotion})
            MERGE (m)-[:REFLECTS_EMOTION {confidence: emo.confidence}]->(e)

            WITH s, m
            FOREACH (_ IN CASE WHEN $behavior_desc <> '' THEN [1] ELSE [] END |
                MERGE (b:Behavior {description: $behavior_desc, type: $behavior_type})
                MERGE (m)-[:SHOWED_BEHAVIOR]->(b)
                MERGE (s)-[:EXHIBITED]->(b)
            )

            WITH s, m
            FOREACH (_ IN CASE WHEN $decision_desc <> '' THEN [1] ELSE [] END |
                MERGE (d:Decision {description: $decision_desc})
                MERGE (m)-[:INVOLVED_DECISION]->(d)
            )

            WITH s, m
            FOREACH (_ IN CASE WHEN $goal_name <> '' THEN [1] ELSE [] END |
                MERGE (g:Goal {name: $goal_name})
                MERGE (s)-[:HAS_GOAL]->(g)
                MERGE (m)-[:RELATES_TO_GOAL]->(g)
            )
            """
            params = {
                "user_id": user_id,
                "memory_id": memory_id,
                "content": content,
                "emotions": emotions if isinstance(emotions, list) else [],
                "behavior_desc": behavior.get("description", "") if isinstance(behavior, dict) else "",
                "behavior_type": behavior.get("type", "") if isinstance(behavior, dict) else "",
                "decision_desc": decision.get("description", "") if isinstance(decision, dict) else "",
                "goal_name": goal_rel.get("goal", "") if (isinstance(goal_rel, dict) and goal_rel.get("related")) else "",
            }

            try:
                with self._driver.session() as session:
                    session.run(cypher, **params)
                logger.debug(f"Created Neo4j relationships for memory {memory_id}")
                return True
            except Exception as e:
                logger.error(f"Neo4j relationship creation error: {e}. Falling back to in-memory graph.")

        # In-memory graph fallback
        if user_id not in self._in_memory_graph:
            self._in_memory_graph[user_id] = {
                "memories": [],
                "behaviors": [],
                "decisions": [],
                "goals": [],
                "emotions": [],
            }

        user_graph = self._in_memory_graph[user_id]
        user_graph["memories"].append({"id": memory_id, "content": content})
        if isinstance(behavior, dict) and behavior.get("description"):
            user_graph["behaviors"].append(behavior)
        if isinstance(decision, dict) and decision.get("description"):
            user_graph["decisions"].append(decision)
        if isinstance(goal_rel, dict) and goal_rel.get("goal"):
            user_graph["goals"].append(goal_rel["goal"])
        if isinstance(emotions, list):
            user_graph["emotions"].extend(emotions)

        return True

    def get_relevant_graph_context(
        self,
        user_id: str,
        keywords: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Queries connected relationship graph for student context (recurring behaviors,
        decisions, related goals).
        """
        user_id = str(user_id)

        if self._driver:
            cypher = """
            MATCH (s:Student {id: $user_id})-[r:HAS_MEMORY]->(m:Memory)
            OPTIONAL MATCH (m)-[:SHOWED_BEHAVIOR]->(b:Behavior)
            OPTIONAL MATCH (m)-[:INVOLVED_DECISION]->(d:Decision)
            OPTIONAL MATCH (m)-[:RELATES_TO_GOAL]->(g:Goal)
            RETURN m.id AS memory_id,
                   m.content AS content,
                   b.description AS behavior,
                   d.description AS decision,
                   g.name AS goal
            LIMIT $limit
            """
            try:
                with self._driver.session() as session:
                    result = session.run(cypher, user_id=user_id, limit=limit)
                    records = [record.data() for record in result]
                    return records
            except Exception as e:
                logger.error(f"Neo4j query error: {e}. Falling back to in-memory graph.")

        # In-memory graph retrieval
        user_graph = self._in_memory_graph.get(user_id)
        if not user_graph:
            return []

        results = []
        for mem in user_graph.get("memories", [])[:limit]:
            results.append({
                "memory_id": mem["id"],
                "content": mem["content"],
                "behaviors": user_graph.get("behaviors", []),
                "decisions": user_graph.get("decisions", []),
                "goals": user_graph.get("goals", []),
            })
        return results

    def link_student_goal(self, user_id: str, goal_title: str) -> bool:
        """Links a student to a goal in the graph."""
        user_id = str(user_id)
        if self._driver:
            cypher = """
            MERGE (s:Student {id: $user_id})
            MERGE (g:Goal {name: $goal_title})
            MERGE (s)-[:HAS_GOAL]->(g)
            """
            try:
                with self._driver.session() as session:
                    session.run(cypher, user_id=user_id, goal_title=goal_title)
                return True
            except Exception as e:
                logger.error(f"Neo4j link_student_goal error: {e}")

        if user_id not in self._in_memory_graph:
            self._in_memory_graph[user_id] = {"memories": [], "behaviors": [], "decisions": [], "goals": [], "emotions": []}
        if goal_title not in self._in_memory_graph[user_id]["goals"]:
            self._in_memory_graph[user_id]["goals"].append(goal_title)
        return True


# Global singleton instance
neo4j_adapter = Neo4jAdapter()
