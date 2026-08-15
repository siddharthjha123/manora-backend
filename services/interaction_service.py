"""
MANORA Interaction Service.
Main orchestrator executing the full conversational lifecycle:
Interaction -> Memory Retrieval Decision -> Emotion Agent -> Mock Data Agent ->
Memory Persistence -> Buddy State Engine -> Buddy Agent -> Persistence -> Response.
"""

import logging
import uuid
from typing import Any, Dict, Optional

from buddy.buddy_engine import BuddyAgent, buddy_agent
from data_agent.mock_data_agent import MockDataAgent, mock_data_agent
from database.connection import DatabaseManager, db
from emotion_agent.emotion_engine import EmotionAgent, emotion_agent
from memory.memory_engine import MemoryEngine, memory_engine
from ml.emotion_classifier import EmotionClassifier, emotion_classifier
from state.state_engine import BuddyState, StateEngine, state_engine

logger = logging.getLogger("manora.interaction.service")


class InteractionService:
    """Orchestrates end-to-end processing of student interactions."""

    def __init__(
        self,
        database: DatabaseManager = db,
        classifier: EmotionClassifier = emotion_classifier,
        memory: MemoryEngine = memory_engine,
        emotion: EmotionAgent = emotion_agent,
        data_mock: MockDataAgent = mock_data_agent,
        state: StateEngine = state_engine,
        buddy: BuddyAgent = buddy_agent,
    ):
        self.db = database
        self.classifier = classifier
        self.memory = memory
        self.emotion = emotion
        self.data_mock = data_mock
        self.state = state
        self.buddy = buddy

    async def process_interaction(
        self,
        user_id: str,
        session_id: str,
        text: str,
    ) -> Dict[str, Any]:
        """
        Executes the 12-step interaction orchestration pipeline.
        """
        user_id = str(user_id)
        session_id = str(session_id)
        interaction_id = str(uuid.uuid4())

        logger.info(f"Processing interaction {interaction_id} for user {user_id}")

        # 1. Store incoming user interaction
        user_interaction = await self.db.save_interaction(
            user_id=user_id,
            session_id=session_id,
            role="user",
            raw_text=text,
            interaction_id=interaction_id,
        )

        # 2. Fetch recent conversation context & active goals
        recent_context = await self.db.get_recent_interactions(
            user_id=user_id,
            session_id=session_id,
            limit=5,
        )
        goals = await self.db.get_user_goals(user_id=user_id)

        # 3. Determine if memory retrieval is needed
        should_retrieve = self.memory.should_retrieve(text=text, user_goals=goals)

        # 4. Retrieve context if needed
        if should_retrieve:
            context = await self.memory.retrieve_context(user_id=user_id, text=text)
        else:
            context = self.memory.empty_context()

        # 5. Run ML Emotion Classifier
        ml_probabilities = self.classifier.predict(text)

        # 6. Run Emotion Agent for structured analysis & reasoning
        emotion_analysis = await self.emotion.analyze(
            interaction_id=interaction_id,
            user_id=user_id,
            session_id=session_id,
            text=text,
            ml_probabilities=ml_probabilities,
            recent_context=recent_context,
            memories=context.get("memories"),
            goals=goals,
        )

        # 7. Store Emotion Analysis JSON
        await self.db.save_interaction_analysis(
            interaction_id=interaction_id,
            analysis_dict=emotion_analysis.model_dump(),
        )

        # 8. Run Mock Data Agent to extract candidate memories
        candidate_memories = self.data_mock.process(
            interaction=user_interaction,
            emotion=emotion_analysis,
        )

        # 9. Filter and persist candidate memories (PostgreSQL, Qdrant, Neo4j)
        await self.memory.persist_candidates(
            user_id=user_id,
            interaction_id=interaction_id,
            candidate_memories=candidate_memories,
        )

        # 10. Fetch current Buddy state & update deterministically
        raw_state = await self.db.get_buddy_state(user_id=user_id)
        if raw_state:
            current_state = BuddyState(**raw_state)
        else:
            current_state = self.state.initialize_state()

        new_state = self.state.update_state(
            current_state=current_state,
            emotion_analysis=emotion_analysis,
        )

        # 11. Persist new Buddy State & record history
        await self.db.upsert_buddy_state(
            user_id=user_id,
            state_dict=new_state.to_dict(),
        )
        await self.db.save_buddy_state_history(
            user_id=user_id,
            previous_state=current_state.to_dict(),
            new_state=new_state.to_dict(),
            trigger_interaction_id=interaction_id,
        )

        # 12. Run Buddy Agent to generate natural response
        buddy_response = await self.buddy.generate(
            user_text=text,
            emotion_analysis=emotion_analysis,
            buddy_state=new_state,
            recent_context=recent_context,
            memories=context.get("memories"),
            goals=goals,
        )

        # 13. Store Buddy's response interaction
        await self.db.save_interaction(
            user_id=user_id,
            session_id=session_id,
            role="buddy",
            raw_text=buddy_response.text,
        )

        # 14. Return structured response payload
        return {
            "interaction_id": interaction_id,
            "emotion": emotion_analysis.model_dump(),
            "buddy_state": new_state.to_dict(),
            "buddy": buddy_response.model_dump(),
        }


# Global singleton instance
interaction_service = InteractionService()
