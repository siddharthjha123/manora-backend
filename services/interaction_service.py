"""
MANORA Interaction Service.
Main orchestrator executing the full conversational lifecycle:
Interaction -> Conversational Retrieval -> Emotion Agent -> Candidate Memories ->
Two-Stage Memory Pipeline -> Buddy State -> Buddy Agent -> Response Persistence.
"""

import logging
import uuid
from collections import Counter
from typing import Any, Dict

from buddy.buddy_engine import BuddyAgent, buddy_agent
from database.connection import DatabaseManager, db
from emotion_agent.emotion_engine import EmotionAgent, emotion_agent
from memory.memory_engine import MemoryEngine, memory_engine
from ml.emotion_classifier import EmotionClassifier, emotion_classifier
from state.state_engine import BuddyState, StateEngine, state_engine
from observability.metrics import INTERACTIONS_TOTAL, BUDDY_STATE_UPDATES_TOTAL
from observability.langfuse import create_trace

logger = logging.getLogger("manora.interaction.service")


class InteractionService:
    """Orchestrates end-to-end processing of student interactions."""

    def __init__(
        self,
        database: DatabaseManager = db,
        classifier: EmotionClassifier = emotion_classifier,
        memory: MemoryEngine = memory_engine,
        emotion: EmotionAgent = emotion_agent,
        state: StateEngine = state_engine,
        buddy: BuddyAgent = buddy_agent,
    ):
        self.db = database
        self.classifier = classifier
        self.memory = memory
        self.emotion = emotion
        self.state = state
        self.buddy = buddy

    async def process_interaction(
        self,
        user_id: str,
        session_id: str,
        text: str,
    ) -> Dict[str, Any]:
        """Orchestrate one complete user-to-MANORA-to-Buddy interaction.

        Args:
            user_id: Student identifier owning the interaction and memories.
            session_id: Conversation session containing the interaction.
            text: Current user message.

        Major stages:
            1. Persist the incoming interaction.
            2. Retrieve recent conversational context and goals.
            3. Retrieve relevant conversational long-term memories when needed.
            4. Analyze emotion and extract temporary candidate memories.
            5. Run the two-stage long-term-memory pipeline for those candidates.
            6. Update and persist Buddy state.
            7. Generate Buddy's response.
            8. Persist Buddy's response and return the existing API payload.

        Conversational retrieval and consolidation retrieval are deliberately
        separate. The former supplies context to Emotion/Buddy; the latter is
        performed independently inside MemoryEngine to decide CREATE, UPDATE,
        or REJECT for each Stage 1 consolidated memory.
        """
        user_id = str(user_id)
        session_id = str(session_id)
        interaction_id = str(uuid.uuid4())

        logger.info("Processing interaction %s for user %s", interaction_id, user_id)

        # Create Langfuse Trace for student interaction lifecycle
        trace = create_trace(
            name="student_interaction",
            user_id=user_id,
            session_id=session_id,
            tags=["interaction", "conversation"],
            metadata={"interaction_id": interaction_id, "text_length": len(text)},
        )

        try:
            # Stage 1 — Interaction persistence
            # Store the incoming message first so it receives a stable interaction ID
            # and immediately becomes part of the conversation history.
            logger.info("[1/8] Saving user interaction")
            user_interaction = await self.db.save_interaction(
                user_id=user_id,
                session_id=session_id,
                role="user",
                raw_text=text,
                interaction_id=interaction_id,
            )

            # Stage 2 — Conversation context
            # Recent interactions and goals support conversational understanding. This
            # context is separate from retrieval used for memory consolidation.
            logger.info("[2/8] Loading conversation context and goals")
            recent_context = await self.db.get_recent_interactions(
                user_id=user_id,
                session_id=session_id,
                limit=5,
            )
            goals = await self.db.get_user_goals(user_id=user_id)

            # Stage 3 — Conversational memory retrieval
            # Retrieve existing long-term memories only when they help understand or
            # answer this message. These results go to Emotion/Buddy, not Stage 2 Qwen.
            logger.info("[3/8] Retrieving conversational memories when needed")
            should_retrieve = self.memory.should_retrieve(text=text, user_goals=goals)
            if should_retrieve:
                context = await self.memory.retrieve_context(user_id=user_id, text=text)
            else:
                context = self.memory.empty_context()

            # Stage 4 — Emotion analysis
            # The ML classifier and Emotion Agent analyze this interaction. The
            # structured result is also the source used to extract candidate memories.
            logger.info("[4/8] Running Emotion Agent")
            ml_probabilities = self.classifier.predict(text)
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

            await self.db.save_interaction_analysis(
                interaction_id=interaction_id,
                analysis_dict=emotion_analysis.model_dump(),
            )

            # Stage 5 — Long-term memory processing
            # Candidate memories are temporary evidence. MemoryEngine passes them
            # through Stage 1 consolidation, independent Qdrant retrieval, Stage 2
            # CREATE/UPDATE/REJECT reasoning, and final PostgreSQL/Qdrant/Neo4j writes.
            logger.info("[5/8] Processing candidate memories through Data Agent")
            candidate_memories = self.memory.extract_candidate_memories(
                interaction=user_interaction,
                emotion_analysis=emotion_analysis,
            )
            logger.info("Candidate memories: %d", len(candidate_memories))
            if candidate_memories:
                memory_result = await self.memory.process_long_term_memories(
                    candidate_memories
                )
                decision_counts = Counter(
                    decision.action.value for decision in memory_result.decisions
                )
                logger.info(
                    "Consolidated memories: %d; memory decisions: CREATE=%d "
                    "UPDATE=%d REJECT=%d",
                    len(memory_result.stage1.consolidated_memories),
                    decision_counts["CREATE"],
                    decision_counts["UPDATE"],
                    decision_counts["REJECT"],
                )
            else:
                logger.info("No candidate memories; skipping long-term memory pipeline")

            # Stage 6 — Buddy State
            # Update Buddy only after memory processing so the current emotional turn
            # influences its internal state in the established pipeline order.
            logger.info("[6/8] Updating Buddy state")
            raw_state = await self.db.get_buddy_state(user_id=user_id)
            if raw_state:
                current_state = BuddyState(**raw_state)
            else:
                current_state = self.state.initialize_state()

            new_state = self.state.update_state(
                current_state=current_state,
                emotion_analysis=emotion_analysis,
            )

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
            BUDDY_STATE_UPDATES_TOTAL.labels(status="success").inc()

            # Stage 7 — Buddy response
            # Buddy responds using the current message, emotion, updated state, recent
            # conversation, goals, and conversationally retrieved memories only.
            logger.info("[7/8] Generating Buddy response")
            buddy_response = await self.buddy.generate(
                user_text=text,
                emotion_analysis=emotion_analysis,
                buddy_state=new_state,
                recent_context=recent_context,
                memories=context.get("memories"),
                goals=goals,
            )

            # Stage 8 — Response persistence
            # Save Buddy's message as its own interaction before returning the existing
            # public response contract.
            logger.info("[8/8] Persisting Buddy response")
            await self.db.save_interaction(
                user_id=user_id,
                session_id=session_id,
                role="buddy",
                raw_text=buddy_response.text,
            )

            # Record interaction success metric
            INTERACTIONS_TOTAL.labels(status="success").inc()
            trace.update(
                output={
                    "interaction_id": interaction_id,
                    "primary_emotion": emotion_analysis.primary_emotion,
                    "expression": buddy_response.expression,
                }
            )

            return {
                "interaction_id": interaction_id,
                "emotion": emotion_analysis.model_dump(),
                "buddy_state": new_state.to_dict(),
                "buddy": buddy_response.model_dump(),
            }

        except Exception as exc:
            INTERACTIONS_TOTAL.labels(status="error").inc()
            trace.update(metadata={"error": str(exc)})
            raise


# Global singleton instance
interaction_service = InteractionService()
