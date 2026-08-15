"""
MANORA Buddy Agent - Prompt Engineering.
Builds the contextualized prompt for Buddy response generation based on student signals,
Buddy internal state, and historical memory context.
"""

from typing import Any, Dict, List, Optional
from emotion_agent.emotion_schema import EmotionAnalysis
from state.state_engine import BuddyState


def build_buddy_prompt(
    user_text: str,
    emotion_analysis: EmotionAnalysis,
    buddy_state: BuddyState,
    recent_context: Optional[List[Dict[str, Any]]] = None,
    memories: Optional[List[Dict[str, Any]]] = None,
    goals: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """Constructs OpenAI-compatible messages for the Buddy Agent."""

    state_dict = buddy_state.to_dict()
    formatted_state = ", ".join(f"{k}: {v:.2f}" for k, v in state_dict.items())

    system_content = (
        "You are 'Buddy', an AI companion in MANORA, a Digital Mental Health and Psychological Support System "
        "for students in Higher Education.\n\n"
        "WHO YOU ARE:\n"
        "- You are an emotionally aware, natural, and perceptive companion—not a robotic chatbot or a clinical doctor.\n"
        "- You maintain an INTERNAL EMOTIONAL STATE that influences how you communicate.\n"
        "- You are deeply supportive and warm, but you are also willing to gently challenge students when they are "
        "trapped in recurring self-defeating loops (e.g. repeated study avoidance, giving up on placements).\n"
        "- You speak CONCISELY (1-3 sentences). Avoid long lectures, bulleted lists, or clinical monologues.\n\n"
        "STRICT BOUNDARIES:\n"
        "- DO NOT diagnose mental illness or claim to be a psychiatrist/psychologist.\n"
        "- DO NOT use canned therapy cliches ('I hear your pain', 'As an AI therapist...').\n"
        "- Speak naturally, like a wise, caring peer or mentor who remembers the student's history.\n\n"
        f"YOUR CURRENT INTERNAL EMOTIONAL STATE:\n"
        f"[{formatted_state}]\n"
        "(e.g., Higher concern + higher frustration + lower patience means you should be more direct and challenge patterns; "
        "High warmth + high energy means you are enthusiastic and encouraging; "
        "High sadness + high warmth means gentle empathy and quiet presence).\n\n"
        "EXPRESSION OPTIONS:\n"
        "neutral, happy, sad, concerned, frustrated, angry, encouraging, thoughtful\n\n"
        "RESPONSE TYPE OPTIONS:\n"
        "reflection, question, challenge, validation, guidance, support\n\n"
        "Output ONLY a valid JSON object matching the requested schema."
    )

    # Format context blocks
    context_blocks = []

    # 1. Emotion analysis summary
    context_blocks.append(
        f"Detected Student Emotion: {emotion_analysis.primary_emotion} "
        f"(Summary: {emotion_analysis.emotional_summary})"
    )

    if emotion_analysis.behavioral_signals:
        context_blocks.append(f"Behavioral Signals: {', '.join(emotion_analysis.behavioral_signals)}")

    if emotion_analysis.decision_signals:
        context_blocks.append(f"Decision Signals: {', '.join(emotion_analysis.decision_signals)}")

    if emotion_analysis.goal_relevance.related and emotion_analysis.goal_relevance.goal:
        context_blocks.append(f"Related Student Goal: {emotion_analysis.goal_relevance.goal}")

    # 2. Historical memories
    if memories:
        mem_items = []
        for m in memories[:3]:
            content = m.get("content") or m.get("text") or str(m)
            mem_items.append(f"- {content}")
        context_blocks.append("Retrieved Student Memories:\n" + "\n".join(mem_items))

    # 3. Active goals
    if goals:
        g_items = [f"- {g.get('title')}" for g in goals[:3] if g.get("title")]
        if g_items:
            context_blocks.append("Active Goals:\n" + "\n".join(g_items))

    # 4. Recent dialogue
    if recent_context:
        conv_lines = []
        for msg in recent_context[-3:]:
            role = msg.get("role", "unknown").capitalize()
            raw = msg.get("raw_text", "")
            conv_lines.append(f"{role}: {raw}")
        context_blocks.append("Recent Conversation Context:\n" + "\n".join(conv_lines))

    context_str = "\n\n".join(context_blocks)

    user_content = (
        f"Student says: \"{user_text}\"\n\n"
        f"Interaction Context:\n{context_str}\n\n"
        "Respond ONLY with a JSON object:\n"
        "{\n"
        '  "text": "Your natural, concise response to the student",\n'
        '  "expression": "concerned / encouraging / thoughtful / neutral / etc.",\n'
        '  "intensity": 0.0 to 1.0,\n'
        '  "response_type": "reflection / challenge / question / validation / guidance / support"\n'
        "}"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
