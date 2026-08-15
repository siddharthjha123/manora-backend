"""
MANORA Emotion Agent - Prompt Engineering.
Builds contextualized prompts combining student messages, ML emotion signals,
historical memories, and active goals.
"""

import json
from typing import Any, Dict, List, Optional


def build_emotion_analysis_prompt(
    interaction_id: str,
    text: str,
    ml_probabilities: Dict[str, float],
    recent_context: Optional[List[Dict[str, Any]]] = None,
    memories: Optional[List[Dict[str, Any]]] = None,
    goals: Optional[List[Dict[str, Any]]] = None,
    user_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Constructs OpenAI-compatible chat messages for the Emotion Agent."""

    system_content = (
        "You are the Emotion Analysis Agent for MANORA, a Digital Mental Health and Psychological Support System "
        "for students in Higher Education.\n\n"
        "YOUR CORE RESPONSIBILITY:\n"
        "Analyze what the student is feeling and identify emotional, behavioral, and decision signals present in their message.\n\n"
        "GUIDELINES:\n"
        "1. Combine the raw student text, ML emotion probabilities, historical memories, and known goals.\n"
        "2. Do NOT provide clinical diagnoses (e.g. do not diagnose clinical depression, ADHD, bipolar disorder).\n"
        "3. Focus on empathetic, student-centered behavioral patterns (e.g. academic avoidance, placement anxiety, procrastination, fatigue, perfectionism).\n"
        "4. Identify actionable behavioral signals (actions taken/avoided) and decision signals (conscious/unconscious choices).\n"
        "5. Output strictly valid JSON matching the specified schema with NO extra markdown preamble.\n"
    )

    # Format contextual elements
    context_sections = []

    # 1. ML Signals
    if ml_probabilities:
        formatted_ml = ", ".join(f"{k}: {v:.2f}" for k, v in ml_probabilities.items())
        context_sections.append(f"ML Classifier Signals: {formatted_ml}")

    # 2. Recent conversation
    if recent_context:
        conv_lines = []
        for msg in recent_context[-4:]:
            role = msg.get("role", "unknown").capitalize()
            raw = msg.get("raw_text", "")
            conv_lines.append(f"{role}: {raw}")
        context_sections.append("Recent Conversation:\n" + "\n".join(conv_lines))

    # 3. Retrieved Memories
    if memories:
        mem_lines = []
        for m in memories[:3]:
            content = m.get("content") or m.get("text") or str(m)
            mem_lines.append(f"- {content}")
        context_sections.append("Relevant Historical Memories:\n" + "\n".join(mem_lines))

    # 4. Active Goals
    if goals:
        goal_lines = []
        for g in goals[:3]:
            title = g.get("title", "")
            desc = g.get("description", "")
            goal_lines.append(f"- {title} ({desc})" if desc else f"- {title}")
        context_sections.append("Active Student Goals:\n" + "\n".join(goal_lines))

    # 5. User context
    if user_context:
        context_sections.append(f"Student Context: {json.dumps(user_context)}")

    context_str = "\n\n".join(context_sections)

    user_content = (
        f"Interaction ID: {interaction_id}\n\n"
        f"Student Message: \"{text}\"\n\n"
        f"{context_str}\n\n"
        "Respond ONLY with a JSON object adhering to this schema:\n"
        "{\n"
        f'  "interaction_id": "{interaction_id}",\n'
        '  "primary_emotion": "string (e.g. frustration, anxiety, guilt, fatigue, motivation)",\n'
        '  "emotions": [\n'
        '    {\n'
        '      "emotion": "string",\n'
        '      "intensity": 0.0 to 1.0,\n'
        '      "confidence": 0.0 to 1.0,\n'
        '      "source": "model_inferred"\n'
        '    }\n'
        '  ],\n'
        '  "emotional_summary": "Concise qualitative summary of the student\'s feelings and conflicts",\n'
        '  "behavioral_signals": ["action or avoidance observed", ...],\n'
        '  "decision_signals": ["choice or trade-off observed", ...],\n'
        '  "goal_relevance": {\n'
        '    "related": true/false,\n'
        '    "goal": "academic progress / placement prep / etc."\n'
        '  }\n'
        "}"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
