"""
MANORA Data Agent - Prompt Scaffolding.
Templates and prompts for the future LLM-based Data Agent implementation.
"""

from typing import Any, Dict


def build_data_agent_prompt(interaction: Dict[str, Any], emotion_analysis: Dict[str, Any]) -> str:
    """Prompt template for future production Data Agent."""
    return (
        "Identify candidate memories, recurring patterns, and key behavioral decisions "
        f"from interaction: {interaction.get('raw_text')} with emotion analysis: {emotion_analysis}"
    )
