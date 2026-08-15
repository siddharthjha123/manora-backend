"""
MANORA LLM Base Abstraction.
Provides OpenRouter / OpenAI compatible client for LLM completions and structured JSON generation.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError

from config.settings import get_settings

logger = logging.getLogger("manora.llm")

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base exception for LLM operations."""
    pass


class LLMJSONParseError(LLMError):
    """Raised when LLM output cannot be parsed into expected JSON schema."""
    pass


class LLMClient:
    """OpenRouter/OpenAI compatible LLM client with structured output validation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.base_url = (base_url or settings.OPENROUTER_BASE_URL).rstrip("/")
        self.model_name = model_name or settings.MODEL_NAME

    def _extract_json_str(self, text: str) -> str:
        """Extracts JSON substring from markdown code blocks or text."""
        text = text.strip()
        # Look for markdown ```json ... ``` or ``` ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        json_mode: bool = False,
        max_tokens: int = 1500,
    ) -> str:
        """Generates text completion using configured OpenRouter / OpenAI model."""
        if not self.api_key:
            logger.info("No OPENROUTER_API_KEY set. Returning fallback deterministic completion.")
            return self._mock_generate(messages, json_mode=json_mode)

        try:
            import httpx
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://manora.ai",
                "X-Title": "MANORA Mental Health Support",
            }

            payload: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})

                content = message.get("content")

                if content is None:
                    logger.error(
                        "LLM returned no message content. Full response: %s",
                        json.dumps(data, ensure_ascii=False)[:2000],
                    )

                    # Some models may return structured/reasoning content instead.
                    reasoning = message.get("reasoning")

                    if reasoning:
                        logger.warning("LLM returned reasoning but no content.")
                    
                    raise LLMError("LLM returned empty message content")

                return content

        except Exception as e:
            logger.error(f"LLM API request failed ({e}). Attempting fallback.")
            if "mock" in self.model_name.lower() or not self.api_key:
                return self._mock_generate(messages, json_mode=json_mode)
            raise LLMError(f"LLM generation failed: {str(e)}") from e

    async def generate_json(
        self,
        messages: List[Dict[str, str]],
        schema: Type[T],
        temperature: float = 0.3,
    ) -> T:
        """Generates response and validates against a Pydantic schema."""
        raw_text = await self.generate(messages, temperature=temperature, json_mode=True)
        json_str = self._extract_json_str(raw_text)

        try:
            parsed_dict = json.loads(json_str)
            return schema.model_validate(parsed_dict)
        except (json.JSONDecodeError, ValidationError) as err:
            logger.warning(f"Failed to parse LLM JSON ({err}). Raw content: {raw_text[:200]}")
            # Try to fix / repair or build fallback matching the schema
            try:
                # If schema has custom construct or fallback
                return self._build_fallback_schema_instance(schema, messages)
            except Exception:
                raise LLMJSONParseError(f"Could not parse valid {schema.__name__} from LLM response: {err}")

    def _mock_generate(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        """Deterministic mock generator for offline development & tests."""
        user_msg = ""
        system_msg = ""
        for m in messages:
            if m.get("role") == "user":
                user_msg = m.get("content", "")
            elif m.get("role") == "system":
                system_msg = m.get("content", "")

        # Emotion Analysis response mock
        if "Emotion" in system_msg or "emotion" in system_msg or "primary_emotion" in system_msg:
            # Detect basic keywords
            is_frustrated = any(w in user_msg.lower() for w in ["study", "netflix", "watching", "again", "frustrated", "give up"])
            primary = "frustration" if is_frustrated else "anxiety"
            return json.dumps({
                "interaction_id": "00000000-0000-0000-0000-000000000000",
                "primary_emotion": primary,
                "emotions": [
                    {
                        "emotion": primary,
                        "intensity": 0.84,
                        "confidence": 0.91,
                        "source": "model_inferred"
                    },
                    {
                        "emotion": "guilt",
                        "intensity": 0.72,
                        "confidence": 0.79,
                        "source": "model_inferred"
                    }
                ],
                "emotional_summary": f"The student appears {primary} and conflicted regarding their current activities.",
                "behavioral_signals": [
                    "avoided planned activity",
                    "continued entertainment despite recognizing the conflict"
                ],
                "decision_signals": [
                    "chose entertainment instead of planned activity"
                ],
                "goal_relevance": {
                    "related": True,
                    "goal": "academic progress"
                }
            })

        # Buddy Agent response mock
        if "Buddy" in system_msg or "buddy" in system_msg or "expression" in system_msg:
            if "again" in user_msg.lower() or "netflix" in user_msg.lower():
                text = "You're repeating the same pattern again. Do you actually want to achieve this goal?"
                expr = "concerned"
                resp_type = "challenge"
            else:
                text = "I hear how overwhelming that feels right now. Let's take a breath and look at what's in front of you."
                expr = "thoughtful"
                resp_type = "reflection"

            return json.dumps({
                "text": text,
                "expression": expr,
                "intensity": 0.72,
                "response_type": resp_type
            })

        return "I am here with you. Let's work through this step by step."

    def _build_fallback_schema_instance(self, schema: Type[T], messages: List[Dict[str, str]]) -> T:
        """Constructs a valid mock schema instance when LLM parsing encounters errors."""
        mock_raw = self._mock_generate(messages, json_mode=True)
        parsed = json.loads(self._extract_json_str(mock_raw))
        return schema.model_validate(parsed)


# Global singleton instance
llm_client = LLMClient()
