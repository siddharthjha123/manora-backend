"""
Dedicated LLM client for the Manora Data Agent.

This client reuses the shared LLMClient abstraction but has its own:
- API key
- base URL
- model
- reasoning configuration
- token usage tracking

The Data Agent can therefore use a different model/provider
without affecting the Emotion Agent or Buddy Agent.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from llm.base import LLMClient, LLMError

from config.settings import get_settings


logger = logging.getLogger("manora.data_agent.llm")


class DataAgentLLMClient(LLMClient):
    """
    LLM client dedicated to the Data Agent.

    It inherits the common JSON parsing and Pydantic validation
    behavior from LLMClient, but sends requests to the Data Agent's
    configured model.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the Data Agent LLM client.

        Values can be passed directly or loaded from environment
        variables.
        """

        settings = get_settings()

        super().__init__(
            api_key=api_key or settings.DATA_AGENT_API_KEY,
            base_url=base_url or settings.DATA_AGENT_BASE_URL,
            model_name=model_name or settings.DATA_AGENT_MODEL_NAME,
        )

        self.enable_thinking = settings.DATA_AGENT_ENABLE_THINKING
        self.reasoning_effort = settings.DATA_AGENT_REASONING_EFFORT

        # Stores information about the most recent request.
        self.last_usage: Dict[str, Any] = {}
        
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        json_mode: bool = False,
        max_tokens: int = 3000,
    ) -> str:
        """
        Send one request to the Data Agent model.

        This method is intentionally separate from the shared
        LLMClient.generate() so that the Data Agent's model,
        reasoning settings, timing, and token usage are easy to inspect.
        """

        if not self.api_key:
            raise LLMError(
                "DATA_AGENT_API_KEY is not configured."
            )

        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # The college gateway supports Qwen thinking controls
        # through chat_template_kwargs.
        # payload["extra_body"] = {
        #     "chat_template_kwargs": {
        #         "enable_thinking": self.enable_thinking,
        #         "reasoning_effort": self.reasoning_effort,
        #     }
        # }
        payload["chat_template_kwargs"] = {
            "enable_thinking": self.enable_thinking,
            "reasoning_effort": self.reasoning_effort,
        }

        if json_mode:
            payload["response_format"] = {
                "type": "json_object"
            }

        start_time = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )

                response.raise_for_status()

                data = response.json()

        except httpx.HTTPError as exc:
            raise LLMError(
                f"Data Agent LLM request failed: {exc}"
            ) from exc

        elapsed_seconds = time.perf_counter() - start_time

        self._record_usage(
            response_data=data,
            elapsed_seconds=elapsed_seconds,
        )

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        content = message.get("content")

        if content is None:
            reasoning = message.get("reasoning")

            logger.error(
                "Data Agent LLM returned no content. "
                "finish_reason=%s",
                choice.get("finish_reason"),
            )

            if reasoning:
                logger.error(
                    "The model returned reasoning but no final content."
                )

            raise LLMError(
                "Data Agent LLM returned empty message content."
            )

        return content

    def _record_usage(
        self,
        response_data: Dict[str, Any],
        elapsed_seconds: float,
    ) -> None:
        """
        Extract token usage and timing information from the API response.
        """

        usage = response_data.get("usage", {})
        choices = response_data.get("choices", [{}])

        finish_reason = None

        if choices:
            finish_reason = choices[0].get("finish_reason")

        self.last_usage = {
            "model": response_data.get(
                "model",
                self.model_name,
            ),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "finish_reason": finish_reason,
            "elapsed_seconds": round(
                elapsed_seconds,
                3,
            ),
        }

        logger.info(
            "Data Agent LLM usage | "
            "model=%s | input=%s | output=%s | total=%s | "
            "time=%.3fs | finish=%s",
            self.last_usage["model"],
            self.last_usage["prompt_tokens"],
            self.last_usage["completion_tokens"],
            self.last_usage["total_tokens"],
            self.last_usage["elapsed_seconds"],
            self.last_usage["finish_reason"],
        )


data_agent_llm_client = DataAgentLLMClient()
