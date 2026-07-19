"""LangChain callback handler that surfaces the AI reasoning/thinking process.

When enabled, it logs the exact prompt sent to the model, streams any tokens the
model emits, and logs the raw completion before it is parsed. This makes the
"thinking process" visible in the backend logs whenever a LangChain chain runs.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from app.core.config import get_settings

logger = logging.getLogger("ai_trace")

_SEPARATOR = "=" * 60


def _render_content(content: Any) -> str:
    """Render message content, collapsing multimodal parts (e.g. base64 images)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        rendered = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type == "text":
                    rendered.append(part.get("text", ""))
                elif part_type == "image_url":
                    rendered.append("[image omitted]")
                else:
                    rendered.append(f"[{part_type or 'unknown part'}]")
            else:
                rendered.append(str(part))
        return "\n".join(rendered)
    return str(content)


class ThinkingTraceCallbackHandler(BaseCallbackHandler):
    """Logs prompts, streamed tokens, and raw completions for one chain run."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._streamed_any = False

    def on_chat_model_start(
        self, serialized: dict, messages: list[list[BaseMessage]], **kwargs: Any
    ) -> None:
        logger.info("\n%s\nAI THINKING START [%s]\n%s", _SEPARATOR, self.label, _SEPARATOR)
        for batch in messages:
            for message in batch:
                logger.info("[%s] prompt (%s):\n%s", self.label, message.type, _render_content(message.content))

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        logger.info("\n%s\nAI THINKING START [%s]\n%s", _SEPARATOR, self.label, _SEPARATOR)
        for prompt in prompts:
            logger.info("[%s] prompt:\n%s", self.label, prompt)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # Only fires when the underlying model streams; shows live reasoning.
        self._streamed_any = True
        print(token, end="", flush=True)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self._streamed_any:
            print(flush=True)
        for generations in response.generations:
            for generation in generations:
                logger.info("[%s] model output:\n%s", self.label, generation.text)
        logger.info("%s\nAI THINKING END [%s]\n%s\n", _SEPARATOR, self.label, _SEPARATOR)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        logger.warning("[%s] AI error: %s", self.label, error)


def trace_config(label: str) -> dict:
    """Return an .ainvoke config that attaches the trace handler when enabled."""
    if not get_settings().ai_trace_enabled:
        return {}
    return {"callbacks": [ThinkingTraceCallbackHandler(label)]}
