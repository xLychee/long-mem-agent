"""Multi-provider LLM client for LoCoMo Agent.

Supports:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude 3.x)
- Ollama (local models)
- vLLM (local/remote inference server)
- Any OpenAI-compatible API
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from .config import LLMProvider, ModelConfig

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the client.

        Args:
            config: Model configuration.
        """
        self.config = config

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Generate a response from the model.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text response.
        """
        pass

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.name


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI and OpenAI-compatible APIs."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        from openai import OpenAI

        api_key = config.get_api_key()
        if not api_key and config.provider == LLMProvider.OPENAI:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")

        # For Ollama and other local providers, API key can be optional
        self.client = OpenAI(
            api_key=api_key or "not-needed",
            base_url=config.base_url,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Generate using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.config.name,
            messages=messages,  # type: ignore
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            **self.config.extra_params,
        )
        return response.choices[0].message.content or ""


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        from anthropic import Anthropic

        api_key = config.get_api_key()
        if not api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var.")

        self.client = Anthropic(api_key=api_key)

    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Generate using Anthropic API."""
        # Anthropic requires system message to be separate
        system_message = ""
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                chat_messages.append(msg)

        response = self.client.messages.create(
            model=self.config.name,
            system=system_message,
            messages=chat_messages,  # type: ignore
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            **self.config.extra_params,
        )

        # Extract text from response
        text_blocks = [block.text for block in response.content if hasattr(block, "text")]
        return "".join(text_blocks)


def create_llm_client(config: ModelConfig) -> BaseLLMClient:
    """Factory function to create appropriate LLM client.

    Args:
        config: Model configuration.

    Returns:
        LLM client instance.

    Raises:
        ValueError: If provider is not supported.
    """
    if config.provider in (
        LLMProvider.OPENAI,
        LLMProvider.OLLAMA,
        LLMProvider.VLLM,
        LLMProvider.OPENAI_COMPATIBLE,
    ):
        return OpenAIClient(config)
    elif config.provider == LLMProvider.ANTHROPIC:
        return AnthropicClient(config)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")


# ============================================================================
# Convenience function for quick testing
# ============================================================================


def quick_test_model(model_name: str, prompt: str = "Hello, how are you?") -> str:
    """Quick test a model configuration.

    Args:
        model_name: Model name (preset or custom format "provider:name").
        prompt: Test prompt.

    Returns:
        Model response.
    """
    from .config import get_model_config

    config = get_model_config(model_name)
    client = create_llm_client(config)

    response = client.generate([
        {"role": "user", "content": prompt}
    ])

    return response

