"""Unified configuration management for LoCoMo Agent.

This module provides a centralized configuration system that supports:
- Multiple LLM providers (OpenAI, Anthropic, local models via Ollama/vLLM)
- Environment variable overrides
- YAML-based experiment configurations
- Easy model comparison and benchmarking
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"  # Local models via Ollama
    VLLM = "vllm"  # Local models via vLLM server
    OPENAI_COMPATIBLE = "openai_compatible"  # Any OpenAI-compatible API


class ModelConfig(BaseModel):
    """Configuration for a specific model."""

    name: str = Field(description="Model name/identifier")
    provider: LLMProvider = Field(default=LLMProvider.OPENAI)
    # API settings
    api_key: str | None = Field(default=None, description="API key (if required)")
    base_url: str | None = Field(default=None, description="Custom API base URL")
    # Generation settings
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1)
    # Provider-specific settings
    extra_params: dict[str, Any] = Field(default_factory=dict)

    def get_api_key(self) -> str | None:
        """Get API key from config or environment."""
        if self.api_key:
            return self.api_key
        # Try environment variables based on provider
        env_vars = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProvider.OLLAMA: None,  # No API key needed
            LLMProvider.VLLM: "VLLM_API_KEY",
            LLMProvider.OPENAI_COMPATIBLE: "LLM_API_KEY",
        }
        env_var = env_vars.get(self.provider)
        return os.getenv(env_var) if env_var else None


class RetrieverConfig(BaseModel):
    """Configuration for the RAG retriever."""

    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )
    top_k: int = Field(default=5, ge=1, description="Number of chunks to retrieve")
    chunk_size: int = Field(default=5, ge=1, description="Dialog turns per chunk")
    chunk_overlap: int = Field(default=2, ge=0, description="Overlapping turns")
    use_observations: bool = Field(default=False)
    use_summaries: bool = Field(default=False)


class EvaluationConfig(BaseModel):
    """Configuration for evaluation runs."""

    max_conversations: int | None = Field(
        default=None, description="Limit conversations (None = all)"
    )
    output_dir: Path = Field(default=Path("./results"))
    save_predictions: bool = Field(default=True)
    # Experiment tracking
    experiment_name: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)


class AgentConfig(BaseSettings):
    """Main configuration for the LoCoMo Agent.

    Configuration priority (highest to lowest):
    1. Explicit values passed to constructor
    2. Environment variables (prefixed with LOCOMO_)
    3. .env file
    4. Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="LOCOMO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model settings
    model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(name="gpt-4o-mini", provider=LLMProvider.OPENAI)
    )

    # RAG settings
    use_rag: bool = Field(default=True, description="Use RAG mode")
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)

    # Context settings (for non-RAG mode)
    max_context_tokens: int = Field(default=8000)

    # Evaluation settings
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    # Data settings
    dataset_path: Path | None = Field(default=None)
    cache_dir: Path = Field(default=Path("./data"))


# ============================================================================
# Preset Model Configurations
# ============================================================================

PRESET_MODELS: dict[str, ModelConfig] = {
    # OpenAI models
    "gpt-4o": ModelConfig(name="gpt-4o", provider=LLMProvider.OPENAI),
    "gpt-4o-mini": ModelConfig(name="gpt-4o-mini", provider=LLMProvider.OPENAI),
    "gpt-4-turbo": ModelConfig(name="gpt-4-turbo", provider=LLMProvider.OPENAI),
    "gpt-3.5-turbo": ModelConfig(name="gpt-3.5-turbo", provider=LLMProvider.OPENAI),
    # Anthropic models
    "claude-3-5-sonnet": ModelConfig(
        name="claude-3-5-sonnet-20241022", provider=LLMProvider.ANTHROPIC
    ),
    "claude-3-5-haiku": ModelConfig(
        name="claude-3-5-haiku-20241022", provider=LLMProvider.ANTHROPIC
    ),
    "claude-3-opus": ModelConfig(
        name="claude-3-opus-20240229", provider=LLMProvider.ANTHROPIC
    ),
    # Local models (Ollama)
    "llama3.2": ModelConfig(
        name="llama3.2",
        provider=LLMProvider.OLLAMA,
        base_url="http://localhost:11434/v1",
    ),
    "llama3.2:1b": ModelConfig(
        name="llama3.2:1b",
        provider=LLMProvider.OLLAMA,
        base_url="http://localhost:11434/v1",
    ),
    "mistral": ModelConfig(
        name="mistral",
        provider=LLMProvider.OLLAMA,
        base_url="http://localhost:11434/v1",
    ),
    "qwen2.5": ModelConfig(
        name="qwen2.5",
        provider=LLMProvider.OLLAMA,
        base_url="http://localhost:11434/v1",
    ),
    "phi3": ModelConfig(
        name="phi3",
        provider=LLMProvider.OLLAMA,
        base_url="http://localhost:11434/v1",
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get a model configuration by name or create a custom one.

    Args:
        model_name: Preset model name or custom model identifier.
            For custom models, use format: "provider:model_name"
            e.g., "ollama:my-finetuned-model"

    Returns:
        ModelConfig for the specified model.
    """
    # Check presets first
    if model_name in PRESET_MODELS:
        return PRESET_MODELS[model_name]

    # Parse custom format "provider:model_name"
    if ":" in model_name:
        provider_str, name = model_name.split(":", 1)
        try:
            provider = LLMProvider(provider_str.lower())
        except ValueError:
            provider = LLMProvider.OPENAI_COMPATIBLE

        config = ModelConfig(name=name, provider=provider)

        # Set default base_url for local providers
        if provider == LLMProvider.OLLAMA:
            config.base_url = "http://localhost:11434/v1"
        elif provider == LLMProvider.VLLM:
            config.base_url = "http://localhost:8000/v1"

        return config

    # Default to OpenAI
    return ModelConfig(name=model_name, provider=LLMProvider.OPENAI)


def load_experiment_config(yaml_path: Path | str) -> AgentConfig:
    """Load configuration from a YAML file.

    Args:
        yaml_path: Path to YAML configuration file.

    Returns:
        AgentConfig loaded from YAML.
    """
    import yaml

    yaml_path = Path(yaml_path)
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Handle model preset
    if "model" in data and isinstance(data["model"], str):
        data["model"] = get_model_config(data["model"]).model_dump()

    return AgentConfig(**data)

