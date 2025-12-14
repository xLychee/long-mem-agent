"""LoCoMo Agent V2 - with multi-provider support and unified config.

This is an enhanced version of the agent that supports:
- Multiple LLM providers via unified config
- YAML-based experiment configuration
- Better experiment tracking
"""

import logging
from typing import Any

from .config import AgentConfig, get_model_config
from .data_models import Conversation, QAAnnotation
from .llm_client import BaseLLMClient, create_llm_client
from .retriever import DialogChunker, SimpleRetriever

logger = logging.getLogger(__name__)


# System prompt for the QA agent
QA_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about conversations between two people.

You will be given:
1. Context from a long conversation between two speakers
2. A question about the conversation

Instructions:
- Answer the question based ONLY on the provided context
- Be concise and specific
- If the information is not in the context, say "I don't know" or "Not mentioned"
- Do not make up information that is not in the context
"""

RAG_QA_PROMPT_TEMPLATE = """Below is relevant context from a conversation:

{context}

---

Question: {question}

Answer the question based on the context above. Be concise and specific."""


FULL_CONTEXT_QA_PROMPT_TEMPLATE = """Below is a conversation between {speaker_a} and {speaker_b}:

{conversation}

---

Question: {question}

Answer the question based on the conversation above. Be concise and specific."""


class LoCoMoAgentV2:
    """Enhanced LoCoMo Agent with multi-provider support.

    This agent uses the unified configuration system and supports
    multiple LLM providers including OpenAI, Anthropic, and local models.

    Attributes:
        config: Agent configuration.
        llm_client: LLM client for generation.
        retriever: RAG retriever (if use_rag=True).
    """

    def __init__(self, config: AgentConfig | None = None, **kwargs: Any) -> None:
        """Initialize the agent.

        Args:
            config: Agent configuration. If None, creates default config.
            **kwargs: Override specific config values.
        """
        if config is None:
            config = AgentConfig(**kwargs)
        self.config = config

        # Initialize LLM client
        self.llm_client: BaseLLMClient = create_llm_client(config.model)
        logger.info(f"Initialized LLM client: {config.model.name} ({config.model.provider})")

        # Initialize retriever for RAG mode
        if config.use_rag:
            self.retriever = SimpleRetriever(
                model_name=config.retriever.model_name,
                top_k=config.retriever.top_k,
            )
            self.chunker = DialogChunker(
                chunk_size=config.retriever.chunk_size,
                overlap=config.retriever.chunk_overlap,
            )
        else:
            self.retriever = None
            self.chunker = None

        self._current_conversation: Conversation | None = None

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.model.name

    def prepare_conversation(
        self,
        conversation: Conversation,
        use_observations: bool | None = None,
        use_summaries: bool | None = None,
    ) -> None:
        """Prepare a conversation for question answering.

        Args:
            conversation: The conversation to prepare.
            use_observations: Override config setting for observations.
            use_summaries: Override config setting for summaries.
        """
        self._current_conversation = conversation

        # Use config values as defaults
        use_obs = use_observations if use_observations is not None else self.config.retriever.use_observations
        use_sum = use_summaries if use_summaries is not None else self.config.retriever.use_summaries

        if self.config.use_rag and self.retriever and self.chunker:
            # Collect all chunks
            all_chunks: list[str] = []
            all_metadata: list[dict[str, Any]] = []

            # Chunk dialog turns
            dialog_chunks, dialog_meta = self.chunker.chunk_conversation(
                conversation.sessions
            )
            all_chunks.extend(dialog_chunks)
            all_metadata.extend(dialog_meta)

            # Optionally add observations
            if use_obs and conversation.observations_raw:
                obs_chunks, obs_meta = self.chunker.chunk_observations(
                    conversation.observations_raw
                )
                all_chunks.extend(obs_chunks)
                all_metadata.extend(obs_meta)

            # Optionally add summaries
            if use_sum and conversation.session_summaries:
                sum_chunks, sum_meta = self.chunker.chunk_summaries(
                    conversation.session_summaries
                )
                all_chunks.extend(sum_chunks)
                all_metadata.extend(sum_meta)

            # Index all chunks
            self.retriever.index_documents(all_chunks, all_metadata)
            logger.info(
                f"Indexed {len(all_chunks)} chunks for conversation {conversation.sample_id}"
            )

    def _build_full_context(self, conversation: Conversation) -> str:
        """Build full conversation context."""
        lines: list[str] = []

        for session in conversation.sessions:
            lines.append(f"\n=== {session.session_id} ({session.date_time}) ===\n")
            for turn in session.turns:
                lines.append(f"{turn.speaker}: {turn.text}")

        text = "\n".join(lines)

        # Truncate if needed
        max_chars = self.config.max_context_tokens * 4
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n... [conversation truncated due to length]"

        return text

    def _build_rag_context(self, question: str) -> str:
        """Build RAG context by retrieving relevant chunks."""
        if self.retriever is None:
            raise ValueError("Retriever not initialized.")

        results = self.retriever.retrieve(question, top_k=self.config.retriever.top_k)

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Chunk {i}] (relevance: {result['score']:.2f})")
            context_parts.append(result["document"])
            context_parts.append("")

        return "\n".join(context_parts)

    def answer_question(self, question: str) -> str:
        """Answer a question about the prepared conversation.

        Args:
            question: The question to answer.

        Returns:
            The generated answer.
        """
        if self._current_conversation is None:
            raise ValueError("No conversation prepared. Call prepare_conversation first.")

        # Build context based on mode
        if self.config.use_rag:
            context = self._build_rag_context(question)
            user_prompt = RAG_QA_PROMPT_TEMPLATE.format(
                context=context,
                question=question,
            )
        else:
            context = self._build_full_context(self._current_conversation)
            user_prompt = FULL_CONTEXT_QA_PROMPT_TEMPLATE.format(
                speaker_a=self._current_conversation.speaker_a,
                speaker_b=self._current_conversation.speaker_b,
                conversation=context,
                question=question,
            )

        # Generate response using LLM client
        try:
            messages = [
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            answer = self.llm_client.generate(messages)
            return answer.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {e}"

    def evaluate_qa(self, qa_annotation: QAAnnotation) -> dict[str, Any]:
        """Evaluate the agent on a single QA annotation.

        Args:
            qa_annotation: The QA annotation to evaluate.

        Returns:
            Dictionary with question, ground truth, prediction, and metrics.
        """
        from .metrics import compute_all_metrics

        prediction = self.answer_question(qa_annotation.question)
        metrics = compute_all_metrics(prediction, qa_annotation.answer)

        return {
            "question": qa_annotation.question,
            "ground_truth": qa_annotation.answer,
            "prediction": prediction,
            "category": qa_annotation.category,
            **metrics,
        }


# ============================================================================
# Factory function for backward compatibility
# ============================================================================


def create_agent(
    model: str = "gpt-4o-mini",
    use_rag: bool = True,
    **kwargs: Any,
) -> LoCoMoAgentV2:
    """Create an agent with the specified model.

    This is a convenience function for quick agent creation.

    Args:
        model: Model name (preset or provider:name format).
        use_rag: Whether to use RAG mode.
        **kwargs: Additional config overrides.

    Returns:
        Configured LoCoMoAgentV2 instance.
    """
    from .config import AgentConfig

    model_config = get_model_config(model)
    config = AgentConfig(model=model_config, use_rag=use_rag, **kwargs)
    return LoCoMoAgentV2(config)

