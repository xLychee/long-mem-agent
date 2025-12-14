"""MVP Agent for LoCoMo evaluation."""

import logging
import os
from typing import Any

from openai import OpenAI

from .data_models import Conversation, QAAnnotation
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


class LoCoMoAgent:
    """MVP Agent for answering questions about long conversations.

    This agent supports two modes:
    1. Full Context: Uses the entire conversation as context (may be truncated)
    2. RAG: Uses retrieval-augmented generation with relevant chunks

    Attributes:
        model: The LLM model to use (e.g., "gpt-4o-mini", "gpt-3.5-turbo").
        use_rag: Whether to use RAG mode.
        retriever: The retriever instance for RAG mode.
        top_k: Number of chunks to retrieve in RAG mode.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        use_rag: bool = True,
        retriever_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        max_context_tokens: int = 8000,
        api_key: str | None = None,
    ) -> None:
        """Initialize the agent.

        Args:
            model: LLM model name for generation.
            use_rag: Whether to use RAG (True) or full context (False).
            retriever_model: Sentence transformer model for retrieval.
            top_k: Number of chunks to retrieve in RAG mode.
            max_context_tokens: Maximum context tokens for full context mode.
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        """
        self.model = model
        self.use_rag = use_rag
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens

        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )
        self.client = OpenAI(api_key=api_key)

        # Initialize retriever for RAG mode
        if use_rag:
            self.retriever = SimpleRetriever(
                model_name=retriever_model,
                top_k=top_k,
            )
            self.chunker = DialogChunker(chunk_size=5, overlap=2)
        else:
            self.retriever = None
            self.chunker = None

        self._current_conversation: Conversation | None = None

    def prepare_conversation(
        self,
        conversation: Conversation,
        use_observations: bool = False,
        use_summaries: bool = False,
    ) -> None:
        """Prepare a conversation for question answering.

        In RAG mode, this indexes the conversation chunks.
        In full context mode, this just stores the conversation.

        Args:
            conversation: The conversation to prepare.
            use_observations: Whether to include observations in RAG index.
            use_summaries: Whether to include summaries in RAG index.
        """
        self._current_conversation = conversation

        if self.use_rag and self.retriever and self.chunker:
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
            if use_observations and conversation.observations_raw:
                obs_chunks, obs_meta = self.chunker.chunk_observations(
                    conversation.observations_raw
                )
                all_chunks.extend(obs_chunks)
                all_metadata.extend(obs_meta)

            # Optionally add summaries
            if use_summaries and conversation.session_summaries:
                sum_chunks, sum_meta = self.chunker.chunk_summaries(
                    conversation.session_summaries
                )
                all_chunks.extend(sum_chunks)
                all_metadata.extend(sum_meta)

            # Index all chunks
            self.retriever.index_documents(all_chunks, all_metadata)
            logger.info(f"Indexed {len(all_chunks)} chunks for conversation {conversation.sample_id}")

    def _build_full_context(self, conversation: Conversation) -> str:
        """Build full conversation context.

        Args:
            conversation: The conversation object.

        Returns:
            Formatted conversation text.
        """
        lines: list[str] = []

        for session in conversation.sessions:
            lines.append(f"\n=== {session.session_id} ({session.date_time}) ===\n")
            for turn in session.turns:
                lines.append(f"{turn.speaker}: {turn.text}")

        text = "\n".join(lines)

        # Truncate if needed (rough token estimate: 4 chars per token)
        max_chars = self.max_context_tokens * 4
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n... [conversation truncated due to length]"

        return text

    def _build_rag_context(self, question: str) -> str:
        """Build RAG context by retrieving relevant chunks.

        Args:
            question: The question to answer.

        Returns:
            Retrieved context formatted as text.
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call prepare_conversation first.")

        results = self.retriever.retrieve(question, top_k=self.top_k)

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

        Raises:
            ValueError: If no conversation is prepared.
        """
        if self._current_conversation is None:
            raise ValueError("No conversation prepared. Call prepare_conversation first.")

        # Build context based on mode
        if self.use_rag:
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

        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,  # Deterministic for evaluation
                max_tokens=256,
            )
            answer = response.choices[0].message.content or ""
            return answer.strip()
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return f"Error: {e}"

    def evaluate_qa(
        self,
        qa_annotation: QAAnnotation,
    ) -> dict[str, Any]:
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

