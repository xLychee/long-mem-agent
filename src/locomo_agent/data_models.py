"""Pydantic models for LoCoMo data structures."""

from typing import Any

from pydantic import BaseModel, Field


class DialogTurn(BaseModel):
    """A single turn in a conversation dialog."""

    speaker: str
    dia_id: str
    text: str
    img_url: str | list[str] | None = None
    blip_caption: str | list[str] | None = None
    search_query: str | list[str] | None = None


class Session(BaseModel):
    """A conversation session with multiple dialog turns."""

    session_id: str
    date_time: str
    turns: list[DialogTurn]


class QAAnnotation(BaseModel):
    """Question-Answer annotation for evaluation."""

    question: str
    answer: str | int  # Can be string or int (e.g., year)
    category: str | int  # Can be string or int category code
    evidence: list[str] = Field(default_factory=list)


class EventSummary(BaseModel):
    """Event summary for a speaker within a session."""

    session_id: str
    speaker: str
    events: list[str]


class Conversation(BaseModel):
    """A complete conversation with all metadata and annotations."""

    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[Session]
    # Raw observations data - complex nested structure from LoCoMo
    # Format: {session_key: {speaker: [[text, dia_id], ...]}}
    observations_raw: dict[str, Any] = Field(default_factory=dict)
    # Raw session summaries
    session_summaries: dict[str, str] = Field(default_factory=dict)
    # Raw event summaries
    event_summaries_raw: dict[str, Any] = Field(default_factory=dict)
    qa_annotations: list[QAAnnotation] = Field(default_factory=list)

    def get_observations_flat(self) -> list[str]:
        """Get flattened observations as list of strings.

        Returns:
            List of observation text strings.
        """
        observations: list[str] = []
        for session_key, session_obs in self.observations_raw.items():
            if isinstance(session_obs, dict):
                for speaker, obs_list in session_obs.items():
                    if isinstance(obs_list, list):
                        for obs in obs_list:
                            if isinstance(obs, list) and len(obs) >= 1:
                                observations.append(f"[{session_key}] {speaker}: {obs[0]}")
                            elif isinstance(obs, str):
                                observations.append(f"[{session_key}] {speaker}: {obs}")
            elif isinstance(session_obs, list):
                for obs in session_obs:
                    if isinstance(obs, str):
                        observations.append(f"[{session_key}] {obs}")
        return observations


class EvaluationResult(BaseModel):
    """Result of a single QA evaluation."""

    question: str
    ground_truth: str
    prediction: str
    category: str | int
    exact_match: float
    f1_score: float
    bleu_score: float
    rouge_l_score: float


class EvaluationSummary(BaseModel):
    """Summary of evaluation results across all questions."""

    total_questions: int
    avg_exact_match: float
    avg_f1_score: float
    avg_bleu_score: float
    avg_rouge_l_score: float
    category_results: dict[str, dict[str, float]] = Field(default_factory=dict)
