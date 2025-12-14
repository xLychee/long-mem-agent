"""Data loader for LoCoMo dataset."""

import json
import logging
from pathlib import Path
from typing import Any

import httpx

from .data_models import Conversation, DialogTurn, QAAnnotation, Session

logger = logging.getLogger(__name__)

# LoCoMo dataset URL from GitHub
LOCOMO_DATA_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)


def download_dataset(cache_dir: Path | None = None) -> Path:
    """Download LoCoMo dataset from GitHub if not already cached.

    Args:
        cache_dir: Directory to cache the dataset. Defaults to ./data/

    Returns:
        Path to the downloaded dataset file.

    Raises:
        httpx.HTTPError: If download fails.
    """
    if cache_dir is None:
        cache_dir = Path("./data")

    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = cache_dir / "locomo10.json"

    if dataset_path.exists():
        logger.info(f"Using cached dataset at {dataset_path}")
        return dataset_path

    logger.info(f"Downloading LoCoMo dataset from {LOCOMO_DATA_URL}")
    response = httpx.get(LOCOMO_DATA_URL, timeout=60.0)
    response.raise_for_status()

    dataset_path.write_text(response.text, encoding="utf-8")
    logger.info(f"Dataset saved to {dataset_path}")

    return dataset_path


def parse_conversation(raw_data: dict[str, Any]) -> Conversation:
    """Parse raw JSON data into a Conversation object.

    Args:
        raw_data: Raw dictionary from JSON file.

    Returns:
        Parsed Conversation object.
    """
    # Parse sessions
    sessions: list[Session] = []
    conversation_data = raw_data.get("conversation", {})

    # Find all session keys
    session_keys = sorted(
        [k for k in conversation_data.keys() if k.startswith("session_") and "_date_time" not in k]
    )

    speaker_a = conversation_data.get("speaker_a", "Speaker A")
    speaker_b = conversation_data.get("speaker_b", "Speaker B")

    for session_key in session_keys:
        session_turns = conversation_data.get(session_key, [])
        date_time_key = f"{session_key}_date_time"
        date_time = conversation_data.get(date_time_key, "")

        turns = []
        for turn_data in session_turns:
            turn = DialogTurn(
                speaker=turn_data.get("speaker", ""),
                dia_id=turn_data.get("dia_id", ""),
                text=turn_data.get("text", ""),
                img_url=turn_data.get("img_url"),
                blip_caption=turn_data.get("blip_caption"),
                search_query=turn_data.get("search_query"),
            )
            turns.append(turn)

        sessions.append(
            Session(
                session_id=session_key,
                date_time=date_time,
                turns=turns,
            )
        )

    # Parse observations - store raw format as complex nested structure
    observations_raw = raw_data.get("observation", {})

    # Parse session summaries
    session_summaries: dict[str, str] = raw_data.get("session_summary", {})

    # Parse event summaries - store raw format
    event_summaries_raw = raw_data.get("event_summary", {})

    # Parse QA annotations
    qa_annotations: list[QAAnnotation] = []
    for qa_data in raw_data.get("qa", []):
        qa = QAAnnotation(
            question=qa_data.get("question", ""),
            answer=qa_data.get("answer", ""),
            category=qa_data.get("category", ""),
            evidence=qa_data.get("evidence", []),
        )
        qa_annotations.append(qa)

    return Conversation(
        sample_id=raw_data.get("sample_id", ""),
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        sessions=sessions,
        observations_raw=observations_raw,
        session_summaries=session_summaries,
        event_summaries_raw=event_summaries_raw,
        qa_annotations=qa_annotations,
    )


def load_locomo_dataset(
    dataset_path: Path | str | None = None,
    cache_dir: Path | None = None,
) -> list[Conversation]:
    """Load and parse the LoCoMo dataset.

    Args:
        dataset_path: Path to existing dataset file. If None, downloads from GitHub.
        cache_dir: Directory to cache downloaded dataset.

    Returns:
        List of parsed Conversation objects.
    """
    if dataset_path is None:
        dataset_path = download_dataset(cache_dir)
    else:
        dataset_path = Path(dataset_path)

    logger.info(f"Loading dataset from {dataset_path}")

    with open(dataset_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    conversations = []
    for item in raw_data:
        try:
            conv = parse_conversation(item)
            conversations.append(conv)
        except Exception as e:
            logger.warning(f"Failed to parse conversation: {e}")
            continue

    logger.info(f"Loaded {len(conversations)} conversations")
    return conversations


def get_conversation_text(conversation: Conversation, max_tokens: int | None = None) -> str:
    """Convert conversation to plain text format.

    Args:
        conversation: Conversation object to convert.
        max_tokens: Optional maximum number of tokens (approximate).

    Returns:
        Plain text representation of the conversation.
    """
    lines: list[str] = []

    for session in conversation.sessions:
        lines.append(f"\n=== {session.session_id} ({session.date_time}) ===\n")
        for turn in session.turns:
            lines.append(f"{turn.speaker}: {turn.text}")

    text = "\n".join(lines)

    # Simple token approximation (rough estimate: 4 chars per token)
    if max_tokens is not None:
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"

    return text

