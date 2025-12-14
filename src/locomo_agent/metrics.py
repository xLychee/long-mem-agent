"""Evaluation metrics for LoCoMo QA task."""

import re
import string
from collections import Counter

import nltk
from rouge_score import rouge_scorer


def normalize_answer(text: str | int) -> str:
    """Normalize answer text for comparison.

    Args:
        text: Raw answer text (can be string or int).

    Returns:
        Normalized text (lowercase, no punctuation, single spaces).
    """
    # Convert to string if needed
    text = str(text)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def exact_match_score(prediction: str | int, ground_truth: str | int) -> float:
    """Calculate exact match score.

    Args:
        prediction: Predicted answer.
        ground_truth: Ground truth answer.

    Returns:
        1.0 if exact match, 0.0 otherwise.
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str | int, ground_truth: str | int) -> float:
    """Calculate token-level F1 score.

    Args:
        prediction: Predicted answer.
        ground_truth: Ground truth answer.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens or not ground_truth_tokens:
        return float(prediction_tokens == ground_truth_tokens)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def bleu_score(prediction: str | int, ground_truth: str | int, n: int = 1) -> float:
    """Calculate BLEU score.

    Args:
        prediction: Predicted answer.
        ground_truth: Ground truth answer.
        n: N-gram order (default: 1 for BLEU-1).

    Returns:
        BLEU score between 0.0 and 1.0.
    """
    try:
        # Convert to string
        prediction = str(prediction)
        ground_truth = str(ground_truth)

        # Ensure nltk data is available
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        prediction_tokens = nltk.word_tokenize(prediction.lower())
        reference_tokens = nltk.word_tokenize(ground_truth.lower())

        if not prediction_tokens or not reference_tokens:
            return 0.0

        # Calculate n-gram BLEU
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        score = nltk.translate.bleu_score.sentence_bleu(
            [reference_tokens],
            prediction_tokens,
            weights=weights,
            smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1,
        )
        return score
    except Exception:
        return 0.0


def rouge_l_score(prediction: str | int, ground_truth: str | int) -> float:
    """Calculate ROUGE-L score.

    Args:
        prediction: Predicted answer.
        ground_truth: Ground truth answer.

    Returns:
        ROUGE-L F1 score between 0.0 and 1.0.
    """
    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(str(ground_truth), str(prediction))
        return scores["rougeL"].fmeasure
    except Exception:
        return 0.0


def compute_all_metrics(prediction: str | int, ground_truth: str | int) -> dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        prediction: Predicted answer.
        ground_truth: Ground truth answer.

    Returns:
        Dictionary with all metric scores.
    """
    return {
        "exact_match": exact_match_score(prediction, ground_truth),
        "f1_score": f1_score(prediction, ground_truth),
        "bleu_score": bleu_score(prediction, ground_truth),
        "rouge_l_score": rouge_l_score(prediction, ground_truth),
    }

