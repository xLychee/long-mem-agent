"""Evaluator for LoCoMo QA task."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .data_loader import load_locomo_dataset
from .data_models import Conversation, EvaluationSummary

# Type alias for agent - supports both original and V2 agent
# Using Any to support duck typing for any agent with prepare_conversation and evaluate_qa methods
from typing import Protocol, runtime_checkable


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol for LoCoMo agents."""

    def prepare_conversation(
        self,
        conversation: Conversation,
        use_observations: bool = False,
        use_summaries: bool = False,
    ) -> None: ...

    def evaluate_qa(self, qa_annotation: Any) -> dict[str, Any]: ...

logger = logging.getLogger(__name__)
console = Console()


class LoCoMoEvaluator:
    """Evaluator for running LoCoMo QA evaluation.

    This class handles:
    1. Loading the dataset
    2. Running the agent on all QA annotations
    3. Computing and aggregating metrics
    4. Saving results

    Attributes:
        agent: The LoCoMoAgent instance to evaluate.
        conversations: List of loaded conversations.
    """

    def __init__(
        self,
        agent: AgentProtocol,
        dataset_path: Path | str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            agent: The agent to evaluate (supports LoCoMoAgent or LoCoMoAgentV2).
            dataset_path: Path to dataset file. If None, downloads from GitHub.
            cache_dir: Directory to cache downloaded dataset.
        """
        self.agent = agent
        self.conversations = load_locomo_dataset(dataset_path, cache_dir)
        self.results: list[dict[str, Any]] = []

    def evaluate_conversation(
        self,
        conversation: Conversation,
        use_observations: bool = False,
        use_summaries: bool = False,
    ) -> list[dict[str, Any]]:
        """Evaluate the agent on a single conversation.

        Args:
            conversation: The conversation to evaluate.
            use_observations: Whether to use observations in RAG.
            use_summaries: Whether to use summaries in RAG.

        Returns:
            List of evaluation results for each QA annotation.
        """
        # Prepare conversation for the agent
        self.agent.prepare_conversation(
            conversation,
            use_observations=use_observations,
            use_summaries=use_summaries,
        )

        results = []
        for qa in conversation.qa_annotations:
            result = self.agent.evaluate_qa(qa)
            result["sample_id"] = conversation.sample_id
            results.append(result)

        return results

    def evaluate_all(
        self,
        use_observations: bool = False,
        use_summaries: bool = False,
        max_conversations: int | None = None,
    ) -> EvaluationSummary:
        """Run evaluation on all conversations.

        Args:
            use_observations: Whether to use observations in RAG.
            use_summaries: Whether to use summaries in RAG.
            max_conversations: Maximum number of conversations to evaluate.

        Returns:
            EvaluationSummary with aggregated results.
        """
        self.results = []
        conversations = self.conversations[:max_conversations] if max_conversations else self.conversations

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Evaluating {len(conversations)} conversations...",
                total=len(conversations),
            )

            for conv in conversations:
                console.print(f"\n[bold blue]Processing: {conv.sample_id}[/bold blue]")
                console.print(f"  Sessions: {len(conv.sessions)}, QA pairs: {len(conv.qa_annotations)}")

                conv_results = self.evaluate_conversation(
                    conv,
                    use_observations=use_observations,
                    use_summaries=use_summaries,
                )
                self.results.extend(conv_results)

                progress.advance(task)

        return self._compute_summary()

    def _compute_summary(self) -> EvaluationSummary:
        """Compute summary statistics from results.

        Returns:
            EvaluationSummary with aggregated metrics.
        """
        if not self.results:
            return EvaluationSummary(
                total_questions=0,
                avg_exact_match=0.0,
                avg_f1_score=0.0,
                avg_bleu_score=0.0,
                avg_rouge_l_score=0.0,
            )

        # Overall averages
        total = len(self.results)
        avg_em = sum(r["exact_match"] for r in self.results) / total
        avg_f1 = sum(r["f1_score"] for r in self.results) / total
        avg_bleu = sum(r["bleu_score"] for r in self.results) / total
        avg_rouge = sum(r["rouge_l_score"] for r in self.results) / total

        # Per-category results
        category_results: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        category_counts: dict[str, int] = defaultdict(int)

        for r in self.results:
            cat = r["category"]
            category_results[cat]["exact_match"] += r["exact_match"]
            category_results[cat]["f1_score"] += r["f1_score"]
            category_results[cat]["bleu_score"] += r["bleu_score"]
            category_results[cat]["rouge_l_score"] += r["rouge_l_score"]
            category_counts[cat] += 1

        # Average per category
        for cat in category_results:
            count = category_counts[cat]
            for metric in category_results[cat]:
                category_results[cat][metric] /= count

        return EvaluationSummary(
            total_questions=total,
            avg_exact_match=avg_em,
            avg_f1_score=avg_f1,
            avg_bleu_score=avg_bleu,
            avg_rouge_l_score=avg_rouge,
            category_results=dict(category_results),
        )

    def print_summary(self, summary: EvaluationSummary) -> None:
        """Print evaluation summary to console.

        Args:
            summary: The evaluation summary to print.
        """
        console.print("\n[bold green]═══ Evaluation Results ═══[/bold green]\n")

        # Overall metrics table
        table = Table(title="Overall Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="magenta")

        table.add_row("Total Questions", str(summary.total_questions))
        table.add_row("Exact Match", f"{summary.avg_exact_match:.4f}")
        table.add_row("F1 Score", f"{summary.avg_f1_score:.4f}")
        table.add_row("BLEU-1", f"{summary.avg_bleu_score:.4f}")
        table.add_row("ROUGE-L", f"{summary.avg_rouge_l_score:.4f}")

        console.print(table)

        # Per-category results
        if summary.category_results:
            console.print("\n[bold yellow]Per-Category Results:[/bold yellow]")

            cat_table = Table(title="Category Breakdown")
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("EM", style="green")
            cat_table.add_column("F1", style="green")
            cat_table.add_column("BLEU", style="green")
            cat_table.add_column("ROUGE-L", style="green")

            for cat, metrics in sorted(summary.category_results.items()):
                cat_table.add_row(
                    cat,
                    f"{metrics['exact_match']:.3f}",
                    f"{metrics['f1_score']:.3f}",
                    f"{metrics['bleu_score']:.3f}",
                    f"{metrics['rouge_l_score']:.3f}",
                )

            console.print(cat_table)

    def save_results(
        self,
        output_path: Path | str,
        summary: EvaluationSummary | None = None,
    ) -> None:
        """Save evaluation results to JSON file.

        Args:
            output_path: Path to output file.
            summary: Optional summary to include.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "summary": summary.model_dump() if summary else None,
            "results": self.results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        console.print(f"\n[green]Results saved to {output_path}[/green]")

