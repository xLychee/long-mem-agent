"""Command-line interface for LoCoMo evaluation."""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from .agent import LoCoMoAgent
from .evaluator import LoCoMoEvaluator

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: Whether to enable verbose logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def main() -> int:
    """Main entry point for CLI.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="LoCoMo Agent - Long-Term Conversational Memory Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use for generation",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rag", "full_context"],
        default="rag",
        help="Evaluation mode: 'rag' for retrieval-augmented, 'full_context' for direct",
    )
    parser.add_argument(
        "--retriever-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for RAG retrieval",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve in RAG mode",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=8000,
        help="Maximum context tokens for full_context mode",
    )
    parser.add_argument(
        "--use-observations",
        action="store_true",
        help="Include observations in RAG index",
    )
    parser.add_argument(
        "--use-summaries",
        action="store_true",
        help="Include session summaries in RAG index",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to LoCoMo dataset JSON (downloads from GitHub if not provided)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./data",
        help="Directory to cache downloaded dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/evaluation_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Maximum number of conversations to evaluate (for testing)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Setup logging
    setup_logging(args.verbose)

    console.print("[bold blue]╔═══════════════════════════════════════════════════╗[/bold blue]")
    console.print("[bold blue]║     LoCoMo Agent - Long-Term Memory Evaluation     ║[/bold blue]")
    console.print("[bold blue]╚═══════════════════════════════════════════════════╝[/bold blue]")
    console.print()

    try:
        # Initialize agent
        console.print(f"[cyan]Model:[/cyan] {args.model}")
        console.print(f"[cyan]Mode:[/cyan] {args.mode}")

        agent = LoCoMoAgent(
            model=args.model,
            use_rag=(args.mode == "rag"),
            retriever_model=args.retriever_model,
            top_k=args.top_k,
            max_context_tokens=args.max_context_tokens,
        )

        # Initialize evaluator
        evaluator = LoCoMoEvaluator(
            agent=agent,
            dataset_path=args.dataset,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        )

        console.print(f"[green]Loaded {len(evaluator.conversations)} conversations[/green]")

        # Run evaluation
        summary = evaluator.evaluate_all(
            use_observations=args.use_observations,
            use_summaries=args.use_summaries,
            max_conversations=args.max_conversations,
        )

        # Print and save results
        evaluator.print_summary(summary)
        evaluator.save_results(args.output, summary)

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logging.exception("Evaluation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

