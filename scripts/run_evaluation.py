#!/usr/bin/env python3
"""Standalone script to run LoCoMo evaluation.

This script provides a simple way to run the evaluation without
installing the package. It can also be used for quick testing.

Usage:
    python scripts/run_evaluation.py --model gpt-4o-mini --mode rag
    python scripts/run_evaluation.py --model gpt-4o-mini --mode full_context --max-conversations 1
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from locomo_agent.agent import LoCoMoAgent
from locomo_agent.evaluator import LoCoMoEvaluator

console = Console()


def main() -> None:
    """Run LoCoMo evaluation."""
    parser = argparse.ArgumentParser(description="Run LoCoMo Evaluation")

    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--mode", choices=["rag", "full_context"], default="rag")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-observations", action="store_true")
    parser.add_argument("--use-summaries", action="store_true")
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--output", type=str, default="./results/eval_results.json")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # Load .env file
    load_dotenv()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console)],
    )

    console.print("\n[bold magenta]🧠 LoCoMo Evaluation[/bold magenta]\n")
    console.print(f"  Model: [cyan]{args.model}[/cyan]")
    console.print(f"  Mode: [cyan]{args.mode}[/cyan]")

    # Create agent
    agent = LoCoMoAgent(
        model=args.model,
        use_rag=(args.mode == "rag"),
        top_k=args.top_k,
    )

    # Create evaluator (will download dataset if needed)
    evaluator = LoCoMoEvaluator(agent=agent)

    # Run evaluation
    summary = evaluator.evaluate_all(
        use_observations=args.use_observations,
        use_summaries=args.use_summaries,
        max_conversations=args.max_conversations,
    )

    # Print results
    evaluator.print_summary(summary)
    evaluator.save_results(args.output, summary)


if __name__ == "__main__":
    main()

