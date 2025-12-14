#!/usr/bin/env python3
"""Run LoCoMo evaluation experiments with YAML configuration.

This script supports:
- Multiple LLM providers (OpenAI, Anthropic, Ollama, vLLM)
- YAML-based experiment configuration
- Result comparison across models
- Experiment tracking with timestamps and tags

Usage:
    # Run with config file
    python scripts/run_experiment.py --config configs/gpt4o-mini.yaml

    # Run with model preset
    python scripts/run_experiment.py --model gpt-4o-mini

    # Run with custom model
    python scripts/run_experiment.py --model ollama:llama3.2

    # Compare multiple models
    python scripts/run_experiment.py --compare gpt-4o-mini claude-3-5-sonnet ollama:llama3.2
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

console = Console()


def run_single_experiment(
    model_name: str | None = None,
    config_path: Path | None = None,
    max_conversations: int | None = None,
) -> dict:
    """Run a single experiment.

    Args:
        model_name: Model name (preset or provider:name format).
        config_path: Path to YAML config file.
        max_conversations: Limit number of conversations.

    Returns:
        Experiment results dictionary.
    """
    from locomo_agent.config import AgentConfig, get_model_config, load_experiment_config
    from locomo_agent.agent_v2 import LoCoMoAgentV2
    from locomo_agent.evaluator import LoCoMoEvaluator

    # Load configuration
    if config_path:
        config = load_experiment_config(config_path)
        console.print(f"[cyan]Loaded config from:[/cyan] {config_path}")
    else:
        model_config = get_model_config(model_name or "gpt-4o-mini")
        config = AgentConfig(model=model_config)

    # Override max_conversations if specified
    if max_conversations is not None:
        config.evaluation.max_conversations = max_conversations

    console.print(f"[cyan]Model:[/cyan] {config.model.name}")
    console.print(f"[cyan]Provider:[/cyan] {config.model.provider.value}")
    console.print(f"[cyan]Mode:[/cyan] {'RAG' if config.use_rag else 'Full Context'}")

    # Create agent
    agent = LoCoMoAgentV2(config)

    # Create evaluator
    evaluator = LoCoMoEvaluator(
        agent=agent,
        dataset_path=config.dataset_path,
        cache_dir=config.cache_dir,
    )

    console.print(f"[green]Loaded {len(evaluator.conversations)} conversations[/green]")

    # Run evaluation
    summary = evaluator.evaluate_all(
        use_observations=config.retriever.use_observations,
        use_summaries=config.retriever.use_summaries,
        max_conversations=config.evaluation.max_conversations,
    )

    # Print results
    evaluator.print_summary(summary)

    # Save results
    experiment_name = config.evaluation.experiment_name or config.model.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = config.evaluation.output_dir / f"{experiment_name}_{timestamp}.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "model": config.model.model_dump(),
        "config": {
            "use_rag": config.use_rag,
            "retriever": config.retriever.model_dump(),
        },
        "summary": summary.model_dump(),
        "tags": config.evaluation.tags,
    }

    if config.evaluation.save_predictions:
        results["predictions"] = evaluator.results

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    console.print(f"\n[green]Results saved to:[/green] {output_file}")

    return results


def compare_models(model_names: list[str], max_conversations: int | None = None) -> None:
    """Compare multiple models.

    Args:
        model_names: List of model names to compare.
        max_conversations: Limit conversations for faster comparison.
    """
    console.print("\n[bold magenta]🔬 Model Comparison[/bold magenta]\n")

    all_results = []

    for model_name in model_names:
        console.print(f"\n[bold blue]{'='*50}[/bold blue]")
        console.print(f"[bold blue]Running: {model_name}[/bold blue]")
        console.print(f"[bold blue]{'='*50}[/bold blue]")

        try:
            results = run_single_experiment(
                model_name=model_name,
                max_conversations=max_conversations,
            )
            all_results.append(results)
        except Exception as e:
            console.print(f"[red]Error with {model_name}: {e}[/red]")
            continue

    # Print comparison table
    if len(all_results) > 1:
        console.print("\n[bold green]📊 Comparison Results[/bold green]\n")

        table = Table(title="Model Comparison")
        table.add_column("Model", style="cyan")
        table.add_column("EM", style="green")
        table.add_column("F1", style="green")
        table.add_column("BLEU", style="green")
        table.add_column("ROUGE-L", style="green")

        for result in all_results:
            summary = result["summary"]
            table.add_row(
                result["model"]["name"],
                f"{summary['avg_exact_match']:.4f}",
                f"{summary['avg_f1_score']:.4f}",
                f"{summary['avg_bleu_score']:.4f}",
                f"{summary['avg_rouge_l_score']:.4f}",
            )

        console.print(table)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run LoCoMo experiments with configurable models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with YAML config
  python scripts/run_experiment.py --config configs/gpt4o-mini.yaml

  # Run with model preset
  python scripts/run_experiment.py --model gpt-4o-mini

  # Run with Ollama local model
  python scripts/run_experiment.py --model ollama:llama3.2

  # Compare multiple models
  python scripts/run_experiment.py --compare gpt-4o-mini ollama:llama3.2
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path, help="Path to YAML config file")
    group.add_argument("--model", type=str, help="Model name (preset or provider:name)")
    group.add_argument("--compare", nargs="+", help="Compare multiple models")

    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Limit conversations for testing",
    )

    args = parser.parse_args()

    console.print("\n[bold magenta]🧠 LoCoMo Experiment Runner[/bold magenta]\n")

    try:
        if args.compare:
            compare_models(args.compare, args.max_conversations)
        else:
            run_single_experiment(
                model_name=args.model,
                config_path=args.config,
                max_conversations=args.max_conversations,
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()

