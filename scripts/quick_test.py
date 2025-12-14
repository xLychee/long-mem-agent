#!/usr/bin/env python3
"""Quick test script to verify the agent works.

This script runs a minimal test with just 1 conversation and a few questions.
Use this to verify your setup is working before running full evaluation.

Usage:
    python scripts/quick_test.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()

console = Console()


def main() -> None:
    """Run quick test."""
    console.print(Panel.fit("🧪 LoCoMo Agent Quick Test", style="bold blue"))

    # Import after path setup
    from locomo_agent.agent import LoCoMoAgent
    from locomo_agent.data_loader import load_locomo_dataset
    from locomo_agent.metrics import compute_all_metrics

    # Load dataset
    console.print("\n[cyan]Loading dataset...[/cyan]")
    conversations = load_locomo_dataset()
    console.print(f"[green]✓ Loaded {len(conversations)} conversations[/green]")

    # Get first conversation
    conv = conversations[0]
    console.print(f"\n[yellow]Testing with conversation: {conv.sample_id}[/yellow]")
    console.print(f"  Speakers: {conv.speaker_a} & {conv.speaker_b}")
    console.print(f"  Sessions: {len(conv.sessions)}")
    console.print(f"  QA pairs: {len(conv.qa_annotations)}")

    # Create agent in RAG mode
    console.print("\n[cyan]Initializing agent (RAG mode)...[/cyan]")
    agent = LoCoMoAgent(
        model="gpt-4o-mini",
        use_rag=True,
        top_k=5,
    )

    # Prepare conversation
    console.print("[cyan]Indexing conversation...[/cyan]")
    agent.prepare_conversation(conv)
    console.print("[green]✓ Conversation indexed[/green]")

    # Test on first 3 QA pairs
    console.print("\n[bold]Testing QA pairs:[/bold]")
    test_qas = conv.qa_annotations[:3]

    for i, qa in enumerate(test_qas, 1):
        console.print(f"\n[cyan]Q{i}:[/cyan] {qa.question}")
        console.print(f"[dim]Category: {qa.category}[/dim]")

        prediction = agent.answer_question(qa.question)
        metrics = compute_all_metrics(prediction, qa.answer)

        console.print(f"[green]Ground Truth:[/green] {qa.answer}")
        console.print(f"[yellow]Prediction:[/yellow] {prediction}")
        console.print(
            f"[dim]Metrics: EM={metrics['exact_match']:.2f}, "
            f"F1={metrics['f1_score']:.2f}, "
            f"BLEU={metrics['bleu_score']:.2f}[/dim]"
        )

    console.print("\n[bold green]✓ Quick test completed successfully![/bold green]")


if __name__ == "__main__":
    main()

