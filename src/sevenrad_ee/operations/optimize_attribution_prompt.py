"""
CLI command to optimize Perplexity attribution prompts using DSPy.

This research command uses BootstrapFinetune to optimize the prompt
used for greenhouse attribution analysis with Perplexity AI.

Usage:
    uv run python -m sevenrad_ee.operations.optimize_attribution_prompt [OPTIONS]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import dspy
    from dspy.teleprompt import BootstrapFewShot
except ImportError as e:
    print("Error: dspy-ai package required")  # noqa: T201
    print("Install with: uv pip install -e '.[dev]'")  # noqa: T201
    sys.exit(1)

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from sevenrad_ee.ai.dspy_attribution import (
    GreenhouseAttributionModule,
    PerplexityAnalysis,
    create_training_examples,
    greenhouse_attribution_metric,
)

console = Console()
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize Perplexity attribution prompts using DSPy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run optimization with default settings
  uv run python -m sevenrad_ee.operations.optimize_attribution_prompt

  # Use specific Perplexity model
  uv run python -m sevenrad_ee.operations.optimize_attribution_prompt \\
    --model perplexity/sonar-pro

  # Save optimized module
  uv run python -m sevenrad_ee.operations.optimize_attribution_prompt \\
    --output optimized_attribution.json

  # Run with custom number of training examples
  uv run python -m sevenrad_ee.operations.optimize_attribution_prompt \\
    --max-examples 10

  # Dry run to see training examples
  uv run python -m sevenrad_ee.operations.optimize_attribution_prompt \\
    --dry-run
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="perplexity/sonar",
        help="Perplexity model to use (default: perplexity/sonar)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Perplexity API key (default: from PERPLEXITY_API_KEY env var)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save optimized module (JSON format)",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Maximum number of training examples to use (default: 5)",
    )

    parser.add_argument(
        "--max-bootstrapped-demos",
        type=int,
        default=3,
        help="Max bootstrapped demonstrations for few-shot (default: 3)",
    )

    parser.add_argument(
        "--max-labeled-demos",
        type=int,
        default=2,
        help="Max labeled demonstrations to use (default: 2)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show training examples without running optimization",
    )

    parser.add_argument(
        "--test-business",
        type=str,
        default=None,
        help="Test optimized module on a specific business name",
    )

    parser.add_argument(
        "--test-location",
        type=str,
        default=None,
        help="Location for test business (required with --test-business)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def display_header() -> None:
    """Display command header."""
    console.print(
        Panel.fit(
            "[bold cyan]DSPy Prompt Optimization for "
            "Greenhouse Attribution[/bold cyan]\n"
            "Using BootstrapFewShot to optimize Perplexity prompts",
            border_style="cyan",
        )
    )


def display_training_examples(examples: list[dspy.Example]) -> None:
    """Display training examples in a table."""
    table = Table(
        title="Training Examples",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Business", style="green")
    table.add_column("Location", style="white")
    table.add_column("Likelihood", style="yellow")
    table.add_column("Crops", style="magenta")

    for ex in examples:
        likelihood = f"{ex.grow_light_likelihood:.2f}"
        crops = ex.primary_crops if hasattr(ex, "primary_crops") else ""
        table.add_row(
            ex.business_name,
            ex.location_context,
            likelihood,
            crops,
        )

    console.print(table)


def configure_dspy(api_key: str, model: str) -> None:
    """Configure DSPy with Perplexity LM."""
    console.print(f"[cyan]Configuring DSPy with model:[/cyan] {model}")

    try:
        lm = dspy.LM(model, api_key=api_key)
        dspy.configure(lm=lm)
        console.print("[green]✓[/green] DSPy configured successfully\n")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to configure DSPy: {e}", style="bold red")
        raise


def run_optimization(
    trainset: list[dspy.Example],
    max_bootstrapped_demos: int,
    max_labeled_demos: int,
) -> GreenhouseAttributionModule:
    """
    Run DSPy optimization using BootstrapFewShot.

    Args:
        trainset: Training examples
        max_bootstrapped_demos: Max bootstrapped demonstrations
        max_labeled_demos: Max labeled demonstrations

    Returns:
        Optimized GreenhouseAttributionModule

    """
    console.print("[yellow]Starting optimization...[/yellow]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Optimizing prompt with BootstrapFewShot...", total=None
        )

        # Create base module
        base_module = GreenhouseAttributionModule()

        # Configure optimizer
        # Note: Using BootstrapFewShot instead of BootstrapFinetune since
        # BootstrapFinetune requires a finetune-capable model
        optimizer = BootstrapFewShot(
            metric=greenhouse_attribution_metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
        )

        # Run optimization
        try:
            optimized_module = optimizer.compile(
                student=base_module,
                trainset=trainset,
            )
            progress.update(task, completed=True)
            console.print("[green]✓[/green] Optimization complete!\n")

            return optimized_module

        except Exception as e:
            progress.update(task, completed=True)
            console.print(f"[red]✗[/red] Optimization failed: {e}", style="bold red")
            raise


def evaluate_module(
    module: GreenhouseAttributionModule,
    testset: list[dspy.Example],
) -> dict[str, float]:
    """
    Evaluate optimized module on test set.

    Args:
        module: Optimized module to evaluate
        testset: Test examples

    Returns:
        Dictionary with evaluation metrics

    """
    console.print("[yellow]Evaluating optimized module...[/yellow]\n")

    scores = []
    for example in testset:
        prediction = module(
            business_name=example.business_name,
            location_context=example.location_context,
            business_types=example.business_types or "",
        )

        score = greenhouse_attribution_metric(example, prediction)
        scores.append(score)

    avg_score = sum(scores) / len(scores) if scores else 0.0

    metrics = {
        "average_score": avg_score,
        "num_examples": len(testset),
        "scores": scores,
    }

    return metrics


def display_evaluation_results(metrics: dict[str, float]) -> None:
    """Display evaluation results."""
    console.print("[bold]Evaluation Results[/bold]\n")

    # Summary table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Average Score", f"{metrics['average_score']:.3f}")
    table.add_row("Test Examples", str(metrics["num_examples"]))

    console.print(table)

    # Individual scores
    if "scores" in metrics:
        scores_str = [f"{s:.3f}" for s in metrics["scores"]]
        console.print(f"\n[cyan]Individual Scores:[/cyan] {scores_str}")


def test_on_business(
    module: GreenhouseAttributionModule,
    business_name: str,
    location: str,
) -> None:
    """Test optimized module on a specific business."""
    console.print(f"\n[yellow]Testing on:[/yellow] {business_name} ({location})\n")

    prediction = module(
        business_name=business_name,
        location_context=location,
        business_types="",
    )

    # Convert to Pydantic for display
    result = module.to_pydantic(prediction)

    # Display results
    console.print("[bold]Analysis Results:[/bold]\n")
    console.print(
        f"[cyan]Grow Light Likelihood:[/cyan] {result.grow_light_likelihood:.2f}"
    )
    console.print(
        f"[cyan]Lighting Evidence:[/cyan] {result.lighting_evidence or 'None'}"
    )
    console.print(
        f"[cyan]Primary Crops:[/cyan] {', '.join(result.primary_crops) or 'None'}"
    )
    console.print(f"[cyan]Size:[/cyan] {result.size_hectares or 'Unknown'} ha")
    console.print(f"[cyan]Instagram:[/cyan] {result.instagram_handle or 'Not found'}")
    console.print(f"[cyan]Sources:[/cyan]")
    for source in result.sources[:3]:
        console.print(f"  - {source}")

    if hasattr(prediction, "reasoning"):
        console.print(f"\n[cyan]Reasoning:[/cyan] {prediction.reasoning}")


def save_optimized_module(
    module: GreenhouseAttributionModule,
    output_path: Path,
) -> None:
    """Save optimized module to file."""
    console.print(f"\n[yellow]Saving optimized module to:[/yellow] {output_path}")

    try:
        # Save module state
        module.save(str(output_path))
        console.print("[green]✓[/green] Module saved successfully")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to save module: {e}", style="bold red")
        # Fallback: save as JSON with predictions
        try:
            output_data = {
                "module_type": "GreenhouseAttributionModule",
                "optimization_method": "BootstrapFewShot",
                "note": "Load with: module.load(path)",
            }
            output_path.write_text(json.dumps(output_data, indent=2))
            console.print(f"[yellow]Saved metadata to:[/yellow] {output_path}")
        except Exception as save_error:
            console.print(
                f"[red]✗[/red] Failed to save metadata: {save_error}",
                style="bold red",
            )


def main() -> int:
    """Run prompt optimization command."""
    args = parse_arguments()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Display header
    display_header()

    # Get API key
    api_key = args.api_key or os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        console.print(
            "[red]✗[/red] PERPLEXITY_API_KEY not set",
            style="bold red",
        )
        console.print("\n[yellow]Set it with:[/yellow]")
        console.print("  export PERPLEXITY_API_KEY='your-key'")
        console.print("  or use --api-key option")
        return 1

    # Create training examples
    console.print("[cyan]Loading training examples...[/cyan]")
    all_examples = create_training_examples()
    trainset = all_examples[: args.max_examples]
    console.print(f"[green]✓[/green] Loaded {len(trainset)} training examples\n")

    # Display examples
    display_training_examples(trainset)

    # Dry run mode
    if args.dry_run:
        console.print("\n[yellow]Dry run mode - stopping here[/yellow]")
        return 0

    # Configure DSPy
    try:
        configure_dspy(api_key, args.model)
    except Exception:
        return 1

    # Run optimization
    try:
        optimized_module = run_optimization(
            trainset=trainset,
            max_bootstrapped_demos=args.max_bootstrapped_demos,
            max_labeled_demos=args.max_labeled_demos,
        )
    except Exception as e:
        console.print(f"[red]Optimization failed:[/red] {e}")
        return 1

    # Evaluate on test set (use same examples for now)
    testset = all_examples
    metrics = evaluate_module(optimized_module, testset)
    display_evaluation_results(metrics)

    # Test on specific business if requested
    if args.test_business:
        if not args.test_location:
            console.print(
                "[red]✗[/red] --test-location required with --test-business",
                style="bold red",
            )
            return 1

        test_on_business(
            optimized_module,
            args.test_business,
            args.test_location,
        )

    # Save optimized module if requested
    if args.output:
        save_optimized_module(optimized_module, args.output)

    console.print("\n[bold green]Optimization complete![/bold green]")
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print("1. Review evaluation metrics above")
    console.print("2. Test on new businesses with --test-business")
    console.print("3. Save optimized module with --output")
    console.print("4. Integrate into attribution.py if results are good")

    return 0


if __name__ == "__main__":
    sys.exit(main())
