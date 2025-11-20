"""
Dry run cost estimation for GEPA optimization with PerplexityLM.

This script runs a minimal GEPA optimization to estimate the total cost
of the full Phase 3 optimization run before committing resources.

Uses PerplexityLM for student model (native structured outputs) and
Gemini for teacher model (GEPA reflection).

Documentation Type: Script (Code to Run)
Part of: Phase 3 - GEPA Optimization Strategy
"""

import argparse
import json
import logging
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    import dspy
    from dspy.teleprompt import GEPA
except ImportError as e:
    msg = "dspy-ai package is required. Install with: uv pip install -e '.[dev]'"
    raise ImportError(msg) from e

from sklearn.model_selection import train_test_split

from sevenrad_ee.ai.dspy_evaluation import gepa_compatible_metric
from sevenrad_ee.ai.dspy_greenhouse import GreenhouseDetector
from sevenrad_ee.ai.dspy_perplexity import PerplexityLM
from sevenrad_ee.ai.retrievers import CachedRetriever

console = Console()
logger = logging.getLogger(__name__)

# Random seed for reproducibility
SEED = 42


class CostTracker:
    """
    Track API calls and token counts for cost estimation.

    Wraps a DSPy LM to count all API interactions during optimization.
    """

    def __init__(self, lm: Any) -> None:  # noqa: ANN401
        """
        Initialize cost tracker.

        Args:
            lm: DSPy language model to wrap

        """
        self.lm = lm
        self.api_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_log: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Delegate attribute access to underlying LM."""
        return getattr(self.lm, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """
        Execute LM call and track costs.

        Args:
            *args: Positional arguments for LM
            **kwargs: Keyword arguments for LM

        Returns:
            LM response

        """
        # Track the call
        self.api_calls += 1

        # Extract prompt for token counting
        prompt = kwargs.get("messages", "") or kwargs.get("prompt", "")
        if isinstance(prompt, list):
            # For chat-style messages
            prompt_text = " ".join(
                msg.get("content", "") if isinstance(msg, dict) else str(msg)
                for msg in prompt
            )
        else:
            prompt_text = str(prompt)

        # Crude token estimation (actual tokenizer would be better)
        # Rough estimate: 1 token ≈ 4 characters
        input_tokens = len(prompt_text) // 4

        # Make the actual call
        response = self.lm(*args, **kwargs)

        # Estimate output tokens
        response_text = str(response)
        output_tokens = len(response_text) // 4

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        self.call_log.append(
            {
                "call_number": self.api_calls,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cumulative_input": self.total_input_tokens,
                "cumulative_output": self.total_output_tokens,
            }
        )

        console.print(
            f"[cyan]API Call #{self.api_calls}[/cyan]: "
            f"{input_tokens:,} in / {output_tokens:,} out "
            f"(Total: {self.total_input_tokens:,} / {self.total_output_tokens:,})"
        )

        return response

    def estimate_cost(
        self,
        input_price_per_1k: float = 0.0025,
        output_price_per_1k: float = 0.01,
    ) -> float:
        """
        Estimate total cost based on token counts.

        Default pricing is for Gemini 2.5 Pro (teacher/reflection model):
        - Input: $0.0025 per 1K tokens
        - Output: $0.01 per 1K tokens

        Note: This only tracks the teacher model (reflection_lm) costs.
        Student model (PerplexityLM) costs are not included.

        Args:
            input_price_per_1k: Price per 1000 input tokens
            output_price_per_1k: Price per 1000 output tokens

        Returns:
            Estimated cost in dollars

        """
        input_cost = (self.total_input_tokens / 1000) * input_price_per_1k
        output_cost = (self.total_output_tokens / 1000) * output_price_per_1k
        return input_cost + output_cost


def load_greenhouse_data(cache_dir: Path) -> list[dspy.Example]:
    """
    Load cached greenhouse data from JSON files.

    Args:
        cache_dir: Directory containing research JSON files

    Returns:
        List of dspy.Example objects with company data

    """
    examples = []
    json_files = list(cache_dir.glob("*.json"))

    console.print(f"[cyan]Loading data from {len(json_files)} JSON files...[/cyan]")

    for json_file in json_files:
        try:
            data = json.loads(json_file.read_text())

            # Extract fields from JSON
            company_name = data.get("company", "")
            location = data.get("location", "")
            is_greenhouse = data.get("is_greenhouse", False)
            uses_growlight = data.get("uses_growlight", "UNKNOWN")

            # Create dspy.Example (using parameter names that match GreenhouseDetector.forward())
            example = dspy.Example(
                location_name=company_name,
                location_area=location,
                is_greenhouse=is_greenhouse,
                uses_growlight=uses_growlight,
            ).with_inputs("location_name", "location_area")

            examples.append(example)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load {json_file}: {e}")
            continue

    console.print(f"[green]✓[/green] Loaded {len(examples)} examples")
    return examples


def run_dry_run(
    cache_dir: Path,
    dry_run_samples: int = 10,
) -> dict[str, Any]:
    """
    Run minimal GEPA optimization to estimate costs.

    Args:
        cache_dir: Directory with cached research data
        dry_run_samples: Number of training samples to use

    Returns:
        Dictionary with cost estimates and recommendations

    """
    # Set random seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

    # Load data
    all_data = load_greenhouse_data(cache_dir)

    if len(all_data) == 0:
        msg = f"No data found in {cache_dir}"
        raise ValueError(msg)

    # Take small subset for dry run
    sample_data = random.sample(all_data, min(dry_run_samples, len(all_data)))

    # Split data
    trainset, valset = train_test_split(
        sample_data,
        test_size=0.3,
        random_state=SEED,
        stratify=[ex.is_greenhouse for ex in sample_data],
    )

    console.print(
        f"[cyan]Dry run with {len(trainset)} train, "
        f"{len(valset)} val examples[/cyan]"
    )

    # Create retriever and detector
    cached_retriever = CachedRetriever(cache_dir=cache_dir)
    detector = GreenhouseDetector(retriever=cached_retriever)

    # Configure student model (Perplexity Sonar) for predictions
    # Using PerplexityLM for native structured outputs (bypasses LiteLLM)
    student_lm = PerplexityLM(
        model="llama-3.1-sonar-large-128k-chat",
        temperature=0,
    )
    dspy.configure(lm=student_lm)

    # Configure teacher model with cost tracking for GEPA feedback
    teacher_lm = dspy.LM("gemini/gemini-2.5-pro", temperature=0)
    cost_tracker = CostTracker(teacher_lm)

    # Configure GEPA (minimal settings for dry run)
    console.print("[yellow]Starting dry run optimization...[/yellow]")

    optimizer = GEPA(
        metric=gepa_compatible_metric,
        auto="light",  # Use 'light' preset for fast, minimal dry run
        reflection_lm=cost_tracker,  # Teacher model for feedback-based optimization
        seed=SEED,
    )

    # Run optimization
    try:
        _optimized = optimizer.compile(
            detector,
            trainset=trainset,
            valset=valset,
        )
    except Exception as e:
        logger.warning(f"Dry run failed: {e}")
        console.print(f"[red]Dry run failed: {e}[/red]")

    # --- Cost Extrapolation ---
    dry_run_cost = cost_tracker.estimate_cost()

    # The full run will use 'auto=medium', which is 2x the work of 'auto=light'.
    WORK_SCALE_FACTOR = 2.0

    # The full run uses the entire dataset, so costs scale with data size.
    full_dataset_size = len(all_data)
    if dry_run_samples > 0:
        DATA_SCALE_FACTOR = full_dataset_size / dry_run_samples
    else:
        DATA_SCALE_FACTOR = 1.0

    # Estimate full cost based on scaling factors.
    # Note: This only tracks reflection_lm costs. The true cost, including
    # evaluation calls, will be higher. This is a lower-bound estimate.
    estimated_full_cost = dry_run_cost * WORK_SCALE_FACTOR * DATA_SCALE_FACTOR
    estimated_calls = int(
        cost_tracker.api_calls * WORK_SCALE_FACTOR * DATA_SCALE_FACTOR
    )

    return {
        "dry_run_cost": dry_run_cost,
        "dry_run_calls": cost_tracker.api_calls,
        "dry_run_input_tokens": cost_tracker.total_input_tokens,
        "dry_run_output_tokens": cost_tracker.total_output_tokens,
        "estimated_full_cost": estimated_full_cost,
        "estimated_calls": estimated_calls,
        "budget_target": 25.0,
        "within_budget": estimated_full_cost <= 25.0,
        "call_log": cost_tracker.call_log,
    }


def display_results(results: dict[str, Any]) -> None:
    """
    Display cost estimation results in formatted table.

    Args:
        results: Dictionary with cost estimates

    """
    # Dry run stats table
    table = Table(title="Dry Run Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("API Calls", f"{results['dry_run_calls']:,}")
    table.add_row("Input Tokens", f"{results['dry_run_input_tokens']:,}")
    table.add_row("Output Tokens", f"{results['dry_run_output_tokens']:,}")
    table.add_row("Dry Run Cost", f"${results['dry_run_cost']:.3f}")

    console.print(table)

    # Full run estimate table
    table2 = Table(title="Full Run Estimate (Lower Bound)", show_header=True, header_style="bold cyan")
    table2.add_column("Metric", style="cyan")
    table2.add_column("Value", justify="right")

    table2.add_row("Estimated API Calls", f"{results['estimated_calls']:,}")
    table2.add_row("Estimated Cost", f"${results['estimated_full_cost']:.2f}")
    table2.add_row("Budget Target", f"${results['budget_target']:.2f}")

    status_color = "green" if results["within_budget"] else "red"
    status = "✓ Within Budget" if results["within_budget"] else "✗ Over Budget"
    table2.add_row("Status", f"[{status_color}]{status}[/{status_color}]")

    console.print(table2)
    console.print(
        "\n[italic yellow]Note: Cost estimate is a lower bound as it only tracks "
        "the teacher model (Gemini).[/italic yellow]\n"
        "[italic yellow]Student model (PerplexityLM) costs are not included "
        "in this estimate.[/italic yellow]"
    )


    # Recommendations
    if not results["within_budget"]:
        console.print(
            "\n[yellow]⚠️  Estimated cost exceeds budget![/yellow]\n"
            "[bold]Recommended adjustments:[/bold]\n"
            "1. For the full run, consider staying with auto='light'.\n"
            "2. Explicitly set max_metric_calls to a lower value to cap evaluations.\n"
            "3. Consider using a cheaper reflection_lm model.\n"
        )
    else:
        console.print(
            "\n[green]✓ Estimated cost is within budget![/green]\n"
            "[bold]Proceed with full optimization run using auto='medium'.[/bold]"
        )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments

    """
    parser = argparse.ArgumentParser(
        description="Estimate costs for GEPA optimization dry run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run dry run with defaults
  uv run python scripts/dry_run_cost_estimate.py --cache-dir data/research

  # Run with a larger sample size for a more accurate estimate
  uv run python scripts/dry_run_cost_estimate.py \\
    --cache-dir data/research \\
    --samples 20 \\
    --output results/dry_run_report.json
        """,
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory containing cached research JSON files",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of training samples to use for the dry run (default: 10)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save cost report (JSON format)",
    )

    return parser.parse_args()


def main() -> int:
    """
    Run dry run cost estimation.

    Returns:
        Exit code (0 for success, 1 for failure)

    """
    # Display header
    console.print(
        Panel.fit(
            "[bold cyan]GEPA Optimization - Dry Run Cost Estimation[/bold cyan]\n"
            "Estimate costs before running full Phase 3 optimization\n"
            "[dim]Student: PerplexityLM | Teacher: Gemini 2.5 Pro[/dim]",
            border_style="cyan",
        )
    )

    args = parse_arguments()

    # Verify cache directory exists
    if not args.cache_dir.exists():
        console.print(
            f"[red]✗[/red] Cache directory not found: {args.cache_dir}",
            style="bold red",
        )
        console.print(
            "\n[yellow]Tip:[/yellow] Ensure Phase 1 architecture is complete "
            "and data is cached in specified directory"
        )
        return 1

    # Run dry run
    try:
        results = run_dry_run(
            cache_dir=args.cache_dir,
            dry_run_samples=args.samples,
        )
    except Exception as e:
        console.print(f"[red]✗[/red] Dry run failed: {e}", style="bold red")
        logger.exception("Dry run failed")
        return 1

    # Display results
    display_results(results)

    # Save report if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        console.print(f"\n[green]✓[/green] Report saved to {args.output}")

    return 0 if results["within_budget"] else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
