#!/usr/bin/env python3
"""Master orchestrator for iterative dataset refinement.

Automates the cycle of:
1. Run optimization
2. Analyze failures
3. Generate targeted company list
4. Collect data
5. Repeat

This enables systematic improvement toward the 95% F1 target.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class IterativeRefinementOrchestrator:
    """Orchestrate iterative refinement process."""

    def __init__(
        self,
        base_results_dir: Path,
        data_dir: Path,
        max_iterations: int = 5,
        target_f1: float = 0.95,
    ):
        """Initialize orchestrator.

        Args:
            base_results_dir: Base directory for storing iteration results
            data_dir: Directory containing research data
            max_iterations: Maximum number of iterations to run
            target_f1: Target F1 score to achieve

        """
        self.base_results_dir = base_results_dir
        self.data_dir = data_dir
        self.max_iterations = max_iterations
        self.target_f1 = target_f1
        self.iterations: list[dict[str, Any]] = []

    def run_optimization(self, iteration_num: int, example_count: int) -> dict[str, Any]:
        """Run GEPA optimization for this iteration.

        Args:
            iteration_num: Iteration number
            example_count: Number of examples in dataset

        Returns:
            Optimization results

        """
        results_dir = self.base_results_dir / f"iteration_{iteration_num}_{example_count}examples"
        results_dir.mkdir(parents=True, exist_ok=True)

        console.print(
            f"\n[bold cyan]Running Iteration {iteration_num} Optimization[/bold cyan]"
        )
        console.print(f"  Examples: {example_count}")
        console.print(f"  Results dir: {results_dir}")

        # Run optimization (assuming optimize_greenhouse_detector.py exists)
        cmd = [
            "uv",
            "run",
            "python",
            "notebooks/optimize_greenhouse_detector.py",
            "--output-dir",
            str(results_dir),
            "--num-examples",
            str(example_count),
            "--budget",
            "medium",
            "--num-repeats",
            "10",
            "--num-folds",
            "5",
        ]

        console.print(f"\n[dim]Command: {' '.join(cmd)}[/dim]")
        console.print(
            "[yellow]⏳[/yellow] This will take 8-10 hours. Running in foreground..."
        )

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            console.print("[green]✓[/green] Optimization complete")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]✗[/red] Optimization failed: {e}")
            console.print(f"[dim]{e.stderr}[/dim]")
            return None

        # Load results
        cv_file = results_dir / "cv_results.json"
        if not cv_file.exists():
            console.print(f"[red]✗[/red] CV results not found: {cv_file}")
            return None

        with open(cv_file) as f:
            cv_results = json.load(f)

        return {
            "iteration": iteration_num,
            "example_count": example_count,
            "results_dir": results_dir,
            "cv_results": cv_results,
        }

    def analyze_failures(self, iteration: dict[str, Any]) -> dict[str, Any]:
        """Analyze optimization failures for this iteration.

        Args:
            iteration: Iteration results

        Returns:
            Failure analysis

        """
        console.print(
            f"\n[bold cyan]Analyzing Iteration {iteration['iteration']} Failures[/bold cyan]"
        )

        results_dir = iteration["results_dir"]
        analysis_file = results_dir / "failure_analysis.json"

        # Run failure analysis
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/analyze_optimization_failures.py",
            "--results-dir",
            str(results_dir),
            "--data-dir",
            str(self.data_dir),
            "--output",
            str(analysis_file),
        ]

        try:
            subprocess.run(cmd, check=True)
            console.print("[green]✓[/green] Failure analysis complete")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]✗[/red] Failure analysis failed: {e}")
            return None

        # Load analysis
        with open(analysis_file) as f:
            return json.load(f)

    def generate_collection_plan(
        self, iteration: dict[str, Any], analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate targeted collection plan for next iteration.

        Args:
            iteration: Current iteration results
            analysis: Failure analysis results

        Returns:
            Collection plan

        """
        console.print(
            f"\n[bold cyan]Generating Collection Plan for Iteration {iteration['iteration'] + 1}[/bold cyan]"
        )

        results_dir = iteration["results_dir"]
        plan_file = results_dir / "collection_plan.json"

        # Run collection plan generator
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/generate_targeted_companies.py",
            "--analysis",
            str(results_dir / "failure_analysis.json"),
            "--output",
            str(plan_file),
            "--current-count",
            str(iteration["example_count"]),
        ]

        try:
            subprocess.run(cmd, check=True)
            console.print("[green]✓[/green] Collection plan generated")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]✗[/red] Plan generation failed: {e}")
            return None

        # Load plan
        with open(plan_file) as f:
            return json.load(f)

    def should_continue(self, iteration: dict[str, Any]) -> tuple[bool, str]:
        """Determine if another iteration is needed.

        Args:
            iteration: Current iteration results

        Returns:
            (should_continue, reason)

        """
        mean_f1 = iteration["cv_results"]["mean_f1"]
        iteration_num = iteration["iteration"]

        # Check if target reached
        if mean_f1 >= self.target_f1:
            return False, f"Target F1 ({self.target_f1:.1%}) achieved: {mean_f1:.1%}"

        # Check if max iterations reached
        if iteration_num >= self.max_iterations:
            return False, f"Max iterations ({self.max_iterations}) reached"

        # Check if progress is minimal
        if len(self.iterations) >= 2:
            prev_f1 = self.iterations[-2]["cv_results"]["mean_f1"]
            improvement = mean_f1 - prev_f1

            if improvement < 0.02:  # Less than 2% improvement
                return (
                    False,
                    f"Minimal improvement ({improvement:.2%}) - may need new approach",
                )

        # Continue iterating
        gap_to_target = self.target_f1 - mean_f1
        return True, f"Continue iterating - gap to target: {gap_to_target:.1%}"

    def run(self, start_iteration: int = 1) -> list[dict[str, Any]]:
        """Run iterative refinement process.

        Args:
            start_iteration: Starting iteration number

        Returns:
            List of iteration results

        """
        console.print(
            Panel.fit(
                "[bold cyan]Iterative Refinement Orchestrator[/bold cyan]\n"
                f"Target: {self.target_f1:.1%} F1 score\n"
                f"Max iterations: {self.max_iterations}",
                border_style="cyan",
            )
        )

        current_iteration = start_iteration

        # Count initial examples
        example_count = len(list(self.data_dir.glob("*.json")))
        console.print(f"\n[bold]Starting dataset:[/bold] {example_count} examples")

        while current_iteration <= self.max_iterations:
            console.print(f"\n{'='*70}")
            console.print(f"ITERATION {current_iteration}")
            console.print(f"{'='*70}")

            # Step 1: Run optimization
            iteration_result = self.run_optimization(current_iteration, example_count)
            if iteration_result is None:
                console.print("[red]✗[/red] Optimization failed, stopping")
                break

            self.iterations.append(iteration_result)

            # Display current performance
            mean_f1 = iteration_result["cv_results"]["mean_f1"]
            std_f1 = iteration_result["cv_results"]["std_f1"]
            console.print(
                f"\n[bold]Iteration {current_iteration} Results:[/bold]"
                f"\n  Mean F1: {mean_f1:.2%}"
                f"\n  Std F1: {std_f1:.2%}"
                f"\n  Examples: {example_count}"
            )

            # Check if should continue
            should_continue, reason = self.should_continue(iteration_result)
            console.print(f"\n[bold]Decision:[/bold] {reason}")

            if not should_continue:
                break

            # Step 2: Analyze failures
            analysis = self.analyze_failures(iteration_result)
            if analysis is None:
                console.print("[yellow]⚠[/yellow] Continuing despite analysis failure")

            # Step 3: Generate collection plan
            collection_plan = self.generate_collection_plan(iteration_result, analysis)
            if collection_plan is None:
                console.print("[red]✗[/red] Cannot generate collection plan, stopping")
                break

            # Step 4: Prompt user for data collection
            target_count = collection_plan["target_count"]
            console.print(
                f"\n[bold yellow]⏸️  MANUAL STEP REQUIRED[/bold yellow]"
                f"\n  Next iteration needs {target_count} more examples"
                f"\n  Collection plan: {iteration_result['results_dir'] / 'collection_plan.json'}"
                f"\n\n  Please:"
                f"\n    1. Review collection plan"
                f"\n    2. Create data/companies_to_research_iteration{current_iteration + 1}.json"
                f"\n    3. Run batch research"
                f"\n    4. Verify total examples: {example_count + target_count}"
                f"\n    5. Press Enter to continue..."
            )

            input()  # Wait for user

            # Update example count for next iteration
            example_count = len(list(self.data_dir.glob("*.json")))
            console.print(f"\n[green]✓[/green] Dataset updated: {example_count} examples")

            current_iteration += 1

        # Final comparison
        console.print(f"\n{'='*70}")
        console.print("FINAL RESULTS")
        console.print(f"{'='*70}")

        if len(self.iterations) > 1:
            # Run comparison
            iteration_dirs = [it["results_dir"] for it in self.iterations]
            comparison_file = self.base_results_dir / "final_comparison.md"

            cmd = [
                "uv",
                "run",
                "python",
                "scripts/compare_iterations.py",
                *[str(d) for d in iteration_dirs],
                "--output",
                str(comparison_file),
            ]

            try:
                subprocess.run(cmd, check=True)
                console.print(
                    f"[green]✓[/green] Final comparison: {comparison_file}"
                )
            except subprocess.CalledProcessError as e:
                console.print(f"[yellow]⚠[/yellow] Comparison failed: {e}")

        return self.iterations


def main() -> int:
    """Run iterative refinement orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Orchestrate iterative dataset refinement"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Base directory for iteration results",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/research"),
        help="Directory containing research data",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "--target-f1",
        type=float,
        default=0.95,
        help="Target F1 score",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=1,
        help="Starting iteration number",
    )

    args = parser.parse_args()

    orchestrator = IterativeRefinementOrchestrator(
        base_results_dir=args.results_dir,
        data_dir=args.data_dir,
        max_iterations=args.max_iterations,
        target_f1=args.target_f1,
    )

    iterations = orchestrator.run(start_iteration=args.start_iteration)

    console.print(
        f"\n[bold green]Completed {len(iterations)} iterations[/bold green]"
    )

    # Display final summary
    if iterations:
        final = iterations[-1]
        console.print(
            f"\n[bold]Final Performance:[/bold]"
            f"\n  Mean F1: {final['cv_results']['mean_f1']:.2%}"
            f"\n  Examples: {final['example_count']}"
            f"\n  Iterations: {len(iterations)}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
