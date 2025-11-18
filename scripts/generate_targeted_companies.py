#!/usr/bin/env python3
"""Generate targeted company list for next iteration based on failure analysis.

This script uses the failure analysis to create a focused list of companies
to research, prioritizing patterns that will most improve model performance.
"""

import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Target distributions for balanced dataset
TARGET_CROP_DISTRIBUTION = {
    "Roses": 0.15,  # 15% roses
    "Orchids": 0.15,  # 15% orchids
    "Tomatoes": 0.15,  # 15% tomatoes
    "Peppers": 0.10,  # 10% peppers
    "Gerbera": 0.08,  # 8% gerbera
    "Chrysanthemums": 0.08,  # 8% chrysanthemums
    "Cucumbers": 0.05,  # 5% cucumbers
    "Strawberries": 0.05,  # 5% strawberries
    "Other": 0.19,  # 19% other crops
}

TARGET_LOCATION_DISTRIBUTION = {
    "Westland": 0.40,  # 40% Westland (main region)
    "Aalsmeer": 0.15,  # 15% Aalsmeer
    "Noord-Brabant": 0.15,  # 15% Noord-Brabant
    "Zuid-Holland (other)": 0.15,  # 15% other Zuid-Holland
    "Other regions": 0.15,  # 15% other regions
}


def load_failure_analysis(analysis_file: Path) -> dict[str, Any]:
    """Load failure analysis results."""
    if not analysis_file.exists():
        console.print(f"[red]✗[/red] Analysis file not found: {analysis_file}")
        console.print(
            "[yellow]Tip:[/yellow] Run analyze_optimization_failures.py first"
        )
        sys.exit(1)

    with open(analysis_file) as f:
        return json.load(f)


def calculate_collection_target(
    current_count: int, mean_f1: float
) -> tuple[int, str]:
    """Calculate how many more examples to collect based on performance."""
    if mean_f1 >= 0.85:
        # Excellent - just polish with edge cases
        target = min(15, max(10, int(current_count * 0.15)))
        rationale = "Polishing edge cases"
    elif mean_f1 >= 0.80:
        # Very good - moderate expansion
        target = min(30, max(20, int(current_count * 0.35)))
        rationale = "Moderate expansion for 85%+ target"
    elif mean_f1 >= 0.75:
        # Good - significant expansion
        target = min(50, max(35, int(current_count * 0.55)))
        rationale = "Significant expansion for 85%+ target"
    elif mean_f1 >= 0.70:
        # Fair - major expansion
        target = min(70, max(50, int(current_count * 0.75)))
        rationale = "Major expansion for 80%+ target"
    else:
        # Poor - investigate issues first
        target = min(40, max(25, int(current_count * 0.40)))
        rationale = "Conservative expansion to debug issues"

    return target, rationale


def generate_crop_targets(
    underrepresented: list[str], target_count: int
) -> dict[str, int]:
    """Generate crop type targets based on ideal distribution."""
    crop_targets = {}

    # Prioritize underrepresented crops
    priority_crops = underrepresented[:5]

    # Distribute target count according to ideal distribution
    for crop, proportion in TARGET_CROP_DISTRIBUTION.items():
        if crop in priority_crops:
            # Boost underrepresented crops
            crop_targets[crop] = max(5, int(target_count * proportion * 1.5))
        else:
            crop_targets[crop] = int(target_count * proportion)

    return crop_targets


def generate_location_targets(
    underrepresented: list[str], target_count: int
) -> dict[str, int]:
    """Generate location targets based on ideal distribution."""
    location_targets = {}

    # Prioritize underrepresented locations
    priority_locs = underrepresented[:3]

    # Distribute target count according to ideal distribution
    for location, proportion in TARGET_LOCATION_DISTRIBUTION.items():
        if location in priority_locs:
            # Boost underrepresented locations
            location_targets[location] = max(3, int(target_count * proportion * 1.3))
        else:
            location_targets[location] = int(target_count * proportion)

    return location_targets


def generate_quality_criteria(
    misclassifications: dict[str, int]
) -> list[dict[str, str]]:
    """Generate quality criteria based on misclassification patterns."""
    criteria = []

    # Handle no-evidence cases
    if misclassifications.get("no_evidence", 0) > 10:
        criteria.append(
            {
                "criterion": "Strong online presence",
                "description": "Company website, trade media articles, or supplier case studies",
                "priority": "HIGH",
            }
        )

    # Handle contradictory evidence
    if misclassifications.get("contradictory", 0) > 5:
        criteria.append(
            {
                "criterion": "Clear lighting signal",
                "description": "Explicit positive OR negative evidence, avoid mixed facilities",
                "priority": "HIGH",
            }
        )

    # Handle weak classifications
    weak_count = misclassifications.get("weak_positive", 0) + misclassifications.get(
        "weak_negative", 0
    )
    if weak_count > 10:
        criteria.append(
            {
                "criterion": "High confidence indicators",
                "description": "Supplier partnerships, trade media features, or explicit statements",
                "priority": "MEDIUM",
            }
        )

    # Always include diversity
    criteria.append(
        {
            "criterion": "Pattern diversity",
            "description": "Target underrepresented crop types, locations, and company sizes",
            "priority": "MEDIUM",
        }
    )

    return criteria


def generate_search_strategies(
    crop_targets: dict[str, int], location_targets: dict[str, int]
) -> list[dict[str, Any]]:
    """Generate search strategies for finding targeted companies."""
    strategies = []

    # Strategy 1: Crop-specific searches
    for crop, target in sorted(
        crop_targets.items(), key=lambda x: x[1], reverse=True
    )[:3]:
        if target > 0:
            strategies.append(
                {
                    "strategy": f"Search for {crop} growers in Netherlands",
                    "keywords": [
                        f"{crop.lower()} kwekerij nederland",
                        f"{crop.lower()} teelt glastuinbouw",
                        f"{crop.lower()} led belichting",
                    ],
                    "target_count": target,
                    "expected_classification": "Mixed (research needed)",
                }
            )

    # Strategy 2: Supplier customer lists
    strategies.append(
        {
            "strategy": "Search Signify/Philips LED customer references",
            "keywords": [
                "signify greenpower case studies nederland",
                "philips led horticulture customers",
                "signify horti partners nederland",
            ],
            "target_count": max(10, int(sum(crop_targets.values()) * 0.2)),
            "expected_classification": "POSITIVE",
        }
    )

    # Strategy 3: Trade media databases
    strategies.append(
        {
            "strategy": "Search trade media (GroentenNieuws, OnderGlas, FloralDaily)",
            "keywords": [
                "site:groentennieuws.nl assimilatiebelichting",
                "site:onderglas.nl led-belichting",
                "site:floraldaily.com dutch led lighting",
            ],
            "target_count": max(8, int(sum(crop_targets.values()) * 0.15)),
            "expected_classification": "Mixed (research needed)",
        }
    )

    # Strategy 4: Regional directories
    for location, target in sorted(
        location_targets.items(), key=lambda x: x[1], reverse=True
    )[:2]:
        if target > 0:
            strategies.append(
                {
                    "strategy": f"Search {location} greenhouse directories",
                    "keywords": [
                        f"glastuinbouw {location.lower()}",
                        f"kwekerij {location.lower()} lijst",
                    ],
                    "target_count": target,
                    "expected_classification": "Mixed (research needed)",
                }
            )

    return strategies


def display_targets(
    target_count: int,
    rationale: str,
    crop_targets: dict[str, int],
    location_targets: dict[str, int],
    quality_criteria: list[dict[str, str]],
    search_strategies: list[dict[str, Any]],
) -> None:
    """Display targeted collection plan with Rich formatting."""
    console.print(
        Panel.fit(
            "[bold cyan]Targeted Data Collection Plan[/bold cyan]\n"
            f"Next iteration: Collect {target_count} targeted examples",
            border_style="cyan",
        )
    )

    # Collection rationale
    console.print(f"\n[bold]🎯 Collection Target:[/bold] {target_count} companies")
    console.print(f"  [dim]Rationale: {rationale}[/dim]")

    # Crop distribution
    console.print("\n[bold]🌱 Crop Type Targets[/bold]")
    crop_table = Table(show_header=True, header_style="bold magenta")
    crop_table.add_column("Crop Type", style="cyan")
    crop_table.add_column("Target Count", style="white")
    crop_table.add_column("% of Total", style="dim")

    for crop, count in sorted(crop_targets.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = count / target_count * 100
            crop_table.add_row(crop, str(count), f"{pct:.1f}%")

    console.print(crop_table)

    # Location distribution
    console.print("\n[bold]📍 Location Targets[/bold]")
    loc_table = Table(show_header=True, header_style="bold magenta")
    loc_table.add_column("Location", style="cyan")
    loc_table.add_column("Target Count", style="white")
    loc_table.add_column("% of Total", style="dim")

    for location, count in sorted(
        location_targets.items(), key=lambda x: x[1], reverse=True
    ):
        if count > 0:
            pct = count / target_count * 100
            loc_table.add_row(location, str(count), f"{pct:.1f}%")

    console.print(loc_table)

    # Quality criteria
    console.print("\n[bold]✅ Quality Criteria[/bold]")
    for criterion in quality_criteria:
        priority = criterion["priority"]
        color = "red" if priority == "HIGH" else "yellow"
        console.print(
            f"  [{color}]{priority}[/{color}] - {criterion['criterion']}"
        )
        console.print(f"    [dim]{criterion['description']}[/dim]")

    # Search strategies
    console.print("\n[bold]🔍 Search Strategies[/bold]")
    for i, strategy in enumerate(search_strategies, 1):
        console.print(f"\n  {i}. [cyan]{strategy['strategy']}[/cyan]")
        console.print(f"     Target: {strategy['target_count']} companies")
        console.print(f"     Expected: {strategy['expected_classification']}")
        console.print("     Keywords:")
        for keyword in strategy["keywords"]:
            console.print(f"       - {keyword}")


def save_collection_plan(
    output_file: Path,
    target_count: int,
    rationale: str,
    crop_targets: dict[str, int],
    location_targets: dict[str, int],
    quality_criteria: list[dict[str, str]],
    search_strategies: list[dict[str, Any]],
) -> None:
    """Save collection plan to JSON file."""
    plan = {
        "target_count": target_count,
        "rationale": rationale,
        "crop_targets": crop_targets,
        "location_targets": location_targets,
        "quality_criteria": quality_criteria,
        "search_strategies": search_strategies,
        "instructions": [
            "Use search strategies to identify companies",
            "Prioritize companies meeting quality criteria",
            "Target distribution according to crop/location targets",
            "Verify all companies before research (avoid duplicates)",
            "Run batch_company_research.py with new company list",
        ],
    }

    with open(output_file, "w") as f:
        json.dump(plan, f, indent=2)

    console.print(f"\n[green]✓[/green] Collection plan saved to {output_file}")


def main() -> int:
    """Generate targeted company collection plan."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate targeted company list for next iteration"
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        required=True,
        help="Failure analysis JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/collection_plan_iteration2.json"),
        help="Output collection plan file",
    )
    parser.add_argument(
        "--current-count",
        type=int,
        default=65,
        help="Current number of examples in dataset",
    )

    args = parser.parse_args()

    # Load failure analysis
    console.print("[dim]Loading failure analysis...[/dim]")
    analysis = load_failure_analysis(args.analysis)

    # Calculate collection target
    mean_f1 = analysis["cv_results"]["mean_f1"]
    target_count, rationale = calculate_collection_target(args.current_count, mean_f1)

    # Generate targets
    console.print("[dim]Generating collection targets...[/dim]")
    underrep_crops = analysis.get("underrepresented", {}).get("crop_types", [])
    underrep_locs = analysis.get("underrepresented", {}).get("locations", [])

    crop_targets = generate_crop_targets(underrep_crops, target_count)
    location_targets = generate_location_targets(underrep_locs, target_count)
    quality_criteria = generate_quality_criteria(analysis.get("misclassifications", {}))
    search_strategies = generate_search_strategies(crop_targets, location_targets)

    # Display plan
    display_targets(
        target_count,
        rationale,
        crop_targets,
        location_targets,
        quality_criteria,
        search_strategies,
    )

    # Save plan
    save_collection_plan(
        args.output,
        target_count,
        rationale,
        crop_targets,
        location_targets,
        quality_criteria,
        search_strategies,
    )

    console.print(
        "\n[bold green]Next Steps:[/bold green]"
        "\n  1. Use search strategies to identify companies"
        "\n  2. Create data/companies_to_research_iteration2.json"
        "\n  3. Run: uv run python -m sevenrad_ee.operations.batch_company_research"
        f"\n  4. Verify {args.current_count + target_count} total examples"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
