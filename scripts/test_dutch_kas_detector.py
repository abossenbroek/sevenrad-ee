"""
Test script for Dutch PerplexityKasClassificatie signature.

This script validates the Dutch-only approach before full optimization.
Tests known failure cases from the first MIPROv2 run.

Gate Criteria:
- Dutch COT is happening (Dutch in responses > 50%)
- Crop identification rate > 80%
- uses_growlight accuracy improvement (baseline: 53.8%)

Documentation Type: Script (Code to Run)
Part of: Phase 4 - Dutch Signature Validation
"""

import argparse
import os

import dspy
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sevenrad_ee.ai.dspy_greenhouse import (
    GroeilichtGebruik,
    PerplexityKasDetector,
)
from sevenrad_ee.ai.dspy_perplexity import PerplexityLM

# Load environment variables
load_dotenv()

console = Console()

# Gate thresholds
THRESHOLD_KAS_ACCURACY = 80
THRESHOLD_GROWLIGHT_BASELINE = 53.8
THRESHOLD_DUTCH_COT = 50
THRESHOLD_CROP_ID = 80

# Test cases with expected values
# Based on retro analysis and domain knowledge
TEST_CASES = [
    {
        "bedrijfsnaam": "Van den Bos Premium Ardisia's",
        "locatie": "'s-Gravenzande",
        "expected_is_kas": True,
        "expected_gebruikt_groeilicht": "JA",  # Ardisia is often lit for winter sales
        "note": "Known failure case - model returned UNKNOWN (should infer from crop)",
    },
    {
        "bedrijfsnaam": "Nunhems Netherlands BV",
        "locatie": "'s-Gravenzande",
        "expected_is_kas": True,  # Has 2.5ha veredelingskassen (breeding greenhouses)
        "expected_gebruikt_groeilicht": "JA",  # Breeding greenhouses typically lit
        "note": "Known failure - model thought 'seed company = not greenhouse'",
    },
    {
        "bedrijfsnaam": "Marjoland",
        "locatie": "Waddinxveen",
        "expected_is_kas": True,
        "expected_gebruikt_groeilicht": "JA",  # Year-round tomato production
        "note": "Large tomato grower - should be easy case",
    },
    {
        "bedrijfsnaam": "Porta Nova",
        "locatie": "Waddinxveen",
        "expected_is_kas": True,
        "expected_gebruikt_groeilicht": "JA",  # Roses are always lit
        "note": "Rose grower - clear case for lighting",
    },
    {
        "bedrijfsnaam": "FloraHolland",
        "locatie": "Aalsmeer",
        "expected_is_kas": False,  # Auction house, not grower
        "expected_gebruikt_groeilicht": "NEE",
        "note": "Negative case - auction house, not a greenhouse",
    },
]

# Dutch keywords to check for COT
DUTCH_KEYWORDS = [
    "kas",
    "kwekerij",
    "glastuinbouw",
    "belichting",
    "gewas",
    "teelt",
    "teler",
    "jaarrond",
    "seizoen",
]

# Crop keywords to check for identification
CROP_KEYWORDS = [
    "rozen",
    "tomaten",
    "gerbera",
    "orchidee",
    "paprika",
    "komkommer",
    "ardisia",
    "phalaenopsis",
    "roos",
    "roses",
]


def run_test(dry_run: bool = False) -> None:
    """Run the Dutch signature test."""
    console.print(
        Panel.fit(
            "[bold cyan]Dutch PerplexityKasDetector Test[/bold cyan]\n"
            "Testing Phase 4 improvements with known failure cases",
            border_style="cyan",
        )
    )

    if dry_run:
        _show_dry_run_info()
        return

    # Initialize PerplexityLM
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        console.print("[red]✗ PERPLEXITY_API_KEY not set[/red]")
        return

    console.print("[green]✓[/green] Initializing PerplexityLM with sonar-pro...")

    lm = PerplexityLM(
        model="sonar-pro",
        api_key=api_key,
        temperature=0.1,  # Low temperature for more consistent results
    )
    dspy.settings.configure(lm=lm)

    # Create detector
    detector = PerplexityKasDetector()

    # Track results
    results = []
    counters = _run_test_cases(detector, results)

    # Summary table
    _show_summary(counters, results)


def _show_dry_run_info() -> None:
    """Show dry run information."""
    console.print("[yellow]DRY RUN MODE - No API calls will be made[/yellow]\n")
    console.print("Would test the following cases:")
    for i, case in enumerate(TEST_CASES, 1):
        console.print(f"  {i}. {case['bedrijfsnaam']} ({case['locatie']})")
        console.print(
            f"     Expected: is_kas={case['expected_is_kas']}, "
            f"gebruikt_groeilicht={case['expected_gebruikt_groeilicht']}"
        )
        console.print(f"     Note: {case['note']}\n")


def _run_test_cases(
    detector: PerplexityKasDetector,
    results: list[dict],
) -> dict[str, int]:
    """Run all test cases and return counters."""
    counters = {
        "dutch_count": 0,
        "crop_identified_count": 0,
        "correct_kas": 0,
        "correct_growlight": 0,
    }

    console.print(f"\n[bold]Running {len(TEST_CASES)} test cases...[/bold]\n")

    for case in TEST_CASES:
        console.print(f"[cyan]Testing: {case['bedrijfsnaam']}[/cyan]")
        _process_single_case(detector, case, results, counters)

    return counters


def _parse_is_kas(value: str) -> bool:
    """Parse is_kas value to boolean."""
    return str(value).strip().lower() in ("true", "ja", "yes", "1")


def _process_single_case(
    detector: PerplexityKasDetector,
    case: dict,
    results: list[dict],
    counters: dict[str, int],
) -> None:
    """Process a single test case."""
    try:
        # Call the Dutch detector (returns Dutch field names)
        result = detector(
            bedrijfsnaam=case["bedrijfsnaam"],
            locatie=case["locatie"],
        )

        # Parse response - result has Dutch fields: is_kas, gebruikt_groeilicht, etc.
        pred_is_kas = _parse_is_kas(result.is_kas)
        pred_growlight_nl = result.gebruikt_groeilicht

        # Convert to English for comparison
        pred_growlight_enum = GroeilichtGebruik.from_string(pred_growlight_nl)
        pred_growlight = pred_growlight_enum.to_english().value

        # Check if reasoning contains Dutch
        reasoning = result.redenering or ""
        has_dutch = any(kw in reasoning.lower() for kw in DUTCH_KEYWORDS)
        if has_dutch:
            counters["dutch_count"] += 1

        # Check if crop was identified (from hoofdgewas or reasoning)
        hoofdgewas = result.hoofdgewas or ""
        has_crop = any(kw in hoofdgewas.lower() for kw in CROP_KEYWORDS) or any(
            kw in reasoning.lower() for kw in CROP_KEYWORDS
        )
        if has_crop:
            counters["crop_identified_count"] += 1

        # Calculate correctness
        kas_correct = pred_is_kas == case["expected_is_kas"]
        if kas_correct:
            counters["correct_kas"] += 1

        # Map expected Dutch value to English for comparison
        expected_growlight_nl = case["expected_gebruikt_groeilicht"]
        expected_growlight = GroeilichtGebruik.from_string(expected_growlight_nl)
        expected_en = expected_growlight.to_english().value

        growlight_correct = pred_growlight == expected_en
        if growlight_correct:
            counters["correct_growlight"] += 1

        # Store result
        results.append(
            {
                "bedrijfsnaam": case["bedrijfsnaam"],
                "pred_is_kas": pred_is_kas,
                "expected_is_kas": case["expected_is_kas"],
                "kas_correct": kas_correct,
                "pred_growlight": pred_growlight,
                "expected_growlight": expected_en,
                "growlight_correct": growlight_correct,
                "has_dutch": has_dutch,
                "has_crop": has_crop,
                "reasoning": reasoning[:200],  # Truncate for display
            }
        )

        _print_case_result(
            case,
            pred_is_kas,
            pred_growlight,
            expected_en,
            kas_correct,
            growlight_correct,
            has_dutch,
            has_crop,
        )

    except Exception as e:
        console.print(f"  [red]✗ Error: {e}[/red]\n")
        results.append(
            {
                "bedrijfsnaam": case["bedrijfsnaam"],
                "error": str(e),
            }
        )


def _print_case_result(  # noqa: PLR0913
    case: dict,
    pred_is_kas: bool,
    pred_growlight: str,
    expected_en: str,
    kas_correct: bool,
    growlight_correct: bool,
    has_dutch: bool,
    has_crop: bool,
) -> None:
    """Print result for a single case."""
    both_correct = kas_correct and growlight_correct
    status = "[green]✓[/green]" if both_correct else "[red]✗[/red]"

    console.print(
        f"  {status} is_kas: {pred_is_kas} " f"(expected: {case['expected_is_kas']})"
    )
    console.print(f"      growlight: {pred_growlight} (expected: {expected_en})")
    console.print(f"      Dutch COT: {'Yes' if has_dutch else 'No'}")
    console.print(f"      Crop found: {'Yes' if has_crop else 'No'}")
    console.print()


def _show_summary(counters: dict[str, int], results: list[dict]) -> None:
    """Show summary table and gate decision."""
    console.print("\n" + "=" * 60)
    console.print("[bold]SUMMARY[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_column("Target")
    table.add_column("Status")

    n = len(TEST_CASES)

    # Calculate accuracies
    kas_acc = counters["correct_kas"] / n * 100
    gl_acc = counters["correct_growlight"] / n * 100
    dutch_rate = counters["dutch_count"] / n * 100
    crop_rate = counters["crop_identified_count"] / n * 100

    # is_kas accuracy
    kas_ok = kas_acc >= THRESHOLD_KAS_ACCURACY
    kas_status = "[green]PASS[/green]" if kas_ok else "[red]FAIL[/red]"
    table.add_row("is_kas accuracy", f"{kas_acc:.1f}%", ">80%", kas_status)

    # uses_growlight accuracy
    gl_ok = gl_acc > THRESHOLD_GROWLIGHT_BASELINE
    gl_status = "[green]PASS[/green]" if gl_ok else "[red]FAIL[/red]"
    table.add_row(
        "uses_growlight accuracy",
        f"{gl_acc:.1f}%",
        ">53.8% (baseline)",
        gl_status,
    )

    # Dutch COT rate
    dutch_ok = dutch_rate >= THRESHOLD_DUTCH_COT
    dutch_status = "[green]PASS[/green]" if dutch_ok else "[red]FAIL[/red]"
    table.add_row("Dutch in responses", f"{dutch_rate:.1f}%", ">50%", dutch_status)

    # Crop identification rate
    crop_ok = crop_rate >= THRESHOLD_CROP_ID
    crop_status = "[green]PASS[/green]" if crop_ok else "[yellow]WARN[/yellow]"
    table.add_row("Crop identification", f"{crop_rate:.1f}%", ">80%", crop_status)

    console.print(table)

    # Gate decision
    console.print()
    all_pass = kas_ok and gl_ok and dutch_ok
    if all_pass:
        console.print(
            "[bold green]✓ GATE PASSED - Ready for full optimization run[/bold green]"
        )
    else:
        console.print(
            "[bold red]✗ GATE FAILED - Review and debug before proceeding[/bold red]"
        )

    # Show detailed results
    console.print("\n[bold]Detailed Results:[/bold]\n")
    for r in results:
        if "error" in r:
            console.print(f"[red]{r['bedrijfsnaam']}: ERROR - {r['error']}[/red]")
        else:
            console.print(f"[cyan]{r['bedrijfsnaam']}[/cyan]")
            console.print(f"  Reasoning preview: {r['reasoning']}...")
            console.print()


def main() -> int:
    """Run the test."""
    parser = argparse.ArgumentParser(
        description="Test Dutch PerplexityKasDetector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (no API calls)
  uv run python scripts/test_dutch_kas_detector.py --dry-run

  # Full test with API calls
  uv run python scripts/test_dutch_kas_detector.py
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be tested without making API calls",
    )

    args = parser.parse_args()
    run_test(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
