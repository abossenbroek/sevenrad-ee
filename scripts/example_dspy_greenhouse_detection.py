"""
Example script for DSPy-based greenhouse detection.

This script demonstrates how to use the GreenhouseDetector with Perplexity
to analyze greenhouse operations for artificial lighting usage.

Usage:
    uv run python scripts/example_dspy_greenhouse_detection.py
"""

import os
import sys
from pathlib import Path

import dspy

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from sevenrad_ee.ai.dspy_greenhouse import (  # noqa: E402
    analyze_greenhouse,
)


def main() -> int:
    """Run example greenhouse detection."""
    # Get API key from environment
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("Error: PERPLEXITY_API_KEY environment variable not set")  # noqa: T201
        print("Set it in .env file or export it:")  # noqa: T201
        print("export PERPLEXITY_API_KEY='your-key-here'")  # noqa: T201
        return 1

    # Configure Perplexity LM with DSPy
    # Try different model options if one doesn't work:
    # - "perplexity/sonar"
    # - "perplexity/sonar-pro"
    # - "perplexity/llama-3.1-sonar-small-128k-online"
    print("Configuring Perplexity LM...")  # noqa: T201
    lm = dspy.LM(
        "perplexity/sonar",
        api_key=api_key,
    )
    dspy.configure(lm=lm)

    # Example 1: Known Dutch greenhouse with artificial lighting
    print("\n" + "=" * 70)  # noqa: T201
    print("Example 1: Royal Van Zanten (Known greenhouse)")  # noqa: T201
    print("=" * 70)  # noqa: T201

    try:
        result1 = analyze_greenhouse(
            company_name="Royal Van Zanten",
            location="Rijsenhout, Nederland",
            additional_context="Major gerbera and rose grower",
        )

        print(f"\nCompany: Royal Van Zanten")  # noqa: T201
        print(f"Uses artificial lighting: {result1.uses_artificial_lighting}")  # noqa: T201
        print(f"Confidence: {result1.confidence:.2%}")  # noqa: T201
        print(f"Lighting type: {result1.lighting_type}")  # noqa: T201
        print(f"Primary crops: {result1.primary_crops}")  # noqa: T201
        print(f"Size: {result1.size_hectares} ha")  # noqa: T201
        print(f"Evidence: {result1.evidence[:2] if len(result1.evidence) > 0 else 'None'}")  # noqa: T201
        print(f"Sources: {result1.sources[:3] if len(result1.sources) > 0 else 'None'}")  # noqa: T201
        print(f"Reasoning: {result1.reasoning}")  # noqa: T201
    except Exception as e:
        print(f"Error analyzing Royal Van Zanten: {e}")  # noqa: T201

    # Example 2: Another Dutch greenhouse
    print("\n" + "=" * 70)  # noqa: T201
    print("Example 2: Custom greenhouse analysis")  # noqa: T201
    print("=" * 70)  # noqa: T201

    company_name = input("Enter greenhouse name (or press Enter for 'Duijvestijn Tomaten'): ")  # noqa: S603, S607
    if not company_name:
        company_name = "Duijvestijn Tomaten"

    location = input("Enter location (or press Enter for 'Pijnacker, Nederland'): ")  # noqa: S603, S607
    if not location:
        location = "Pijnacker, Nederland"

    try:
        result2 = analyze_greenhouse(
            company_name=company_name,
            location=location,
        )

        print(f"\nCompany: {company_name}")  # noqa: T201
        print(f"Uses artificial lighting: {result2.uses_artificial_lighting}")  # noqa: T201
        print(f"Confidence: {result2.confidence:.2%}")  # noqa: T201
        print(f"Lighting type: {result2.lighting_type}")  # noqa: T201
        print(f"Primary crops: {result2.primary_crops}")  # noqa: T201
        print(f"Size: {result2.size_hectares} ha")  # noqa: T201
        print(f"\nReasoning: {result2.reasoning}")  # noqa: T201

        # Show Dutch sources if any
        dutch_sources = [s for s in result2.sources if ".nl" in s]
        if dutch_sources:
            print(f"\nDutch sources found: {len(dutch_sources)}")  # noqa: T201
            for source in dutch_sources[:3]:
                print(f"  - {source}")  # noqa: T201

    except Exception as e:
        print(f"Error analyzing {company_name}: {e}")  # noqa: T201

    print("\n" + "=" * 70)  # noqa: T201
    print("Analysis complete!")  # noqa: T201
    print("=" * 70)  # noqa: T201

    return 0


if __name__ == "__main__":
    sys.exit(main())
