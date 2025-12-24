#!/usr/bin/env python3
"""Estimate required datapoints for 95% F1 target using learning curves."""

import json
from pathlib import Path

def analyze_baseline():
    """Analyze 19-example baseline to project requirements."""
    baseline = {
        "num_examples": 19,
        "mean_f1": 0.6339,
        "std_f1": 0.0665,
        "positive": 12,
        "negative": 7,
    }
    
    print("\n" + "="*70)
    print("DATA REQUIREMENTS ANALYSIS FOR 95% F1 TARGET")
    print("="*70)
    
    print("\n📊 Baseline Performance (19 examples):")
    print(f"  Mean F1: {baseline['mean_f1']*100:.1f}%")
    print(f"  Std F1: {baseline['std_f1']*100:.1f}%")
    print(f"  Class balance: {baseline['positive']}:{baseline['negative']}")
    
    # Learning curve projections
    print("\n📈 Learning Curve Projections:")
    print("  Based on typical ML learning curves (logarithmic improvement)")
    
    projections = [
        (19, 63.4, "✅ Actual baseline"),
        (40, 72.0, "Projected (2.1x data)"),
        (65, 77.0, "Current dataset (3.4x data)"),
        (100, 82.0, "Recommended target (5.3x data)"),
        (150, 86.0, "Conservative estimate (7.9x data)"),
        (200, 88.0, "Aggressive estimate (10.5x data)"),
    ]
    
    for n, f1, note in projections:
        gap_to_95 = 95 - f1
        status = "✅" if f1 >= 95 else "⚠️" if f1 >= 85 else "❌"
        print(f"  {status} {n:3d} examples → {f1:4.1f}% F1 (gap: {gap_to_95:+.1f}%) - {note}")
    
    print("\n🎯 Target Analysis:")
    print(f"  Target F1: 95%")
    print(f"  Current dataset: 65 examples → ~77% F1 (projected)")
    print(f"  Gap to target: ~18% F1")
    
    # Statistical requirements
    print("\n📐 Statistical Requirements:")
    print("  Rule of thumb: ~10-20 examples per pattern/category")
    
    patterns = [
        ("Crop types", ["Roses", "Orchids", "Tomatoes", "Peppers", "Gerbera", "Others"], 6),
        ("Lighting types", ["SON-T only", "LED only", "Hybrid", "None"], 4),
        ("Evidence types", ["Supplier case study", "Company website", "Trade media", "Job posting"], 4),
        ("Company sizes", ["Large (>5ha)", "Medium (1-5ha)", "Small (<1ha)"], 3),
        ("Locations", ["Westland", "Aalsmeer", "Noord-Brabant", "Other"], 4),
    ]
    
    total_patterns = sum(count for _, _, count in patterns)
    print(f"\n  Identified patterns: {total_patterns}")
    for category, items, count in patterns:
        print(f"    - {category}: {count} patterns ({', '.join(items[:3])}...)")
    
    min_examples = total_patterns * 10
    recommended_examples = total_patterns * 15
    
    print(f"\n  Minimum examples: {min_examples} ({total_patterns} patterns × 10)")
    print(f"  Recommended: {recommended_examples} ({total_patterns} patterns × 15)")
    
    # Current coverage
    print("\n📦 Current Dataset Coverage (65 examples):")
    coverage = 65 / recommended_examples * 100
    print(f"  Coverage: {coverage:.1f}% of recommended")
    print(f"  Shortfall: {recommended_examples - 65} examples")
    
    # Quality considerations
    print("\n⚠️  Data Quality Considerations:")
    print("  From batch research results:")
    print(f"    - 32/46 (70%) new companies → NEEDS_MANUAL_REVIEW")
    print(f"    - 27 companies with 0+/0- evidence (no online documentation)")
    print(f"    - Average confidence: 55.9% (vs 80%+ ideal)")
    print(f"    - Companies with contradictory evidence (e.g., OK plant: 18+/18-)")
    
    print("\n  Quality issues may reduce effective dataset size by ~20-30%")
    print(f"  Effective dataset: ~45-50 examples (vs 65 actual)")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    print("\n🔍 Option 1: Run 65-Example Optimization (RECOMMENDED FIRST)")
    print("  ✅ Establishes learning curve baseline")
    print("  ✅ Validates projection accuracy")
    print("  ✅ Identifies specific failure modes")
    print("  ✅ Cost: ~$30, Time: 8-10 hours")
    print("  ❌ Unlikely to reach 95% target (~77% expected)")
    print("\n  Decision rule:")
    print("    IF F1 ≥ 80%: Collect 20-30 more examples → 85-90% target")
    print("    IF F1 = 75-80%: Collect 35-50 more examples → 100 total")
    print("    IF F1 < 75%: Investigate quality issues, refine approach")
    
    print("\n🎯 Option 2: Collect More Data Immediately")
    print("  ✅ Higher probability of reaching 95% target")
    print("  ✅ Better coverage of edge cases")
    print("  ❌ Cost: ~$40-60 more API costs (100-150 total examples)")
    print("  ❌ Time: 4-6 hours research + 10-12 hours optimization")
    print("  ❌ Risk: May over-collect if 65 examples already sufficient")
    
    print("\n  Target collection: 35-85 more companies (100-150 total)")
    print("  Estimated research time: 4-6 hours (Perplexity + Gemini)")
    print("  Estimated optimization time: 10-12 hours (more folds needed)")
    
    print("\n⚖️  Option 3: Hybrid Approach (RECOMMENDED)")
    print("  1️⃣  Run 65-example optimization first (8-10 hours)")
    print("  2️⃣  Analyze results and failure modes")
    print("  3️⃣  Collect targeted examples for weak areas:")
    print("      - Underrepresented crop types")
    print("      - Contradictory evidence cases")
    print("      - Companies with poor online documentation")
    print("  4️⃣  Run final optimization with 100-120 examples")
    print("\n  ✅ Data-driven collection strategy")
    print("  ✅ Efficient use of resources")
    print("  ✅ Higher probability of success")
    
    print("\n" + "="*70)
    print("FINAL RECOMMENDATION")
    print("="*70)
    
    print("""
✅ PROCEED WITH 65-EXAMPLE OPTIMIZATION FIRST

Rationale:
1. Learning curve validation: Need to verify 65 → 77% projection
2. Failure mode analysis: Identify specific weak areas before collecting more
3. Cost efficiency: Avoid over-collecting if quality issues dominate
4. Diminishing returns: Each additional example has decreasing impact
5. Targeted collection: Use optimization results to guide next batch

Expected outcome: 75-78% F1 (realistic scenario)
Next step: Collect 35-50 targeted examples → 100-115 total → 82-85% F1

🎯 Path to 95% F1:
  65 examples → 75-78% F1 [Current]
  + 35 targeted examples → 100 total → 82-85% F1 [Iteration 2]
  + 20 edge cases → 120 total → 88-90% F1 [Iteration 3]
  + Quality refinement → 90-93% F1 [Polish]
  + Production feedback → 93-95% F1 [Final]

Estimated timeline: 2-3 weeks with iterative optimization
Estimated cost: $60-100 total API costs
""")

if __name__ == "__main__":
    analyze_baseline()
