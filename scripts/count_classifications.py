#!/usr/bin/env python3
"""Count final classification distribution."""

import json
from pathlib import Path
from collections import Counter

def main():
    research_dir = Path("data/research")
    classifications = []
    
    for json_file in research_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        
        suggestion = data.get("classification_suggestion", "UNKNOWN")
        
        # Normalize to POSITIVE/NEGATIVE/MANUAL_REVIEW
        if "POSITIVE" in suggestion:
            classifications.append("POSITIVE")
        elif "NEGATIVE" in suggestion:
            classifications.append("NEGATIVE")
        else:
            classifications.append("NEEDS_MANUAL_REVIEW")
    
    counts = Counter(classifications)
    total = sum(counts.values())
    
    print("\nFinal Dataset Distribution:")
    print("=" * 50)
    print(f"Total companies: {total}")
    print(f"  POSITIVE:            {counts['POSITIVE']:2d} ({counts['POSITIVE']/total*100:.1f}%)")
    print(f"  NEGATIVE:            {counts['NEGATIVE']:2d} ({counts['NEGATIVE']/total*100:.1f}%)")
    if counts['NEEDS_MANUAL_REVIEW'] > 0:
        print(f"  NEEDS_MANUAL_REVIEW: {counts['NEEDS_MANUAL_REVIEW']:2d} ({counts['NEEDS_MANUAL_REVIEW']/total*100:.1f}%)")
    print("=" * 50)
    print(f"\nClass balance: {counts['POSITIVE']}:{counts['NEGATIVE']} "
          f"(ratio {counts['POSITIVE']/counts['NEGATIVE']:.2f}:1)")
    
    if counts['NEEDS_MANUAL_REVIEW'] == 0:
        print("\n✅ All companies classified! Ready for GEPA optimization.")
    else:
        print(f"\n⚠️  {counts['NEEDS_MANUAL_REVIEW']} companies still need manual review")

if __name__ == "__main__":
    main()
