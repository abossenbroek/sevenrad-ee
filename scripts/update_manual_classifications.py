#!/usr/bin/env python3
"""Update manual classifications for the 9 NEEDS_MANUAL_REVIEW companies."""

import json
from pathlib import Path
from datetime import datetime

# Manual classifications from user
MANUAL_CLASSIFICATIONS = {
    "Batist_Westmade_Made.json": {
        "classification_suggestion": "POSITIVE (uses growlights)",
        "species_grown": "Gerbera",
        "confidence_score": 0.85,
        "notes": "Manual classification: Gerbera cultivation confirmed positive for grow lights"
    },
    "Fachjan_Project_Plants_Honselersdijk.json": {
        "classification_suggestion": "NEGATIVE (no growlights)",
        "species_grown": "Unknown",
        "confidence_score": 0.80,
        "notes": "Manual classification: Positive but no nightlight, unlikely to show as extreme case in VIIRS data"
    },
    "Kwekerij_De_Opstal_V.O.F._Naaldwijk.json": {
        "classification_suggestion": "POSITIVE (uses growlights)",
        "species_grown": "Roses",
        "confidence_score": 0.95,
        "notes": "Manual classification: Roses cultivation - absolutely positive for grow lights"
    },
    "Kwekerij_Figaro_B.V._Naaldwijk.json": {
        "classification_suggestion": "POSITIVE (uses growlights)",
        "species_grown": "Unknown",
        "confidence_score": 0.85,
        "notes": "Manual classification: Confirmed positive for grow lights"
    },
    "Kwekerij_Overgaag_Maasland.json": {
        "classification_suggestion": "POSITIVE (uses growlights)",
        "species_grown": "Red Peppers",
        "confidence_score": 0.90,
        "notes": "Manual classification: Red peppers cultivation - positive for grow lights"
    },
    "Kwekerij_Ted_Vijverberg_B.V._De Lier.json": {
        "classification_suggestion": "NEGATIVE (no growlights)",
        "species_grown": "Unknown",
        "confidence_score": 0.70,
        "notes": "Manual classification: Inconclusive; lower light levels, not likely to be extreme case visible on VIIRS data"
    },
    "Kwekerij_Vicini_Herenwerf_Maasland.json": {
        "classification_suggestion": "NEGATIVE (no growlights)",
        "species_grown": "Unknown",
        "confidence_score": 0.80,
        "notes": "Manual classification: No grow lights confirmed"
    },
    "P.J.J.M._Verbeek_en_P.H.M._Verbeek_Maasland.json": {
        "classification_suggestion": "NEGATIVE (no growlights)",
        "species_grown": "Unknown",
        "confidence_score": 0.75,
        "notes": "Manual classification: Lower light levels, less likely to be seen as extreme cases visible in VIIRS data"
    },
    "Van_der_Sar_Plants_'s Gravenzande.json": {
        "classification_suggestion": "NEGATIVE (no growlights)",
        "species_grown": "Garden plants",
        "confidence_score": 0.80,
        "notes": "Manual classification: Garden plants typically only grown with light in winter, very unlikely to show as extreme high value in VIIRS data"
    },
}

def main():
    """Update manual classifications."""
    data_dir = Path("data/research")
    updated_count = 0

    print("Updating manual classifications...")
    print("=" * 80)

    for filename, updates in MANUAL_CLASSIFICATIONS.items():
        filepath = data_dir / filename

        if not filepath.exists():
            print(f"⚠️  File not found: {filename}")
            continue

        # Load existing data
        with open(filepath) as f:
            data = json.load(f)

        old_classification = data.get("classification_suggestion", "Unknown")
        new_classification = updates["classification_suggestion"]

        # Update fields
        data["classification_suggestion"] = new_classification
        data["species_grown"] = updates["species_grown"]
        data["confidence_score"] = updates["confidence_score"]
        data["notes"] = updates["notes"]
        data["manual_review_date"] = datetime.now().isoformat()

        # Save updated data
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        updated_count += 1
        print(f"✅ {data['company']:<40} {old_classification:<30} → {new_classification}")

    print("=" * 80)
    print(f"\nUpdated {updated_count}/{len(MANUAL_CLASSIFICATIONS)} companies")

    # Show new distribution
    positive = 0
    negative = 0

    for updates in MANUAL_CLASSIFICATIONS.values():
        if "POSITIVE" in updates["classification_suggestion"]:
            positive += 1
        elif "NEGATIVE" in updates["classification_suggestion"]:
            negative += 1

    print(f"\nManual Review Results:")
    print(f"  POSITIVE: {positive}")
    print(f"  NEGATIVE: {negative}")

if __name__ == "__main__":
    main()
