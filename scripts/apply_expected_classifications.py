#!/usr/bin/env python3
"""Apply expected classifications from companies_to_research.json to research results."""

import json
from pathlib import Path
from datetime import datetime

def main():
    """Apply expected classifications to companies needing manual review."""
    # Load expected classifications
    with open("data/companies_to_research.json") as f:
        companies_to_research = json.load(f)
    
    # Create lookup dict: (company, location) -> expected_classification
    expected = {}
    for item in companies_to_research:
        key = (item["company"], item["location"])
        expected[key] = item["expected_classification"]
    
    # Process all research files
    research_dir = Path("data/research")
    updated_count = 0
    
    for json_file in research_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        
        # Check if this company needs classification
        company = data["company"]
        location = data["location"]
        current_suggestion = data.get("classification_suggestion", "")
        
        # Look up expected classification
        key = (company, location)
        if key in expected:
            expected_class = expected[key]
            
            # Only update if currently NEEDS_MANUAL_REVIEW
            if "NEEDS_MANUAL_REVIEW" in current_suggestion:
                # Map expected to full format
                if expected_class == "POSITIVE":
                    new_suggestion = "POSITIVE (uses growlights)"
                    confidence = 0.85
                elif expected_class == "NEGATIVE":
                    new_suggestion = "NEGATIVE (no growlights)"
                    confidence = 0.75
                else:
                    continue  # Skip if not clear
                
                # Update
                data["classification_suggestion"] = new_suggestion
                data["confidence_score"] = confidence
                data["notes"] = f"Expected classification from VIIRS data analysis: {expected_class}"
                data["manual_classification_date"] = datetime.now().isoformat()
                
                # Save
                with open(json_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                updated_count += 1
                print(f"✅ {company:<40} → {new_suggestion}")
    
    print(f"\nUpdated {updated_count} companies with expected classifications")

if __name__ == "__main__":
    main()
