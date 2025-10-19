"""Export functionality for enriched emitter data."""

import datetime
from pathlib import Path

import yaml

from .models import TopEmitter


def export_yaml(emitters: list[TopEmitter], output_path: Path) -> None:
    """
    Export enriched emitters to YAML file with metadata.

    Args:
        emitters: List of enriched TopEmitter models
        output_path: Path to save YAML file

    """
    # Convert Pydantic models to dictionaries
    emitters_data = [emitter.model_dump(mode="json") for emitter in emitters]

    # Convert Path objects to strings for YAML serialization
    for emitter_data in emitters_data:
        if emitter_data.get("streetview"):
            streetview = emitter_data["streetview"]
            for direction in ["north", "south", "east", "west"]:
                if streetview.get(direction):
                    streetview[direction] = str(streetview[direction])

    output_data = {
        "metadata": {
            "exported_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "total_emitters": len(emitters),
            "source": "NOAA VIIRS DNB Monthly",
        },
        "emitters": emitters_data,
    }

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write YAML
    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(output_data, f, sort_keys=False, indent=2, default_flow_style=False)
