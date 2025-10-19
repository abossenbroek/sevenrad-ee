"""
Unified CLI for VIIRS satellite data analysis tools.

This module provides a single entry point for all VIIRS-related commands:
- maps: Generate VIIRS DNB composite maps for regions
- top-polluters: Identify top nighttime light emitters
"""

import click

from sevenrad_ee.operations.generate_viirs_maps import maps
from sevenrad_ee.top_polluters.cli import top_polluters


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="VIIRS satellite data analysis tools",
)
def main() -> None:
    """Run the unified VIIRS CLI."""


# Register subcommands
main.add_command(maps)
main.add_command(top_polluters)


if __name__ == "__main__":
    main()
