"""Tests for unified CLI entry point."""

import subprocess


class TestUnifiedCLI:
    """Test unified CLI structure and subcommands."""

    def test_viirs_help_shows_subcommands(self) -> None:
        """Main help shows both subcommands."""
        result = subprocess.run(
            ["uv", "run", "viirs", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert "maps" in result.stdout
        assert "top-polluters" in result.stdout

    def test_viirs_maps_help(self) -> None:
        """Maps subcommand help is accessible."""
        result = subprocess.run(
            ["uv", "run", "viirs", "maps", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert "--start-date" in result.stdout
        assert "--regions" in result.stdout  # Renamed from --maps

    def test_viirs_top_polluters_help(self) -> None:
        """Top polluters subcommand help is accessible."""
        result = subprocess.run(
            ["uv", "run", "viirs", "top-polluters", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert "REGION" in result.stdout  # Click shows arguments in caps
        assert "--start-date" in result.stdout

    def test_viirs_no_subcommand_shows_help(self) -> None:
        """Running 'viirs' alone shows help."""
        result = subprocess.run(
            ["uv", "run", "viirs"],
            capture_output=True,
            text=True,
            check=False,
        )

        # Should show help
        output = result.stdout + result.stderr
        assert "maps" in output or "top-polluters" in output
