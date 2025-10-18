# Claude Code Instructions for sevenrad-ee

## Project Overview

**sevenrad-ee** is a Python 3.12+ application for performing Google Earth Engine (GEE) computations. The project emphasizes modern Python tooling, strict code quality standards, and systematic organization.

## Core Technologies

- **Python**: 3.12+ (strict type checking with mypy)
- **Package Manager**: uv (fast, modern dependency management)
- **Version Manager**: mise (tool version management)
- **Code Quality**: ruff (formatting + linting), mypy (type checking), pre-commit (automated checks)
- **Testing**: pytest with coverage reporting
- **CI/CD**: GitHub Actions (pre-commit checks, tests on Python 3.12 & 3.13)

## Dependency Management Policy

**CRITICAL**: All project dependencies MUST use the `~=` (compatible release) constraint:

```toml
# Correct - allows bug fixes only (1.1.0 → 1.1.x)
"earthengine-api~=1.1.0"

# Wrong - allows minor versions (1.1.0 → 1.x.x)
"earthengine-api>=1.1.0"
```

**Rationale**: The `~=` constraint allows only patch/bug fix updates, preventing unexpected breaking changes while still receiving security fixes.

**When adding dependencies**:
1. Pin to a specific major.minor version with `~=`
2. Apply to both production AND dev dependencies
3. Update pyproject.toml following this pattern

## Code Organization (Arkalos Structure)

We follow the [Arkalos structure](https://arkalos.com/docs/structure/) for organizing code:

### Directory Structure

```
src/sevenrad_ee/     # Reusable code (code to reuse)
  ├── ai/            # AI/ML functionality
  ├── data/          # Data processing modules
  ├── operations/    # CLI operations and tools
  ├── workflows/     # GEE workflow orchestration
  └── utils/         # Utility functions

notebooks/           # Jupyter notebooks (code to run)
scripts/             # Standalone scripts (code to run)
tests/               # Test files
_private/            # Personal/WIP code (git-ignored)
```

### Key Principles

1. **Separation of Concerns**:
   - **`src/sevenrad_ee/`**: Reusable functions, classes, and modules
   - **`notebooks/` & `scripts/`**: Executable code for specific tasks
   - **`_private/`**: Experimental/WIP code not ready for commit

2. **Modularity**: Organize code into logical, domain-specific subfolders

3. **Reusability**: If code is used more than once, it belongs in `src/`

### When Adding New Code

**Ask yourself**: "Is this code to run or code to reuse?"

- **Code to reuse** → Place in `src/sevenrad_ee/` subdirectory
- **Code to run** → Place in `notebooks/` or `scripts/`
- **Experimental** → Place in `_private/` until ready

## Documentation Approach (Diátaxis Framework)

We follow the [Diátaxis framework](https://diataxis.fr/) for all documentation. Before creating any documentation, identify which of the four types applies:

### 1. **Tutorials** (Learning-oriented)
- **Purpose**: Teach beginners through hands-on lessons
- **Form**: Step-by-step guidance with concrete examples
- **Example**: "Getting Started with GEE VIIRS Data"
- **Tone**: Encouraging, educational, safe to follow

### 2. **How-to Guides** (Task-oriented)
- **Purpose**: Solve specific problems for experienced users
- **Form**: Numbered steps to achieve a goal
- **Example**: "How to Process VIIRS Nighttime Light Data"
- **Tone**: Direct, practical, goal-focused

**Standard How-to Guide Structure:**

1. **Document Type Label**: State "Documentation Type: How-to Guide" at top
2. **Brief Introduction**: One sentence describing what the guide covers
3. **Prerequisites**: Required setup, authentication, or files
4. **Quick Start**: Most common use case, including discovery options
   - Start with `--list-X` option if applicable
   - Show the simplest successful execution
5. **Configuration**: Explain configuration files and options
6. **Common Tasks**: Specific examples for frequent scenarios
7. **Troubleshooting**: Common errors and solutions
8. **Advanced Usage**: Optional advanced patterns
9. **Related Documentation**: Links to tutorials, references, explanations

**Example Quick Start Section:**
```markdown
## Quick Start

### 1. List Available Options

\```bash
uv run python -m module.name --list-options
\```

This displays all available choices with descriptions.

### 2. Run with Default Settings

\```bash
uv run python -m module.name --option value
\```
```

**Always include discovery mechanisms** (`--list-X`, `--help`) as the first step to help users explore capabilities.

### 3. **Technical Reference** (Information-oriented)
- **Purpose**: Describe the API, functions, and code structure
- **Form**: Systematic descriptions of functions, parameters, returns
- **Example**: API documentation, function docstrings
- **Tone**: Neutral, accurate, complete

### 4. **Explanation** (Understanding-oriented)
- **Purpose**: Explain concepts, design decisions, and context
- **Form**: Discursive writing about "why" and "how it works"
- **Example**: "Why We Use ~= Version Constraints"
- **Tone**: Thoughtful, conceptual, discussion-based

**When writing documentation**:
- Clearly identify which Diátaxis type you're creating
- Follow the appropriate style and structure for that type
- Don't mix types (e.g., don't put tutorials in reference docs)

## Development Workflow

### Always Use `uv run`

**CRITICAL**: All Python commands MUST be prefixed with `uv run` to ensure they execute in the correct virtual environment:

```bash
# Correct
uv run pytest
uv run ruff format .
uv run mypy src/sevenrad_ee
uv run earthengine authenticate

# Wrong
pytest
ruff format .
```

**Why**: `uv run` ensures commands use the project's virtual environment without manual activation.

### Standard Development Commands

```bash
# Setup
mise install                    # Install uv
uv venv                         # Create virtual environment
uv pip install -e ".[dev]"      # Install dependencies
uv run pre-commit install       # Install git hooks

# Code Quality
uv run ruff format .            # Format code
uv run ruff check . --fix       # Lint with auto-fix
uv run mypy src/sevenrad_ee     # Type check

# Testing
uv run pytest                   # Run tests
uv run pytest --cov             # Run with coverage

# GEE Authentication
uv run earthengine authenticate # Authenticate with Google Earth Engine
```

## Code Quality Standards

### Type Hints (Required)

**All functions MUST have complete type hints**:

```python
# Correct
def process_viirs_data(
    image_collection: ee.ImageCollection,
    start_date: str,
    end_date: str
) -> ee.Image:
    """Process VIIRS nighttime light data."""
    ...

# Wrong - missing type hints
def process_viirs_data(image_collection, start_date, end_date):
    ...
```

### Docstrings (Required)

**All public functions, classes, and modules MUST have docstrings**:

```python
def extract_nighttime_lights(
    region: ee.Geometry,
    date_range: tuple[str, str]
) -> ee.Image:
    """
    Extract VIIRS nighttime light data for a specified region and date range.

    Args:
        region: Geographic region of interest as Earth Engine Geometry
        date_range: Tuple of (start_date, end_date) in 'YYYY-MM-DD' format

    Returns:
        Composite image of nighttime lights for the specified period

    Raises:
        ValueError: If date_range is invalid or region is empty
    """
    ...
```

### Ruff Linting Rules

The project enforces comprehensive linting:
- **D**: pydocstyle (docstring conventions)
- **S**: flake8-bandit (security)
- **B**: flake8-bugbear (common bugs)
- **ANN**: flake8-annotations (type annotations)
- **I**: isort (import sorting)
- **Plus**: Many more (see pyproject.toml)

**Pre-commit hooks automatically check these before every commit.**

## CLI Development Standards

### Rich Library for Output

**All CLI tools MUST use the Rich library** for colorful, user-friendly output:

```python
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
```

### Required CLI Features

Every CLI tool MUST include:

1. **Discovery Options**: Provide `--list-X` flags to show available choices
2. **Progress Indicators**: Use spinners for operations without known duration
3. **Colorful Output**: Use Rich for tables, panels, and formatted text
4. **Help Examples**: Include usage examples in argparse epilog
5. **Error Guidance**: Error messages should suggest how to fix the issue

### Standard CLI Pattern

```python
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tool description",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available options
  uv run python -m module.name --list-options

  # Run with specific option
  uv run python -m module.name --option value
        """,
    )

    parser.add_argument(
        "--list-options",
        action="store_true",
        help="List available options and exit",
    )

    return parser.parse_args()

def main() -> int:
    """Run the CLI."""
    # Display header
    console.print(
        Panel.fit(
            "[bold cyan]Tool Name[/bold cyan]\\n"
            "Tool description",
            border_style="cyan",
        )
    )

    args = parse_arguments()

    # Handle discovery options first
    if args.list_options:
        display_options()
        return 0

    # Main processing with progress indicators
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=None)
        # ... do work ...
        progress.update(task, completed=True)

    return 0
```

### Error Handling Pattern

Provide actionable error messages:

```python
if not config_file.exists():
    console.print(
        f"[red]✗[/red] Config file not found: {config_file}",
        style="bold red",
    )
    console.print(
        "\\n[yellow]Tip:[/yellow] Specify config file with --config option"
    )
    return 1
```

## Git Workflow

### Remote Configuration

**IMPORTANT**: This repository uses SSH for git operations:

```bash
# Remote should be SSH, not HTTPS
origin  git@github.com:abossenbroek/sevenrad-ee.git
```

If you encounter permission errors with workflows, verify the remote URL uses SSH.

### Branch Strategy

- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/***: Feature development branches

**Always create PRs to `develop`, not `main`.**

### Commit Messages

Follow the established pattern:
```
<Brief description of change>

<Detailed explanation if needed>
- Bullet points for specifics
- Additional context

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Pull Request Updates

**When updating PRs based on feedback:**

1. **Create new commits** (don't amend existing commits after PR creation)
   - Each logical improvement should be its own commit
   - Example: Initial implementation → Documentation → Enum improvements

2. **Push updates to the same branch**:
   ```bash
   git add .
   git commit -m "Improve region selection with enum and discovery features

   Add KnownRegion enum for structured region validation...

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>"

   git push origin feature/branch-name
   ```

3. **PR automatically updates** when you push to the branch
   - No need to create a new PR
   - Each commit appears in the PR timeline
   - Reviewers can see iterative improvements

4. **When to amend commits**:
   - Only amend commits that haven't been pushed yet
   - Never amend commits after creating a PR (creates history conflicts)

**Pattern:**
- Initial work → Create PR → Feedback → New commit → Push → PR updates automatically

## Google Earth Engine Specifics

### Authentication

Always use `uv run` for authentication:
```bash
uv run earthengine authenticate
```

### Common GEE Patterns

When working with Earth Engine code:
1. Import `ee` module
2. Initialize with `ee.Initialize()`
3. Use type hints: `ee.Image`, `ee.ImageCollection`, `ee.Geometry`, etc.
4. Handle authentication errors gracefully

### Data Processing

Organize GEE-specific code:
- **`src/sevenrad_ee/data/`**: Data acquisition and preprocessing
- **`src/sevenrad_ee/operations/`**: CLI tools for GEE operations
- **`src/sevenrad_ee/workflows/`**: Multi-step GEE processing pipelines
- **`notebooks/`**: Exploratory analysis and visualization

### CLI Tools for GEE Operations

**Structure for GEE CLI tools:**
- Place in `src/sevenrad_ee/operations/` directory
- Use Rich library for progress indicators
- Provide `--list-X` options for region/dataset discovery
- Handle authentication errors with helpful messages

**Progress Reporting Pattern:**

Use spinners for server-side operations where duration is unknown:

```python
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
) as progress:
    task = progress.add_task("Loading VIIRS DNB collection...", total=None)
    collection = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
    progress.update(task, completed=True)
```

**GEE-Specific Error Handling:**

```python
def initialize_earth_engine() -> None:
    """Initialize Google Earth Engine API."""
    try:
        ee.Initialize()
        console.print("[green]✓[/green] Earth Engine initialized")
    except Exception as e:
        console.print(
            f"[red]✗[/red] Failed to initialize: {e}",
            style="bold red"
        )
        console.print(
            "\\n[yellow]Run:[/yellow] uv run earthengine authenticate",
            style="italic"
        )
        raise
```

### Quotas and Rate Limits

- GEE operations are subject to quotas
- Design CLI tools to process regions/time periods in batches
- Provide options to limit scope (single region, shorter time range)
- Document quota considerations in tool documentation

### Type Hints for Earth Engine

Use TYPE_CHECKING pattern for ee module:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ee
else:
    import ee  # type: ignore[no-redef]
```

Configure mypy to ignore ee imports:
```toml
[[tool.mypy.overrides]]
module = "ee"
ignore_missing_imports = true
```

## CI/CD

### GitHub Actions Workflows

Two workflows run automatically:

1. **Pre-commit** (`.github/workflows/pre-commit.yml`)
   - Runs ruff format, ruff check, mypy
   - Triggers on push and PRs
   - Uploads error logs on failure

2. **Tests** (`.github/workflows/test.yml`)
   - Runs pytest on Python 3.12 & 3.13
   - Coverage reporting to Codecov
   - Triggers on push and PRs

**Ensure code passes both workflows before merging PRs.**

## Common Patterns

### Adding a New Module

1. Identify if it's "code to reuse" or "code to run"
2. Place in appropriate directory (src/ vs notebooks/scripts/)
3. Add type hints to all functions
4. Write docstrings (identify Diátaxis type)
5. Add tests in `tests/`
6. Run quality checks: `uv run ruff format . && uv run ruff check . && uv run mypy src/sevenrad_ee`

### Adding a New Dependency

1. Determine the current stable version
2. Add to `pyproject.toml` with `~=` constraint:
   ```toml
   dependencies = [
       "new-package~=2.3.0",  # Note the ~= constraint
   ]
   ```
3. Install: `uv pip install -e ".[dev]"`
4. Test that it works
5. Commit pyproject.toml changes

### Parsing External Configuration Files

When parsing JavaScript or other external configuration files:

1. **Use regex for simple JavaScript parsing**:
   ```python
   import re

   def parse_js_value(js_content: str, var_name: str) -> str | None:
       """Parse JavaScript variable value."""
       pattern = rf"var\\s+{var_name}\\s*=\\s*([^;]+);"
       match = re.search(pattern, js_content)
       return match.group(1) if match else None
   ```

2. **Provide built-in fallbacks**:
   - Never depend solely on user-provided config files
   - Include default values or built-in constants
   - Example: Built-in WAVELENGTH_PALETTE_1025 when JS palette missing

3. **Handle user-specific config files**:
   - Add to `.gitignore` (e.g., `extract_geotiffs.js`, `*_config.js`)
   - Document expected format in docs/
   - Provide examples or templates in documentation

4. **Validate parsed values**:
   - Check types and ranges after parsing
   - Provide clear error messages if validation fails
   - Suggest correct format in error messages

### Enums for User-Selectable Values

**Use Enums for known user-facing choices** to enable validation and discovery:

```python
from enum import Enum

class KnownRegion(str, Enum):
    """Known region names for processing."""

    DRC = "drc"
    US = "us"
    EUROPE = "europe"

    @classmethod
    def all_regions(cls) -> list[str]:
        """Get list of all region codes."""
        return [region.value for region in cls]

    @classmethod
    def display_names(cls) -> dict[str, str]:
        """Get friendly display names."""
        return {
            cls.DRC.value: "Democratic Republic of Congo",
            cls.US.value: "United States",
            cls.EUROPE.value: "Europe",
        }
```

**Benefits:**
- **Discovery**: Users can list available options (`--list-maps`)
- **Validation**: Easy to check if user input is valid
- **Documentation**: Self-documenting code with display names
- **Type Safety**: Prevents typos and invalid values

**Pattern for CLI Discovery:**
```python
def list_available_options() -> None:
    """Display available options with descriptions."""
    display_names = KnownOption.display_names()
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Code", style="green")
    table.add_column("Description", style="white")

    for code in KnownOption.all_options():
        table.add_row(code, display_names.get(code, code))

    console.print(table)
```

**When to use Enums:**
- Region selections
- Processing modes or strategies
- Output formats
- Any fixed set of user-facing choices

## Summary

When contributing to sevenrad-ee:
- ✅ Use `uv run` for all Python commands
- ✅ Use `~=` for all dependency version constraints
- ✅ Follow Arkalos structure (src/ for reusable, notebooks/scripts/ for runnable)
- ✅ Apply Diátaxis framework for documentation
- ✅ Add type hints and docstrings to all code
- ✅ Use Rich library for all CLI tools
- ✅ Provide `--list-X` discovery options in CLIs
- ✅ Use Enums for user-selectable values
- ✅ Run pre-commit checks before committing
- ✅ Create PRs to `develop` branch
- ✅ Create new commits (don't amend) when updating PRs
- ✅ Ensure CI/CD workflows pass
