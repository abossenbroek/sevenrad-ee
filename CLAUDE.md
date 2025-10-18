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
- **`src/sevenrad_ee/workflows/`**: Multi-step GEE processing pipelines
- **`notebooks/`**: Exploratory analysis and visualization

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

## Summary

When contributing to sevenrad-ee:
- ✅ Use `uv run` for all Python commands
- ✅ Use `~=` for all dependency version constraints
- ✅ Follow Arkalos structure (src/ for reusable, notebooks/scripts/ for runnable)
- ✅ Apply Diátaxis framework for documentation
- ✅ Add type hints and docstrings to all code
- ✅ Run pre-commit checks before committing
- ✅ Create PRs to `develop` branch
- ✅ Ensure CI/CD workflows pass
