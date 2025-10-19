# sevenrad-ee

A Python application for performing Google Earth Engine computations using modern Python 3.12.

## Prerequisites

- Python 3.12+
- [mise](https://mise.jdx.dev/) - Tool version manager
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer (installed via mise)

## Setup

### 1. Install mise

Follow the [mise installation guide](https://mise.jdx.dev/getting-started.html) for your platform.

### 2. Configure mise and install tools

```bash
mise install
```

This will install `uv` as specified in `.mise.toml`.

### 3. Create and activate virtual environment

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 4. Install dependencies

```bash
# Install package with development dependencies
uv pip install -e ".[dev]"
```

### 5. Install pre-commit hooks

```bash
uv run pre-commit install
```

### 6. Authenticate with Google Earth Engine

Before using this package, authenticate with Google Earth Engine:

```bash
uv run earthengine authenticate
```

## Development

### Code Quality Tools

This project enforces high code quality standards using:

- **ruff**: For code formatting and linting
- **mypy**: For static type checking
- **pre-commit**: For automated pre-commit checks

### Running Quality Checks

```bash
# Format code
uv run ruff format .

# Lint code (with auto-fix)
uv run ruff check . --fix

# Type check
uv run mypy src/sevenrad_ee
```

### Running Tests

```bash
uv run pytest
```

## Features

### VIIRS Top Polluters

Identify and analyze top light-emitting locations using NOAA VIIRS DNB satellite data:

- Query Google Earth Engine for brightest nighttime light patches
- Optional enrichment with Google Maps APIs (geocoding, business lookup, Street View)
- Export results to YAML with comprehensive metadata

**Quick Start:**

```bash
uv run viirs-top-polluters \
  --region my_region.geojson \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --businesses
```

📖 **Full Documentation**: [VIIRS Top Polluters Guide](src/sevenrad_ee/top_polluters/README.md)

## Project Structure

```
.
├── src/
│   └── sevenrad_ee/          # Main package
│       ├── top_polluters/    # VIIRS top emitters analysis
│       └── ...               # Other modules
├── tests/                    # Test files
├── pyproject.toml            # Project configuration
├── .mise.toml                # Tool version management
└── .pre-commit-config.yaml   # Pre-commit hooks
```

### Code Organization

This project follows the [Arkalos structure](https://arkalos.com/docs/structure/) for organizing code:

- **`src/sevenrad_ee/`**: Reusable code (functions, classes, modules)
  - Organized into logical, modular subfolders by functional domain
  - Examples: `ai/`, `data/`, `workflows/`, etc.
- **`notebooks/`**: Jupyter notebooks for exploration and analysis (code to run)
- **`scripts/`**: Standalone scripts for specific tasks (code to run)
- **`_private/`**: Personal or work-in-progress code (git-ignored)

**Key Principles:**
- **Separation of Concerns**: Distinguish between "code to run" (notebooks/scripts) and "code to reuse" (src/)
- **Modularity**: Place reusable components in logical subfolders
- **Privacy**: Use `_private/` for experimental work without accidentally committing

### Documentation Approach

We follow the [Diátaxis framework](https://diataxis.fr/) for documentation, which organizes content into four types based on user needs:

1. **Tutorials**: Learning-oriented lessons for beginners
2. **How-to guides**: Task-oriented directions for specific problems
3. **Technical reference**: Information-oriented API and code documentation
4. **Explanation**: Understanding-oriented discussions of concepts and design decisions

When contributing documentation, identify which type best serves the user's goal and follow the appropriate style and structure.

## Code Style

This project follows:
- PEP 8 style guide
- PEP 484 type hints
- Strict mypy checking
- Comprehensive ruff linting rules

All code is automatically checked by pre-commit hooks before committing.

## License

MIT
