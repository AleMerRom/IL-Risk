"""Module 1 entrypoint.

Examples:
    python scripts/data_extraction.py extract all
    python scripts/data_extraction.py validate
"""

from il_risk.cli import app


if __name__ == "__main__":
    app()
