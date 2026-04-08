"""Small `.env` loader for local development.

Loads KEY=VALUE pairs from a `.env` file in the project root without requiring
extra dependencies. Existing environment variables are preserved.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def load_dotenv(dotenv_path: Path | None = None) -> Path:
    """Load a `.env` file into `os.environ` if it exists."""
    path = dotenv_path or (PROJECT_ROOT / ".env")
    if not path.exists():
        return path

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key or key in os.environ:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ[key] = value

    return path
