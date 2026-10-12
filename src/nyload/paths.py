"""Filesystem layout. Relative paths resolve against the repository root.

Override locations with NYLOAD_DATA_DIR / NYLOAD_OUTPUT_DIR (e.g. a RunPod network volume).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | os.PathLike) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def data_dir() -> Path:
    return resolve(os.environ.get("NYLOAD_DATA_DIR", "data"))


def raw_dir() -> Path:
    return data_dir() / "raw"


def processed_dir() -> Path:
    return data_dir() / "processed"


def output_dir() -> Path:
    return resolve(os.environ.get("NYLOAD_OUTPUT_DIR", "outputs"))


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader: KEY=VALUE lines, existing environment wins."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
