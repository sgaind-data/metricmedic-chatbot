"""Central configuration. Everything can be overridden with environment variables,
which is what Docker/CI will use in later milestones."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = Path(
    os.getenv(
        "METRICMEDIC_DATA_PATH",
        PROJECT_ROOT / "data" / "troubleshooting_guide.json",
    )
)

# Local cache for knowledge-base embeddings (free, avoids re-encoding 51 docs
# on every start). Set METRICMEDIC_CACHE_DIR="" to disable.
_cache_env = os.getenv("METRICMEDIC_CACHE_DIR", str(PROJECT_ROOT / ".cache"))
CACHE_DIR = Path(_cache_env) if _cache_env else None

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Check which models are free on your project in Google AI Studio
# (Dashboard -> Rate limits) and set GEMINI_MODEL in .env if this one isn't.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))

DEFAULT_TOP_K = int(os.getenv("METRICMEDIC_TOP_K", "3"))


def get_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found. Add it to a .env file before running MetricMedic."
        )
    return api_key
