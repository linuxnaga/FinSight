"""Central configuration. Everything is read from environment / .env so the
same code runs locally, in Docker and in the eval harness."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("FINSIGHT_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
SECTIONS_DIR = DATA_DIR / "sections"
XBRL_DIR = DATA_DIR / "xbrl_cache"
INDEX_DIR = DATA_DIR / "index"
TRACE_DIR = DATA_DIR / "traces"
CACHE_DB = DATA_DIR / "cache.sqlite"

# 20 companies chosen to overlap with FinanceBench (open-source subset) so its
# gold questions can be reused. ticker -> (CIK, display name)
COMPANIES: dict[str, tuple[int, str]] = {
    "AMD": (2488, "AMD"),
    "BA": (12927, "Boeing"),
    "AXP": (4962, "American Express"),
    "PEP": (77476, "PepsiCo"),
    "MMM": (66740, "3M"),
    "ADBE": (796343, "Adobe"),
    "AMCR": (1748790, "Amcor"),
    "BBY": (764478, "Best Buy"),
    "VZ": (732712, "Verizon"),
    "GLW": (24741, "Corning"),
    "CVS": (64803, "CVS Health"),
    "GIS": (40704, "General Mills"),
    "NKE": (320187, "Nike"),
    "AES": (874761, "AES Corporation"),
    "AMZN": (1018724, "Amazon"),
    "AWK": (1410636, "American Water Works"),
    "KO": (21344, "Coca-Cola"),
    "JNJ": (200406, "Johnson & Johnson"),
    "LMT": (936468, "Lockheed Martin"),
    "PFE": (78003, "Pfizer"),
}
FISCAL_YEARS = [2020, 2021, 2022, 2023]


def fiscal_year_of(period_end: str) -> int:
    """Fiscal year label for a period ending on `period_end` (YYYY-MM-DD).

    Companies name a fiscal year after the calendar year in which it ends
    (Nike FY2023 ends May 2023, Best Buy FY2023 ends 28 Jan 2023). Exception:
    52/53-week years that spill a few days into January belong to the prior
    year (J&J FY2022 ends 1 Jan 2023)."""
    y, m, d = int(period_end[:4]), int(period_end[5:7]), int(period_end[8:10])
    return y - 1 if m == 1 and d <= 7 else y


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "none"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ"))
    llm_json_mode: str = field(default_factory=lambda: _env("LLM_JSON_MODE", "json_object"))

    embed_base_url: str = field(default_factory=lambda: _env("EMBED_BASE_URL") or _env("LLM_BASE_URL"))
    embed_api_key: str = field(default_factory=lambda: _env("EMBED_API_KEY") or _env("LLM_API_KEY", "none"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    embed_backend: str = field(default_factory=lambda: _env("EMBED_BACKEND", "api"))

    reranker: str = field(default_factory=lambda: _env("RERANKER", "none"))
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL", "BAAI/bge-reranker-base"))

    sec_user_agent: str = field(default_factory=lambda: _env("SEC_USER_AGENT", "FinSight-Capstone contact@example.com"))

    max_steps: int = field(default_factory=lambda: int(_env("MAX_STEPS", "10")))
    max_tokens_per_query: int = field(default_factory=lambda: int(_env("MAX_TOKENS_PER_QUERY", "12000")))
    max_critic_revisions: int = field(default_factory=lambda: int(_env("MAX_CRITIC_REVISIONS", "2")))
    rate_limit_per_min: int = field(default_factory=lambda: int(_env("RATE_LIMIT_PER_MIN", "20")))


def settings() -> Settings:
    return Settings()
