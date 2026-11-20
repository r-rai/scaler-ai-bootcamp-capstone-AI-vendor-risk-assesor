import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import pytest

# Force the deterministic, free configuration BEFORE assessor.config is imported: a local .env that
# selects a live provider must never make the suite call a real API. Run outputs go to a temp folder.
os.environ["LLM_PROVIDER"] = "mock"
os.environ["SEARCH_PROVIDER"] = "offline"
os.environ["VRA_ENV"] = "dev"
os.environ["VRA_CACHE"] = "0"          # caches are off unless a test builds one explicitly
os.environ["VRA_LOG_LEVEL"] = "WARNING"
os.environ.setdefault("VRA_RUNS_DIR", tempfile.mkdtemp(prefix="vra-test-runs-"))
for _k in ("LLM_SMALL_PROVIDER", "LLM_EXTRACT_PROVIDER", "VRA_TOKENS_FILE"):
    os.environ.pop(_k, None)
os.environ["FIRECRAWL_API_KEY"] = ""    # a key in .env must never let the suite call (and pay) Firecrawl; tests that need one set it themselves

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assessor.config import SETTINGS  # noqa: E402
from assessor.policy_rag import PolicyLibrary  # noqa: E402
from assessor.schemas import VendorFact, VendorProfile  # noqa: E402

SETTINGS.verbose = False
AS_OF = date(2026, 11, 1)


@pytest.fixture(scope="session")
def library() -> PolicyLibrary:
    return PolicyLibrary()


def fact(field: str, value, level: str = "documented", as_of: date = date(2026, 6, 1)) -> VendorFact:
    return VendorFact(field=field, value=value, display=str(value), confidence=0.8, as_of=as_of, evidence_level=level,
                      sources=[f"https://vendor.example/{field}"])


def profile(**facts) -> VendorProfile:
    p = VendorProfile(vendor="TestVendor", official_domain="vendor.example", as_of=AS_OF)
    for k, v in facts.items():
        p.facts[k] = v
    return p
