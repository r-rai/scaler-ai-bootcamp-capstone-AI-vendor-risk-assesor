"""
memory.py — long-term vendor assessment memory for SEC-12.1 annual/change-triggered re-reviews.

COURSE MAP: Deep Search course Module 7 (Context, memory and iteration, Lab 8).
Stores:
- Known look-alike domains to skip immediately (entity memory)
- Known quarantined / injection URLs (source memory)
- Snapshot of the last VendorProfile per vendor so a re-review highlights what changed (SEC-12.1)
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Optional

from .config import RUNS_DIR
from .schemas import VendorProfile

MEMORY_FILE = RUNS_DIR / "vendor_memory.json"


class VendorMemory:
    def __init__(self, path: Path = MEMORY_FILE):
        self.path = Path(path)
        self.data: dict[str, Any] = {"vendors": {}, "bad_domains": [], "quarantined_urls": []}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    def get_vendor_snapshot(self, vendor: str) -> Optional[dict]:
        return self.data.get("vendors", {}).get(vendor.lower())

    def record_assessment(self, profile: VendorProfile) -> list[str]:
        """Compare with the previous snapshot for SEC-12.1 diffing and persist the new snapshot."""
        key = profile.vendor.lower()
        prev = self.data.setdefault("vendors", {}).get(key)
        diffs: list[str] = []
        cur_facts = {k: v.display for k, v in profile.facts.items()}
        if prev and isinstance(prev.get("facts"), dict):
            old_facts = prev["facts"]
            for f, disp in cur_facts.items():
                if f not in old_facts:
                    diffs.append(f"added {f}: {disp}")
                elif old_facts[f] != disp:
                    diffs.append(f"changed {f}: '{old_facts[f]}' -> '{disp}'")
        self.data["vendors"][key] = {
            "vendor": profile.vendor,
            "official_domain": profile.official_domain,
            "last_reviewed": str(profile.as_of or date.today()),
            "facts": cur_facts,
            "rejected": list(profile.rejected),
            "quarantined": list(profile.quarantined),
        }
        bad = set(self.data.get("bad_domains", [])) | set(profile.rejected)
        quar = set(self.data.get("quarantined_urls", [])) | set(profile.quarantined)
        self.data["bad_domains"] = sorted(bad)
        self.data["quarantined_urls"] = sorted(quar)
        try:
            self.save()
        except Exception:  # noqa: BLE001
            pass
        return diffs
