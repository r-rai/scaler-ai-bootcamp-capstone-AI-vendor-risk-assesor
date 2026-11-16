"""
memory.py — long-term memory that makes the NEXT review better (Deep Search course Module 7, Lab 8).

What is stored (runs/vendor_memory.json):
- vendors     a snapshot of the last profile per vendor, so a re-review highlights what changed (SEC-12.1)
- url_flags   pages an earlier run rejected as a different company or quarantined for prompt injection.
              A later run skips them BEFORE fetching: no download, no model call, no second chance to attack.
- lessons     short, reusable heuristics the plan creator reads before planning, with counts and dates

Guardrails from the course, enforced here:
- Memory never overrides fresh evidence. It only removes known-bad pages from the candidate list; every
  fact in a report still comes from a page read in that run.
- Flags expire (FLAG_TTL_DAYS). A site that was compromised last year gets re-checked rather than banned forever,
  and a re-review (`refresh`) ignores flags altogether.
- No personal data: vendors, URLs and lessons only.
- Writes are atomic and merge with what is on disk, so two concurrent runs cannot lose each other's updates.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import date
from pathlib import Path
from typing import Any, Optional

from .config import RUNS_DIR
from .schemas import VendorProfile

MEMORY_FILE = RUNS_DIR / "vendor_memory.json"
FLAG_TTL_DAYS = 180
_SAVE_LOCK = threading.Lock()


def _empty() -> dict[str, Any]:
    return {"vendors": {}, "bad_domains": [], "quarantined_urls": [], "url_flags": {}, "lessons": []}


class VendorMemory:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else MEMORY_FILE
        self.data = self._read()

    def _read(self) -> dict[str, Any]:
        data = _empty()
        if self.path.exists():
            try:
                data.update(json.loads(self.path.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001 — a corrupt memory file must not stop a review
                pass
        return data

    def save(self) -> None:
        """Atomic: write a temp file, then rename. A crash mid-write can no longer truncate the memory."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.data, indent=2, default=str, ensure_ascii=False)
        tmp = self.path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.path)

    # ---- reading ---------------------------------------------------------------
    def get_vendor_snapshot(self, vendor: str) -> Optional[dict]:
        return self.data.get("vendors", {}).get(vendor.lower())

    def flagged(self, url: str, today: date | None = None) -> Optional[str]:
        """'identity' or 'injection' if a recent run flagged this exact URL, else None."""
        flag = self.data.get("url_flags", {}).get(url)
        if not flag:
            return None
        try:
            age = ((today or date.today()) - date.fromisoformat(flag["last_seen"])).days
        except (KeyError, ValueError):
            return None
        return flag.get("reason") if age <= FLAG_TTL_DAYS else None

    def planner_tips(self, limit: int = 5) -> list[str]:
        lessons = sorted(self.data.get("lessons", []), key=lambda l: -l.get("times_seen", 1))
        return [l["text"] for l in lessons[:limit]]

    # ---- writing ---------------------------------------------------------------
    def _add_lesson(self, data: dict, text: str, evidence: str, today: str) -> None:
        for lesson in data["lessons"]:
            if lesson["text"] == text:
                lesson["times_seen"] = lesson.get("times_seen", 1) + 1
                lesson["last_seen"] = today
                return
        data["lessons"].append({"text": text, "evidence": evidence, "times_seen": 1, "first_seen": today, "last_seen": today})

    def record_assessment(self, profile: VendorProfile, *, lessons: dict[str, str] | None = None) -> list[str]:
        """Compare with the previous snapshot for SEC-12.1 diffing, then persist the new snapshot, flags and lessons."""
        key = profile.vendor.lower()
        today = date.today().isoformat()
        cur_facts = {k: v.display for k, v in profile.facts.items()}
        with _SAVE_LOCK:
            data = self._read()          # merge with whatever another run wrote since we loaded
            prev = data["vendors"].get(key)
            diffs: list[str] = []
            if prev and isinstance(prev.get("facts"), dict):
                old_facts = prev["facts"]
                for f, disp in cur_facts.items():
                    if f not in old_facts:
                        diffs.append(f"added {f}: {disp}")
                    elif old_facts[f] != disp:
                        diffs.append(f"changed {f}: '{old_facts[f]}' -> '{disp}'")
            data["vendors"][key] = {
                "vendor": profile.vendor,
                "official_domain": profile.official_domain,
                "last_reviewed": str(profile.as_of or today),
                "facts": cur_facts,
                "rejected": list(profile.rejected),
                "quarantined": list(profile.quarantined),
            }
            for url in profile.rejected:
                data["url_flags"][url] = {"reason": "identity", "vendor": key, "last_seen": today}
            for url in profile.quarantined:
                data["url_flags"][url] = {"reason": "injection", "vendor": key, "last_seen": today}
            data["bad_domains"] = sorted(set(data.get("bad_domains", [])) | set(profile.rejected))
            data["quarantined_urls"] = sorted(set(data.get("quarantined_urls", [])) | set(profile.quarantined))
            for text, evidence in (lessons or {}).items():
                self._add_lesson(data, text, evidence, today)
            self.data = data
            try:
                self.save()
            except Exception:  # noqa: BLE001
                pass
        return diffs
