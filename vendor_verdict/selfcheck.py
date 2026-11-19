"""`python -m vendor_verdict selfcheck`: a free mechanics check of the REAL graph against the REAL competition `rules_lite` and
requirements on a few dev cases (ledger ruling R30: no pytest test may read `data_kaggle/`, so nothing else runs `nodes.merge`/`verdict`
and the runner's preds path against the real `evaluate(facts, reqs, tier, users, as_of)` / `overall` and the real data shapes).

    python -m vendor_verdict selfcheck [--split train] [--subset dev] [--limit 3] [--only ids] [--data DIR]

No model call and no network call: every model is a stub, and a socket guard counts (and refuses) any connection attempt.
- Replay reader: the specialist stub answers each (page, family) with the BASELINE's cached claims of that page for the family's
  fields (`baseline.baseline_readings`, list values split into one reading per item, the specialists' shape); an iteration-2 case read
  gets the same claims for the requested fields of the pages it names.
- Neutral stubs: planner no routes, router every family, grader "supported", decision maker abstain.
- Every request a stub cannot answer (an unknown schema, stage, system prompt or page) is counted as unexpected and raised as an
  LLMError, so the agents take their ordinary fallback; an unexpected request fails the check.
The pipeline runs through `runner.run` (one worker) with its cache and outputs in a temporary directory that is deleted afterwards, and
its preds.csv rows are compared with the baseline path (`run_kaggle.assemble_case` over the same cached claims). A case whose readable
pages are not all in the baseline extraction cache is skipped (counted, never filled by a model).

This is a mechanics check, not an accuracy measurement: dev only, no label or gold file is read, nothing is compared to labels. The
output is counts and ids only (never page text, a claim value or a label). Differing rows are reported, not failed: the per-item
splitting of list fields may legitimately change a row. Exit 0 when at least one case ran, every case run finished, the run is not
partial and no model or network call was attempted; 1 otherwise.
"""
from __future__ import annotations

import csv
import json
import socket
import tempfile
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from assessor.llm import LLMError
from eval import run_kaggle as rk

from . import baseline as baseline_mod
from . import data as data_mod
from . import runner
from .cache import CallCache
from .config import FAMILIES, ROOT
from .nodes import Agents
from .prompts import SPECIALIST_SYSTEM, page_user_prompt
from .safe import safe_id
from .schemas import DecisionOut, GradeOut, PlanOut, RouterOut, SpecialistRaw

LIST_FIELDS = frozenset({"security_certifications", "subprocessors", "security_incidents"})
_CASE_HEAD = "Report only these fields: "
_FAMILY_OF_SYSTEM = {v: k for k, v in SPECIALIST_SYSTEM.items()}


def add_arguments(p) -> None:
    p.add_argument("--data", default=str(ROOT / "data_kaggle"))
    p.add_argument("--split", choices=["train"], default="train", help="train only (dev is the smoke subset)")
    p.add_argument("--subset", choices=["dev"], default="dev", help="dev only: never holdout or test")
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--only", default="", help="comma-separated dev case ids")


class Unexpected(LLMError):
    """A stub was asked for something the replay cannot answer (in a real run it would be a model call). An LLMError, so every agent
    takes its ordinary fallback; the tally counts it and the check fails. The message holds no quota wording."""


class Tally:
    def __init__(self):
        self._lock = threading.Lock()
        self.answers: Counter = Counter()
        self.unexpected: Counter = Counter()

    def answer(self, role: str) -> None:
        with self._lock:
            self.answers[role] += 1

    def refuse(self, role: str):
        with self._lock:
            self.unexpected[role] += 1
        return Unexpected("selfcheck stub: unexpected request")


def specialist_items(readings: list[dict]) -> list[dict]:
    """Baseline readings -> the specialists' reply items: a non-empty list value of a list field becomes one item per reading; an empty
    list (e.g. "no incidents") and every other value stay one reading."""
    out = []
    for r in readings:
        base = {"page_id": r["page_id"], "field": r["field"], "quote": r.get("quote") or "", "display": r.get("display") or ""}
        values = r["value"] if r["field"] in LIST_FIELDS and isinstance(r["value"], list) and r["value"] else [r["value"]]
        out += [{**base, "value": v} for v in values]
    return out


class PlannerStub:
    model = "selfcheck-planner"

    def __init__(self, tally: Tally | None = None):
        self.tally = tally or Tally()

    def structured(self, system, user, schema, *args, stage: str = "", **kw):
        if schema is not PlanOut or stage != "planner":
            raise self.tally.refuse("planner")
        self.tally.answer("planner")
        return PlanOut(routes=[])                       # no routes: the reviewer's rules decide


class LocalStub:
    model = "selfcheck-local"

    def __init__(self, tally: Tally | None = None):
        self.tally = tally or Tally()

    def structured(self, system, user, schema, *args, stage: str = "", **kw):
        if schema is RouterOut and stage == "router":
            self.tally.answer("router")
            return RouterOut(families=list(FAMILIES))
        if schema is GradeOut and stage == "grader":
            self.tally.answer("grader")
            return GradeOut(grade="supported")
        if schema is DecisionOut and stage == "decision":
            self.tally.answer("decision")
            return DecisionOut(choice=None)
        raise self.tally.refuse("local")


class ReplayReader:
    """The specialists' model: answers from the baseline's cached claims (see the module docstring)."""
    model = "selfcheck-replay-reader"

    def __init__(self, by_prompt: dict[str, str], items: dict[str, list[dict]], pages: dict[str, dict], tally: Tally | None = None):
        self.by_prompt, self.items, self.pages, self.tally = by_prompt, items, pages, tally or Tally()

    def _for(self, pid: str, fields) -> list[dict]:
        return [dict(i, page_id=pid) for i in self.items.get(pid, []) if i["field"] in fields]

    def structured(self, system, user, schema, *args, stage: str = "", **kw):
        family = _FAMILY_OF_SYSTEM.get(system)
        if schema is not SpecialistRaw or family is None or stage != f"specialist:{family}":
            raise self.tally.refuse("specialist")
        if user in self.by_prompt:
            self.tally.answer("specialist")
            return SpecialistRaw(readings=self._for(self.by_prompt[user], FAMILIES[family]))
        if user.startswith(_CASE_HEAD):
            from .agents.specialists import case_user_prompt
            fields = user.split("\n", 1)[0][len(_CASE_HEAD):].split(", ")
            pids = [line[len("page_id: "):] for line in user.split("\n") if line.startswith("page_id: ")]
            known = [p for p in pids if p in self.pages]
            # exact: the prompt must be the one the specialists build for exactly these pages and fields
            if len(known) == len(pids) and pids and case_user_prompt([self.pages[p] for p in pids], fields) == user:
                self.tally.answer("specialist")
                return SpecialistRaw(readings=[i for p in pids for i in self._for(p, fields)])
        raise self.tally.refuse("specialist")


class _NoCall:
    def __init__(self, model: str):
        self.model = model

    def structured(self, *a, **kw):
        raise RuntimeError("the baseline path never calls a model")


@contextmanager
def no_network():
    """Refuse and count every socket connection attempt while the check runs (a counter dict: {"attempts": n})."""
    counter = {"attempts": 0}
    lock = threading.Lock()
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def refuse(self, *a, **kw):
        with lock:
            counter["attempts"] += 1
        raise OSError("selfcheck: network disabled")
    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    try:
        yield counter
    finally:
        socket.socket.connect, socket.socket.connect_ex = real_connect, real_connect_ex


def _rows(cid: str, verdicts: dict, overall) -> dict[str, str]:
    """The runner's preds.csv rows of one case: `<case>__<requirement>` in rules_lite order, then `<case>__OVERALL`."""
    return {**{f"{cid}__{rid}": v for rid, v in verdicts.items()}, f"{cid}__OVERALL": overall}


def check(data_dir, *, split: str = "train", subset: str = "dev", limit: int | None = 3, only: list[str] | None = None, rl, reqs,
          baseline_model: str, baseline_dir, say=print) -> dict:
    """Run the check and print its counts; returns them with "exit" (0 ok, 1 failed)."""
    t0 = time.perf_counter()
    with no_network() as net:
        res = _check(Path(data_dir), split, subset, limit, only, rl, reqs, baseline_model, Path(baseline_dir))
    res["network"] = net["attempts"]
    res["wall_seconds"] = round(time.perf_counter() - t0, 3)
    reasons = [name for name, bad in (("no case ran", res["cases_run"] == 0), ("case errors", res["errors"] > 0),
                                      ("unfinished cases", res["unfinished"] > 0), ("partial run", res["partial"]),
                                      ("unexpected stub requests", res["unexpected"] > 0), ("network attempts", res["network"] > 0),
                                      ("ambiguous page ids", res["ambiguous"] > 0)) if bad]
    res["exit"] = 1 if reasons else 0
    res["reasons"] = reasons
    for line in _format(res, split, subset, limit):
        say(line)
    return res


def _check(data_dir: Path, split, subset, limit, only, rl, reqs, model: str, baseline_dir: Path) -> dict:
    cases, pages, ids = data_mod.load(data_dir, split, subset, limit, only)
    ids = sorted(ids)
    system = baseline_mod.baseline_prompt()
    extractor = rk.Extractor(_NoCall(model), baseline_dir, True, system=system) if baseline_dir.is_dir() else None  # init mkdirs
    covered, skipped, missing_pages = [], [], 0
    base_rows: dict[str, str] = {}
    by_prompt: dict[str, str] = {}
    items: dict[str, list[dict]] = {}
    page_of: dict[str, dict] = {}
    ambiguous = 0
    for cid in ids:
        recs = rk.triage(cases[cid], pages[cid])
        readable = [r for r in recs if rk.readable(r)]
        miss = len(readable) if extractor is None else 0
        for r in readable if extractor is not None else []:
            try:
                r["extract"] = extractor.run(r["page"])
            except Exception:                           # noqa: BLE001  a corrupt cache file: that page is not covered
                r["extract"] = {"status": "unreadable", "claims": []}
            miss += r["extract"].get("status") != "ok"
        if miss:
            skipped.append(cid)
            missing_pages += miss
            continue
        base = rk.assemble_case(cases[cid], recs, reqs, rl)
        base_rows.update(_rows(cid, base["verdicts"], base["overall"]))
        readings = baseline_mod.baseline_readings(model, recs, baseline_dir, system=system)
        for r in readable:
            pid = str(r["page"]["page_id"])
            got = specialist_items(readings.get(pid, []))
            if pid in items and (items[pid] != got or page_of[pid] != r["page"]):
                ambiguous += 1
            items.setdefault(pid, got)
            page_of.setdefault(pid, r["page"])
            by_prompt.setdefault(page_user_prompt(r["page"]), pid)
        covered.append(cid)

    res = {"cases_selected": len(ids), "cases_run": 0, "skipped": skipped, "missing_pages": missing_pages, "errors": 0,
           "error_frames": {}, "unfinished": 0, "partial": False, "rows": 0, "equal": 0, "differing": [], "verdicts": {},
           "overall": {}, "iteration2_cases": 0, "answers": {}, "unexpected": 0, "ambiguous": ambiguous}
    if not covered:
        return res
    tally = Tally()
    with tempfile.TemporaryDirectory(prefix="vv-selfcheck-") as tmp:
        ag = Agents(main_llm=PlannerStub(tally), reader_llm=ReplayReader(by_prompt, items, page_of, tally), local_llm=LocalStub(tally),
                    cache=CallCache(Path(tmp) / "cache"), rl=rl, reqs=reqs, baseline_dir=baseline_dir, baseline_model=model)
        rj = runner.run(data_dir, split, subset, agents=ag, out=Path(tmp) / "out", only=covered, workers=1, say=lambda s: None)
        with (Path(tmp) / "out" / "preds.csv").open(encoding="utf-8") as f:
            preds = {r["id"]: r["verdict"] for r in csv.DictReader(f)}
    keys = sorted(set(base_rows) | set(preds))
    res.update(cases_run=len(covered), errors=rj["cases"]["errors"], error_frames=rj["error_frames"],
               unfinished=rj["cases"]["unfinished"], partial=bool(rj["partial"]), rows=len(preds),
               equal=sum(1 for k in keys if k in preds and preds.get(k) == base_rows.get(k)),
               differing=[safe_id(k) for k in keys if preds.get(k) != base_rows.get(k)],
               verdicts=dict(sorted(Counter(v for k, v in preds.items() if not k.endswith("__OVERALL")).items())),
               overall=dict(sorted(Counter(v for k, v in preds.items() if k.endswith("__OVERALL")).items())),
               iteration2_cases=rj["iteration2_cases"], answers=dict(sorted(tally.answers.items())),
               unexpected=sum(tally.unexpected.values()), unexpected_by_stub=dict(sorted(tally.unexpected.items())),
               flags={k: v for k, v in rj["flags"].items() if v}, partial_reasons=rj["partial_reasons"])
    return res


def _format(res: dict, split: str, subset: str, limit) -> list[str]:
    lines = [f"selfcheck {split}/{subset} (limit {limit or 'none'}): the real graph, rules_lite and requirements; replay readers; "
             "no model, no network",
             f"  cases selected {res['cases_selected']}, run {res['cases_run']}, skipped (no baseline cache) {len(res['skipped'])}"
             + (f" {res['skipped']} ({res['missing_pages']} readable pages not cached)" if res["skipped"] else "")
             + f", errors {res['errors']}, unfinished {res['unfinished']}"]
    if res["error_frames"]:
        lines.append(f"  error frames (repo path:line): {json.dumps(res['error_frames'])}")
    if res["cases_run"]:
        lines += [f"  rows {res['rows']}; verdicts {json.dumps(res['verdicts'])}; overall {json.dumps(res['overall'])}",
                  f"  versus the baseline path: equal {res['equal']}, differing {len(res['differing'])}"
                  + (f" {res['differing']}" if res["differing"] else ""),
                  f"  iteration 2 reached in {res['iteration2_cases']} of {res['cases_run']} cases; node flags "
                  f"{json.dumps(res.get('flags') or {})}; partial {res['partial']}",
                  f"  stub answers {json.dumps(res['answers'])}"]
    lines.append(f"  model or network calls attempted: {res['unexpected'] + res['network']} (unexpected stub requests "
                 f"{res['unexpected']}, network connects {res['network']}); ambiguous page ids {res['ambiguous']}")
    lines.append(f"  wall seconds {res['wall_seconds']:.1f}")
    lines.append("selfcheck OK" if res["exit"] == 0 else "selfcheck FAILED: " + ", ".join(res["reasons"]))
    return lines


def main(a) -> int:
    """The `selfcheck` subcommand: the real rules_lite and requirements from `--data`, the baseline model named as `build_agents`
    names it (the extractor's model: it is part of the baseline cache key). Builds no model that is ever called."""
    runner.disable_assessor_cache()
    data_dir = Path(a.data)
    only = [c for c in a.only.split(",") if c] or None
    with no_network() as net:
        from assessor.llm import LLM
        reader = LLM.extractor()                        # its name only: never called
        model, mock = reader.model, reader.mock_mode
    if net["attempts"]:
        print(f"selfcheck FAILED: {net['attempts']} network attempts while naming the baseline model")
        return 1
    if mock:
        print("selfcheck FAILED: the extractor is on the mock provider, so the baseline cache cannot be named: set the real provider "
              "(via .env); no call is made")
        return 1
    reqs = json.loads((data_dir / "requirements.json").read_text(encoding="utf-8"))
    rl = rk.load_module("kaggle_rules_lite", data_dir / "rules_lite.py")
    print(f"baseline model {model}")
    res = check(data_dir, split=a.split, subset=a.subset, limit=a.limit or None, only=only, rl=rl, reqs=reqs, baseline_model=model,
                baseline_dir=data_dir / "cache" / "extract")
    return res["exit"]
