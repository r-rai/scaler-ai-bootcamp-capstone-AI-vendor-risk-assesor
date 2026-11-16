"""
detectors.py — find issues: pytest failures, runtime errors, eval regressions, manual reports.

Test failures are GROUPED by root-cause signature so 65 failing tests become a handful of issues.
Every subprocess has a timeout; a hang or crash becomes an issue instead of killing the CLI.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from assessor.db import get_failed_jobs
from assessor.errors import ERRORS_FILE, mask_secrets, read_errors

from .models import Issue
from .tools import mock_env

ROOT = Path(__file__).resolve().parent.parent
PYTEST_TIMEOUT = 900
EVAL_TIMEOUT = 900
MAX_EVIDENCE = 4000
STATE_FILE = ROOT / "runs" / "issues" / "state.json"
_FRAME_RE = re.compile(r"^(?P<file>[\w./-]+\.py):(?P<line>\d+): (?:in (?P<func>\w+)|(?P<exc>\w+(?:Error|Exception|Exit)\b.*))", re.M)


def _id(kind: str, signature: str) -> str:
    return hashlib.sha1(f"{kind}|{signature}".encode()).hexdigest()[:10]


def _norm_msg(msg: str) -> str:
    msg = re.sub(r"0x[0-9a-f]+|\b\d+\b", "N", msg)
    msg = re.sub(r"\s+", " ", msg)
    return msg[:80]


def _clip(text: str, n: int = MAX_EVIDENCE) -> str:
    text = mask_secrets(text)
    return text if len(text) <= n else text[: n // 2] + "\n...[clipped]...\n" + text[-n // 2:]


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def mark_fixed(issue: Issue, commit: str) -> None:
    st = _load_state()
    st[issue.id] = {"commit": commit, "title": issue.title}
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(st, indent=2), encoding="utf-8")


def _annotate(issue: Issue) -> Issue:
    prev = _load_state().get(issue.id)
    if prev:
        issue.note = f"previously fixed in {prev['commit'][:8]} — possibly fixed / regressed"
    return issue


# ---------------------------------------------------------------------------- pytest
def _run(cmd: list[str], env: dict, timeout: int) -> tuple[int, str, str, bool]:
    try:
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr, False
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return -1, out, "", True


def run_pytest(node_ids: list[str] | None = None, *, live_env: bool = False, root: Path | None = None,
               timeout: int = PYTEST_TIMEOUT) -> dict:
    """Run pytest; return {"failures": {nodeid: {...}}, "rc": int, "timeout": bool, "stderr": str}."""
    import os
    env = dict(os.environ) if live_env else mock_env()
    cwd_root = root or ROOT
    with tempfile.TemporaryDirectory() as td:
        xml = Path(td) / "junit.xml"
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short", "--continue-on-collection-errors", f"--junitxml={xml}"]
        cmd += node_ids or []
        try:
            r = subprocess.run(cmd, cwd=cwd_root, env=env, capture_output=True, text=True, timeout=timeout)
            rc, out, err, to = r.returncode, r.stdout, r.stderr, False
        except subprocess.TimeoutExpired as exc:
            rc, out, err, to = -1, "", "", True
        failures: dict[str, dict] = {}
        if xml.exists():
            try:
                for tc in ET.parse(xml).getroot().iter("testcase"):
                    bad = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
                    if bad is None:
                        continue
                    nodeid = _nodeid(tc.get("classname", ""), tc.get("name", "?"), cwd_root)
                    failures[nodeid] = {"type": bad.get("type") or "Error", "message": bad.get("message") or "",
                                        "text": bad.text or ""}
            except ET.ParseError:
                pass
    return {"failures": failures, "rc": rc, "timeout": to, "stderr": (err or "")[-3000:], "stdout": (out or "")[-3000:]}


def _nodeid(classname: str, name: str, root: Path) -> str:
    """junit classname 'tests.test_x.TestC' -> 'tests/test_x.py::TestC::name' (longest existing file prefix wins)."""
    parts = [p for p in classname.split(".") if p]
    for i in range(len(parts), 0, -1):
        if (root / ("/".join(parts[:i]) + ".py")).is_file():
            return "::".join(["/".join(parts[:i]) + ".py", *parts[i:], name])
    return name if not parts else f"{'/'.join(parts)}.py::{name}"


def _signature(f: dict) -> str:
    frames = _FRAME_RE.findall(f["text"])
    files = [fr[0] for fr in frames if fr[0] and ".venv" not in fr[0] and "site-packages" not in fr[0]]
    non_test = [x for x in files if not x.startswith("tests/")]
    where = (non_test or files or ["?"])[-1]
    return f"{f['type']}|{where}|{_norm_msg(f['message'])}"


def detect_test_failures(*, live_env: bool = False, rerun_flaky: bool = True) -> list[Issue]:
    res = run_pytest(live_env=live_env)
    issues: list[Issue] = []
    if res["timeout"]:
        sig = "pytest|timeout"
        return [_annotate(Issue(id=_id("test", sig), kind="test", title=f"pytest timed out after {PYTEST_TIMEOUT}s",
                                signature=sig, evidence="The full suite hung. Find which test or code path loops/blocks."))]
    fails = res["failures"]
    if not fails and res["rc"] not in (0, 1, 5):   # 2 interrupted, 3 internal, 4 usage error
        sig = f"pytest|rc{res['rc']}"
        return [_annotate(Issue(id=_id("test", sig), kind="test", title=f"pytest could not run (exit {res['rc']})",
                                signature=sig, evidence=_clip(res["stderr"] + "\n" + res["stdout"])))]
    flaky_ids: set[str] = set()
    if fails and rerun_flaky and len(fails) <= 100:
        again = run_pytest(list(fails), live_env=live_env)
        if not again["timeout"]:
            flaky_ids = {n for n in fails if n not in again["failures"]}
    groups: dict[str, list[str]] = defaultdict(list)
    for nodeid, f in fails.items():
        groups[_signature(f)].append(nodeid)
    for sig, ids in groups.items():
        first = fails[ids[0]]
        is_flaky = all(i in flaky_ids for i in ids)
        title = f"{first['type'].split('.')[-1]}: {first['message'][:90] or sig} ({len(ids)} test{'s' if len(ids) != 1 else ''})"
        issues.append(_annotate(Issue(id=_id("test", sig), kind="test", title=title, signature=sig,
                                      evidence=_clip(first["text"]), related=sorted(ids), flaky=is_flaky)))
    return issues


# ---------------------------------------------------------------------------- runtime
def detect_runtime_errors(errors_path: Path | None = None) -> list[Issue]:
    from assessor.guardrails import redact_input
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in read_errors(errors_path or ERRORS_FILE):
        groups[f"{e.get('source')}|{e.get('type')}|{_norm_msg(e.get('message', ''))}"].append(e)
    try:
        for j in get_failed_jobs():
            groups[f"job|{_norm_msg(j.get('error') or '')}"].append(
                {"source": "api:job", "type": "JobFailed", "message": j.get("error") or "", "traceback": "", "context": {"job_id": j["job_id"]}})
    except Exception:  # noqa: BLE001 — db missing/corrupt must not break scanning
        pass
    out = []
    for sig, items in groups.items():
        last = items[-1]
        tb, _ = redact_input(last.get("traceback") or last.get("message", ""))
        ctx = json.dumps(last.get("context", {}), default=str)
        files = sorted(set(re.findall(r'File "([^"]+\.py)"', last.get("traceback", ""))))
        rel = []
        for f in files:
            try:
                rel.append(Path(f).resolve().relative_to(ROOT).as_posix())
            except ValueError:
                pass
        out.append(_annotate(Issue(
            id=_id("runtime", sig), kind="runtime",
            title=f"{last.get('type')} in {last.get('source')}: {last.get('message', '')[:80]} (x{len(items)})",
            signature=sig, evidence=_clip(f"{tb}\ncontext: {ctx}"), related=[r for r in rel if ".venv" not in r])))
    return out


# ---------------------------------------------------------------------------- eval
def detect_eval_regressions(*, live_env: bool = False) -> list[Issue]:
    import os
    env = dict(os.environ) if live_env else mock_env()
    rc, out, err, to = _run([sys.executable, "eval/run_eval.py", "--strict", "--real", "--ablations"], env, EVAL_TIMEOUT)
    if to:
        sig = "eval|timeout"
        return [_annotate(Issue(id=_id("eval", sig), kind="eval", title="Eval run timed out", signature=sig, evidence=_clip(out)))]
    m = re.search(r"FAILED thresholds:\s*(.+)", out)
    if m:
        raw = m.group(1).strip()
        return [_annotate(Issue(id=_id("eval", raw), kind="eval", title=f"Eval below threshold: {raw[:100]}", signature=f"eval|{raw}",
                                evidence=_clip(out[-MAX_EVIDENCE:]), related=["eval/run_eval.py", "reports/eval_report.md"]))]
    if rc != 0:
        sig = f"eval|rc{rc}|{_norm_msg(err.strip().splitlines()[-1] if err.strip() else '')}"
        return [_annotate(Issue(id=_id("eval", sig), kind="eval", title=f"Eval crashed (exit {rc})", signature=sig,
                                evidence=_clip(err + "\n" + out)))]
    return []


# ---------------------------------------------------------------------------- manual
def manual_issue(text: str) -> Issue:
    text = text.strip()
    return Issue(id=_id("manual", text), kind="manual", title=text.splitlines()[0][:100] or "manual report",
                 signature=f"manual|{text[:200]}", evidence=_clip(text))
