"""
scripts/bugfix_agent.py — agentic bug fixing with a human approval gate.

  python scripts/bugfix_agent.py scan                      # list issues (tests grouped by root cause, runtime, eval)
  python scripts/bugfix_agent.py fix                       # pick an issue from the scan
  python scripts/bugfix_agent.py fix <issue-id>            # diagnose one issue
  python scripts/bugfix_agent.py fix --manual "describe the bug"

Flow: detect -> DIAGNOSE (read-only agent) -> you approve / reject / give feedback -> git branch ->
FIX (edit agent, approved files only) -> verify against the pre-fix baseline -> you commit or discard.
There is no auto-approve option.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assessor.config import RUNS_DIR, SETTINGS  # noqa: E402
from assessor.llm import LLM  # noqa: E402
from assessor.tracing import Tracer  # noqa: E402
from bugfix import detectors, gitops  # noqa: E402
from bugfix.agent import apply_fix, diagnose, probe_tools  # noqa: E402
from bugfix.models import FixProposal, Issue  # noqa: E402
from bugfix.tools import is_protected  # noqa: E402
from bugfix.verify import scan_diff, verify  # noqa: E402

SETTINGS.verbose = False
NOTICE = RUNS_DIR / "bugfix" / ".notice_shown"


def ask(prompt: str, choices: str) -> str:
    while True:
        try:
            v = input(prompt).strip().lower()
        except EOFError:
            return "r" if "r" in choices else choices[0]
        if v[:1] in choices:
            return v[:1]
        print(f"  please answer one of: {', '.join(choices)}")


def gather(args) -> list[Issue]:
    issues: list[Issue] = []
    want_all = not (args.tests or args.runtime or args.eval)
    live = getattr(args, "live_tests", False)
    if live:
        print("! --live-tests: tests/eval run with your real .env provider (this can cost money).")
    if want_all or args.tests:
        print("Running pytest (offline mock mode) ...")
        issues += detectors.detect_test_failures(live_env=live)
    if want_all or args.runtime:
        issues += detectors.detect_runtime_errors()
    if want_all or args.eval:
        print("Running eval (strict) ...")
        issues += detectors.detect_eval_regressions(live_env=live)
    return issues


def print_issues(issues: list[Issue]) -> None:
    if not issues:
        print("No issues found.")
        return
    print(f"\n{len(issues)} issue(s):\n")
    for n, i in enumerate(issues, 1):
        tags = (" [flaky]" if i.flaky else "") + (f" [{i.note}]" if i.note else "")
        print(f"  {n:>2}. {i.id}  {i.kind:<7} {i.title}{tags}")


def fingerprint(files: list[str]) -> dict:
    fp = {"head": gitops.head(ROOT)}
    for f in files:
        p = ROOT / f
        fp[f] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    return fp


def show_proposal(p: FixProposal) -> None:
    print("\n" + "=" * 78)
    print(f"Fix type : {p.fix_type}    Confidence: {p.confidence:.0%}    Risk: {p.risk or '-'}")
    if p.confidence < 0.5:
        print("!! LOW CONFIDENCE — treat this proposal with extra suspicion.")
    print(f"Root cause: {p.root_cause}")
    for label, items in (("Evidence", p.evidence), ("Plan", p.plan), ("Edit files", p.files_to_edit),
                         ("New files", p.new_files), ("Verify with", p.verification_tests)):
        if items:
            print(f"{label}:")
            for x in items:
                print(f"  - {x}")
    print("=" * 78)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Agentic bug fixing (human-approved)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scan", "fix"):
        sp = sub.add_parser(name)
        sp.add_argument("--tests", action="store_true")
        sp.add_argument("--runtime", action="store_true")
        sp.add_argument("--eval", action="store_true")
        sp.add_argument("--live-tests", action="store_true", help="run tests/eval with the real .env provider instead of mock")
    fx = sub.choices["fix"]
    fx.add_argument("issue_id", nargs="?")
    fx.add_argument("--manual", help="describe a bug in text")
    fx.add_argument("--include-flaky", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "scan":
        print_issues(gather(a))
        return 0

    # ---------------------------------------------------------------- fix
    if not sys.stdin.isatty():
        print("`fix` needs an interactive terminal: approval cannot be given without one. (`scan` works anywhere.)")
        return 2
    try:
        gitops.preconditions(ROOT)
    except gitops.GitError as exc:
        print(f"Cannot start: {exc}")
        return 2

    if a.manual:
        issue = detectors.manual_issue(a.manual)
    else:
        issues = gather(a)
        print_issues(issues)
        if not issues:
            return 0
        pick = next((i for i in issues if i.id == a.issue_id), None) if a.issue_id else None
        if a.issue_id and not pick:
            print(f"No issue with id {a.issue_id}.")
            return 2
        if not pick:
            try:
                n = int(input("\nIssue number to fix (blank to quit): ").strip() or 0)
            except (ValueError, EOFError):
                n = 0
            if not 1 <= n <= len(issues):
                return 0
            pick = issues[n - 1]
        issue = pick
    if issue.flaky and not a.include_flaky:
        print("That issue is flaky (passed on re-run), so a code fix is unlikely to help. Use --include-flaky to override.")
        return 0

    tracer = Tracer(f"bugfix-{issue.id}", root=RUNS_DIR / "bugfix")
    tracer.write_json("issue.json", issue.model_dump())
    llm = LLM.for_bugfix(tracer=tracer)
    if not NOTICE.exists():
        print(f"Note: repository code and test output are sent to the '{llm.provider}' model ({llm.model}) "
              "(secrets are masked; .env is never read).")
        NOTICE.parent.mkdir(parents=True, exist_ok=True)
        NOTICE.write_text("shown")
    if llm.mock_mode:
        print("The bug-fix agent needs a real tool-calling model, but the provider is 'mock'.\n"
              "Set LLM_PROVIDER (or BUGFIX_LLM_PROVIDER / BUGFIX_LLM_MODEL) in .env. Tests/eval still run in mock mode.")
        return 2
    problem = probe_tools(llm)
    if problem:
        print(problem + "\nSet BUGFIX_LLM_PROVIDER / BUGFIX_LLM_MODEL to a model with tool-calling support.")
        return 2

    feedback: str | None = None
    while True:  # ---------------- diagnose / approve loop
        print(f"\nDiagnosing {issue.id}: {issue.title}\n")
        proposal, loop, errs = diagnose(issue, llm, ROOT, feedback=feedback, tracer=tracer,
                                        max_steps=SETTINGS.bugfix_max_steps, on_step=lambda s: print(f"  · {s}"))
        tracer.write_json("proposal.json", proposal.model_dump())
        if loop.reason != "done":
            print(f"(investigation ended early: {loop.reason})")
        show_proposal(proposal)
        if proposal.fix_type == "config":
            print("This looks like an environment/config problem, not a code bug (the agent never edits .env):\n")
            print(proposal.config_instructions or "(no instructions given)")
            return 0
        if proposal.fix_type == "no_fix":
            print("The agent found nothing to change.")
            return 0
        if proposal.fix_type == "needs_info":
            print("The agent needs more information:")
            for q in proposal.questions:
                print(f"  ? {q}")
            feedback = input("Your answers (blank to stop): ").strip()
            if not feedback:
                return 0
            continue
        if errs:
            print("Proposal is invalid and cannot be approved:")
            for e in errs:
                print(f"  ! {e}")
            c = ask("[f]eedback and re-diagnose / [q]uit: ", "fq")
            if c == "q":
                return 1
            feedback = input("Feedback: ").strip()
            continue
        fp = fingerprint(proposal.files_to_edit)
        c = ask("[a]pprove / [r]eject / [f]eedback: ", "arf")
        if c == "r":
            print("Rejected. Nothing was changed.")
            return 0
        if c == "f":
            feedback = input("Feedback for the agent: ").strip()
            continue
        if fingerprint(proposal.files_to_edit) != fp:
            print("The repository changed since diagnosis; re-diagnosing.")
            continue
        break

    allow_protected = any(is_protected(f) for f in proposal.files_to_edit + proposal.new_files)
    if allow_protected:
        if ask("The plan edits tests or gold sets. Allow that? [y]es / [n]o: ", "yn") == "n":
            print("Aborted. Nothing was changed.")
            return 0

    try:
        gitops.preconditions(ROOT)
    except gitops.GitError as exc:
        print(f"Cannot create a fix branch: {exc}")
        return 2
    print("\nRecording the pre-fix baseline (full test suite) ...")
    base = detectors.run_pytest(live_env=a.live_tests)
    if base["timeout"]:
        print("Baseline run timed out; aborting before any edit.")
        return 2
    baseline = set(base["failures"])
    original = gitops.current_branch(ROOT)
    branch = gitops.create_branch(ROOT, issue.id)
    print(f"On branch {branch}. Running the fix agent ...\n")

    def scope_cb(path: str, reason: str) -> bool:
        print(f"\nThe fix agent asks to ALSO edit {path}\n  reason: {reason}")
        if is_protected(path):
            print("  (this is a tests/gold file)")
        return ask("Allow? [y]es / [n]o: ", "yn") == "y"

    # a second, fresh budget for the fix phase
    fix_llm = LLM.for_bugfix(tracer=tracer)
    sb = None
    try:
        result, sb = apply_fix(issue, proposal, fix_llm, ROOT, allow_protected=allow_protected, scope_cb=scope_cb,
                               tracer=tracer, max_steps=SETTINGS.bugfix_max_steps, on_step=lambda s: print(f"  · {s}"))
    except KeyboardInterrupt:
        print(f"\nInterrupted. Nothing was discarded. You are on branch {branch}.")
        print(f"Review: git status / git diff.  Restore: git checkout -- . && git checkout {original} && git branch -D {branch}")
        return 130

    edited, created = result.edited, result.created
    if loop_reason := result.loop_reason:
        if loop_reason != "done":
            print(f"\nFix agent stopped early: {loop_reason}")
    print("\nVerifying (targeted tests + full suite vs. baseline) ...")
    v = verify(issue, proposal, baseline, ROOT, live_env=a.live_tests)
    result.targeted_ok, result.new_failures = v["targeted_ok"], v["new_failures"]
    diff_text = gitops.diff(ROOT, edited + created)
    result.flags = scan_diff(diff_text)
    ok = result.status == "fixed" and v["targeted_ok"] is not False and not v["new_failures"] and not v["timeout"]
    result.status = "fixed" if ok else ("stopped" if result.status == "stopped" else "not_fixed")
    tracer.write_json("fix_result.json", {**result.model_dump(), "verify": v})

    print("\n" + "=" * 78)
    print(f"Result: {result.status.upper()}   targeted tests: {v['targeted_ok']}   new failures: {len(v['new_failures'])}")
    for n in v["notes"]:
        print(f"  note: {n}")
    for t in v["new_failures"][:10]:
        print(f"  NEW FAILURE: {t}")
    for t in v["still_failing"][:10]:
        print(f"  still failing: {t}")
    for f in result.flags:
        print(f"  !! REVIEW: {f}")
    print(gitops.diff(ROOT, edited + created, stat=True) or "(no file changes)")
    print("=" * 78)
    if not (edited or created):
        gitops.discard(ROOT, [], [], branch, original)
        print("The agent made no changes; branch removed.")
        return 1
    while True:
        c = ask("[c]ommit on the fix branch / [d]iscard / [s]how diff / [k]eep uncommitted: ", "cdsk")
        if c == "s":
            print(diff_text)
        else:
            break
    if c == "d":
        gitops.discard(ROOT, edited, created, branch, original)
        print(f"Discarded; back on {original}.")
        return 1
    if c == "k":
        print(f"Kept on branch {branch} (uncommitted). Restore with: git checkout -- . && git checkout {original} && git branch -D {branch}")
        return 0
    if not ok and ask("Verification did NOT pass. Commit anyway? [y]es / [n]o: ", "yn") == "n":
        print(f"Left uncommitted on {branch}.")
        return 1
    try:
        sha = gitops.commit(ROOT, edited + created, f"bugfix({issue.kind}): {issue.title[:60]}\n\nIssue {issue.id}. Proposed by the bugfix agent, approved by a human.")
    except gitops.GitError as exc:
        print(f"Commit failed: {exc}\nChanges remain on {branch}.")
        return 1
    detectors.mark_fixed(issue, sha)
    print(f"Committed {sha[:8]} on {branch}. Merge when satisfied:  git checkout {original} && git merge {branch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
