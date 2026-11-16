"""Offline tests for the agentic bug-fixing toolkit (bugfix/ + assessor/errors.py). No network, no live LLM."""

import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from assessor.errors import mask_secrets, read_errors, record_error
from assessor.llm import LLM
from bugfix import detectors, gitops
from bugfix.agent import apply_fix, diagnose, run_tool_loop, validate_proposal
from bugfix.models import FixProposal, Issue
from bugfix.tools import Sandbox
from bugfix.verify import scan_diff


def _mock_llm() -> LLM:
    return LLM(provider="mock")


def _project(files: dict[str, str]) -> tempfile.TemporaryDirectory:
    td = tempfile.TemporaryDirectory()
    for rel, text in files.items():
        p = Path(td.name) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return td


def _git_repo(td: str) -> Path:
    root = Path(td)
    for args in (["init", "-q", "-b", "main"], ["config", "user.name", "t"], ["config", "user.email", "t@example.com"]):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    (root / ".gitignore").write_text("secret.log\n")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True, capture_output=True)
    return root


# ------------------------------------------------------------------ errors
def test_record_error_writes_masks_and_skips_corrupt_lines():
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "errors.jsonl"
        try:
            raise ValueError("boom sk-ABCDEFGHIJKLMNOP1234")
        except ValueError as exc:
            record_error("test", exc, {"k": "v"}, path=f)
        with open(f, "a") as fh:
            fh.write("{not json\n")
        errs = read_errors(f)
        assert len(errs) == 1 and errs[0]["type"] == "ValueError"
        assert "sk-ABCDEFGHIJKLMNOP1234" not in json.dumps(errs)


def test_record_error_never_raises():
    record_error("test", ValueError("x"), {"bad": object()}, path=Path("/proc/definitely/not/writable/e.jsonl"))


def test_mask_secrets_hides_env_values():
    os.environ["FAKE_TEST_API_KEY"] = "supersecretvalue123"
    try:
        assert "supersecretvalue123" not in mask_secrets("token is supersecretvalue123 ok")
    finally:
        del os.environ["FAKE_TEST_API_KEY"]


# ------------------------------------------------------------------ path guard & tools
def test_path_guard_denies_secrets_escapes_and_internal_dirs():
    with _project({".env": "KEY=1", "a.py": "x=1\n", ".git/config": "x"}) as root:
        sb = Sandbox(Path(root))
        for bad in (".env", "../etc/passwd", ".git/config", "/etc/passwd", "runs/x"):
            assert sb.dispatch("read_file", json.dumps({"path": bad}), edit=False).startswith("ERROR"), bad
        assert "x=1" in sb.dispatch("read_file", json.dumps({"path": "a.py"}), edit=False)


def test_symlink_escape_is_blocked():
    with _project({"a.py": "x=1\n"}) as root, tempfile.TemporaryDirectory() as outside:
        (Path(outside) / "s.txt").write_text("outside")
        os.symlink(Path(outside) / "s.txt", Path(root) / "link.txt")
        out = Sandbox(Path(root)).dispatch("read_file", json.dumps({"path": "link.txt"}), edit=False)
        assert out.startswith("ERROR") and "outside" not in out


def test_edit_only_allowlisted_files_and_protected_needs_consent():
    with _project({"src.py": "a = 1\n", "other.py": "b = 1\n", "tests/test_x.py": "def test_x():\n    assert True\n"}) as root:
        sb = Sandbox(Path(root), allowed_edit=["src.py", "tests/test_x.py"])
        assert sb.dispatch("replace_in_file", json.dumps({"path": "src.py", "old": "a = 1", "new": "a = 2"}), edit=True).startswith("Edited")
        assert sb.dispatch("replace_in_file", json.dumps({"path": "other.py", "old": "b = 1", "new": "b = 2"}), edit=True).startswith("ERROR")
        r = sb.dispatch("replace_in_file", json.dumps({"path": "tests/test_x.py", "old": "assert True", "new": "pass"}), edit=True)
        assert r.startswith("ERROR") and "protected" in r
        assert (Path(root) / "other.py").read_text() == "b = 1\n"


def test_replace_requires_unique_match_and_reports_syntax_errors():
    with _project({"a.py": "x = 1\nx = 1\n", "b.py": "y = 1\n"}) as root:
        sb = Sandbox(Path(root), allowed_edit=["a.py", "b.py"])
        assert "matched 2 times" in sb.dispatch("replace_in_file", json.dumps({"path": "a.py", "old": "x = 1", "new": "x = 2"}), edit=True)
        r = sb.dispatch("replace_in_file", json.dumps({"path": "b.py", "old": "y = 1", "new": "y = ("}), edit=True)
        assert "SYNTAX ERROR" in r


def test_write_file_only_new_approved_files():
    with _project({"a.py": "x=1\n"}) as root:
        sb = Sandbox(Path(root), allowed_edit=["a.py"], allowed_new=["new.py"])
        assert sb.dispatch("write_file", json.dumps({"path": "new.py", "content": "z = 1\n"}), edit=True).startswith("Created")
        assert sb.dispatch("write_file", json.dumps({"path": "new.py", "content": "z = 2\n"}), edit=True).startswith("ERROR")
        assert sb.dispatch("write_file", json.dumps({"path": "a.py", "content": "q"}), edit=True).startswith("ERROR")
        assert sb.created == ["new.py"]


def test_dispatch_handles_malformed_calls_without_raising():
    with _project({"a.py": "x=1\n"}) as root:
        sb = Sandbox(Path(root))
        assert sb.dispatch("nope", "{}", edit=False).startswith("ERROR: unknown tool")
        assert sb.dispatch("read_file", "{not json", edit=False).startswith("ERROR")
        assert sb.dispatch("read_file", "[1,2]", edit=False).startswith("ERROR")
        assert sb.dispatch("read_file", json.dumps({"wrong": 1}), edit=False).startswith("ERROR")
        assert sb.dispatch("replace_in_file", "{}", edit=False).startswith("ERROR: unknown tool")  # edit tools hidden when read-only


def test_tool_output_is_masked_wrapped_and_truncated():
    os.environ["FAKE_TEST_API_KEY"] = "supersecretvalue123"
    try:
        with _project({"a.py": "k = 'supersecretvalue123'\n" + "# pad\n" * 5000}) as root:
            out = Sandbox(Path(root)).dispatch("grep", json.dumps({"pattern": "supersecret|pad"}), edit=False)
            assert "supersecretvalue123" not in out and out.startswith("<untrusted_data")
            assert len(out) < 9500
    finally:
        del os.environ["FAKE_TEST_API_KEY"]


def test_run_tests_rejects_bad_ids():
    with _project({"a.py": ""}) as root:
        sb = Sandbox(Path(root))
        for bad in (["--help"], ["a.py"], [], ["tests/../../x.py; rm -rf /"], "tests/x.py"):
            try:
                out = sb.dispatch("run_tests", json.dumps({"node_ids": bad}), edit=False)
            except Exception:  # noqa: BLE001
                raise AssertionError(f"dispatch raised for {bad}")
            assert out.startswith("ERROR"), bad


# ------------------------------------------------------------------ agent loop
def _turns(*calls):
    return iter(calls)


def test_loop_stops_on_identical_repeated_calls():
    with _project({"a.py": "x=1\n"}) as root:
        call = {"tool_calls": [{"name": "read_file", "arguments": {"path": "a.py"}}]}
        r = run_tool_loop(_mock_llm(), "s", "u", Sandbox(Path(root)), edit=False, max_steps=20, mock_turns=iter([call] * 10))
        assert r.reason.startswith("loop")


def test_loop_stops_at_max_steps_and_survives_bad_calls():
    with _project({"a.py": "x=1\n"}) as root:
        turns = iter({"tool_calls": [{"name": "grep", "arguments": {"pattern": f"p{i}"}}]} for i in range(50))
        r = run_tool_loop(_mock_llm(), "s", "u", Sandbox(Path(root)), edit=False, max_steps=4, mock_turns=turns)
        assert r.reason == "max_steps" and r.steps == 4
        bad = iter([{"tool_calls": [{"name": "ghost", "arguments": "{oops"}]}, {"content": "finished"}])
        r2 = run_tool_loop(_mock_llm(), "s", "u", Sandbox(Path(root)), edit=False, max_steps=5, mock_turns=bad)
        assert r2.reason == "done" and r2.final_text == "finished"


# ------------------------------------------------------------------ proposals
def test_validate_proposal_rejects_bad_paths():
    with _project({"a.py": "x=1\n"}) as root:
        r = Path(root)
        ok = FixProposal(root_cause="c", files_to_edit=["a.py"], verification_tests=["tests/test_x.py::t"])
        assert validate_proposal(ok, r) == []
        bad = FixProposal(root_cause="c", files_to_edit=["missing.py", "../x.py", ".env"], new_files=["a.py"],
                          verification_tests=["rm -rf /"])
        errs = validate_proposal(bad, r)
        assert len(errs) >= 5
        assert validate_proposal(FixProposal(root_cause="c"), r)           # code fix with no files
        assert validate_proposal(FixProposal(fix_type="config", root_cause="c"), r) == []


def test_diagnose_then_apply_fix_end_to_end_in_mock_mode():
    with _project({"calc.py": "def add(a, b):\n    return a - b\n",
                   "tests/test_calc.py": "from calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"}) as td:
        root = Path(td)
        issue = Issue(id="abc", kind="manual", title="add is wrong", signature="s", evidence="add(1,2) != 3")
        proposal, loop, errs = diagnose(
            issue, _mock_llm(), root,
            mock_turns=_turns({"tool_calls": [{"name": "read_file", "arguments": {"path": "calc.py"}}]}, {"content": "add subtracts"}),
            proposal_mock=lambda: {"root_cause": "add subtracts", "files_to_edit": ["calc.py"],
                                   "verification_tests": ["tests/test_calc.py::test_add"], "confidence": 0.9})
        assert loop.reason == "done" and errs == [] and proposal.files_to_edit == ["calc.py"]
        res, sb = apply_fix(
            issue, proposal, _mock_llm(), root,
            mock_turns=_turns({"tool_calls": [{"name": "replace_in_file", "arguments": {"path": "calc.py", "old": "a - b", "new": "a + b"}}]},
                              {"tool_calls": [{"name": "replace_in_file", "arguments": {"path": "tests/test_calc.py", "old": "== 3", "new": "== 99"}}]},
                              {"content": "fixed"}))
        assert res.edited == ["calc.py"] and res.status == "fixed"
        assert "a + b" in (root / "calc.py").read_text()
        assert "== 3" in (root / "tests/test_calc.py").read_text()      # the test was NOT weakened


def test_cut_off_investigation_caps_confidence():
    with _project({"a.py": "x=1\n"}) as td:
        issue = Issue(id="z", kind="manual", title="t", signature="s")
        call = lambda i: {"tool_calls": [{"name": "grep", "arguments": {"pattern": f"p{i}"}}]}
        p, loop, _ = diagnose(issue, _mock_llm(), Path(td), max_steps=2, mock_turns=iter([call(0), call(1), call(2)]),
                              proposal_mock=lambda: {"root_cause": "x", "files_to_edit": ["a.py"], "confidence": 0.9})
        assert loop.reason == "max_steps" and p.confidence <= 0.4 and "not confirmed" in p.risk


# ------------------------------------------------------------------ cheating scanner
def test_scan_diff_flags_cheating_patterns():
    diff = ("+++ b/tests/test_a.py\n-    assert x == 1\n+    pass\n"
            "+++ b/app.py\n+@pytest.mark.skip\n+    except:\n+    if 'pytest' in sys.modules:\n")
    flags = scan_diff(diff)
    joined = " | ".join(flags)
    assert "removes an assertion" in joined and "skip" in joined and "bare" in joined and "pytest" in joined
    assert scan_diff("+++ b/app.py\n-    return a - b\n+    return a + b\n") == []


# ------------------------------------------------------------------ detectors
def test_nodeid_resolution():
    with _project({"tests/test_x.py": ""}) as root:
        r = Path(root)
        assert detectors._nodeid("tests.test_x", "test_a", r) == "tests/test_x.py::test_a"
        assert detectors._nodeid("tests.test_x.TestC", "test_a", r) == "tests/test_x.py::TestC::test_a"


def test_run_pytest_groups_collection_errors_and_failures():
    files = {"tests/test_a.py": "def test_ok():\n    assert True\n\ndef test_bad():\n    assert 1 == 2\n",
             "tests/test_broken.py": "import definitely_not_a_module_xyz\n\ndef test_never():\n    pass\n"}
    with _project(files) as root:
        res = detectors.run_pytest(root=Path(root), timeout=120)
        assert any(k.endswith("test_bad") for k in res["failures"])
        assert any("test_broken" in k for k in res["failures"])           # collection error is reported, not lost
        sigs = {detectors._signature(f) for f in res["failures"].values()}
        assert len(sigs) >= 2


def test_signature_normalises_numbers_so_variants_group():
    a = {"type": "AssertionError", "message": "assert 3 == 4", "text": "tests/test_x.py:10: in test_a\n    assert 3 == 4"}
    b = {"type": "AssertionError", "message": "assert 5 == 6", "text": "tests/test_x.py:10: in test_a\n    assert 5 == 6"}
    assert detectors._signature(a) == detectors._signature(b)


def test_runtime_detector_dedupes_and_ignores_missing_files():
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "e.jsonl"
        for i in range(3):
            try:
                raise KeyError(f"missing-{i}")
            except KeyError as exc:
                record_error("cli:x", exc, {"n": i}, path=f)
        issues = detectors.detect_runtime_errors(f)
        mine = [i for i in issues if i.signature.startswith("cli:x")]
        assert len(mine) == 1 and "x3" in mine[0].title
        assert detectors.detect_runtime_errors(Path(td) / "nope.jsonl") is not None


# ------------------------------------------------------------------ git
def test_git_preconditions_dirty_tree_and_branch_flow():
    with _project({"a.py": "x=1\n"}) as td:
        root = _git_repo(td)
        gitops.preconditions(root)                                    # clean: fine
        (root / "a.py").write_text("x=2\n")
        try:
            gitops.preconditions(root)
            raise AssertionError("dirty tree should be refused")
        except gitops.GitError as exc:
            assert "uncommitted" in str(exc)
        subprocess.run(["git", "checkout", "--", "a.py"], cwd=root, check=True)
        b1 = gitops.create_branch(root, "abc")
        subprocess.run(["git", "checkout", "main"], cwd=root, check=True, capture_output=True)
        b2 = gitops.create_branch(root, "abc")                        # name collision -> suffix
        assert b1 == "bugfix/abc" and b2 != b1


def test_not_a_repo_is_refused():
    with _project({"a.py": ""}) as td:
        try:
            gitops.preconditions(Path(td))
            raise AssertionError("expected GitError")
        except gitops.GitError as exc:
            assert "git init" in str(exc)


def test_discard_only_touches_agent_files():
    with _project({"a.py": "x=1\n", "keep.py": "k=1\n"}) as td:
        root = _git_repo(td)
        branch = gitops.create_branch(root, "d1")
        (root / "a.py").write_text("x=999\n")                         # agent edit
        (root / "new.py").write_text("n=1\n")                         # agent-created
        (root / "secret.log").write_text("ignored user file\n")       # not the agent's: must survive
        gitops.discard(root, ["a.py"], ["new.py"], branch, "main")
        assert (root / "a.py").read_text() == "x=1\n"
        assert not (root / "new.py").exists()
        assert (root / "secret.log").exists() and (root / "keep.py").exists()
        assert gitops.current_branch(root) == "main"
        assert subprocess.run(["git", "rev-parse", "--verify", "--quiet", branch], cwd=root).returncode != 0


def test_commit_without_identity_gives_actionable_error():
    with _project({"a.py": "x=1\n"}) as td:
        root = _git_repo(td)
        subprocess.run(["git", "config", "--unset", "user.name"], cwd=root, check=True)
        subprocess.run(["git", "config", "--unset", "user.email"], cwd=root, check=True)
        saved = {k: os.environ.get(k) for k in ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM", "HOME")}
        os.environ.update({"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "HOME": td})
        try:
            (root / "a.py").write_text("x=2\n")
            try:
                gitops.commit(root, ["a.py"], "m")
                raise AssertionError("expected GitError")
            except gitops.GitError as exc:
                assert "identity" in str(exc)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


# ------------------------------------------------------------------ CLI
def test_cli_fix_refuses_without_a_tty():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import bugfix_agent
    saved = sys.stdin
    sys.stdin = io.StringIO("")
    try:
        assert bugfix_agent.main(["fix", "--manual", "x"]) == 2
    finally:
        sys.stdin = saved
