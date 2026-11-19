"""The Kaggle runner's plumbing (eval/run_kaggle.py, eval/kaggle_policy.py): prompt selection, estimates, submission writing.

Invented data only: no competition file is read and no LLM is called (a stub stands in for it)."""
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from assessor.schemas import ClaimList

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import kaggle_policy as kp  # noqa: E402
import run_kaggle as rk  # noqa: E402

PAGE = {"page_id": "p1", "url": "https://acme.example/s", "title": "t", "text": "x" * 80}


def _has_commit(rev: str) -> bool:
    return subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{rev}^{{commit}}"], capture_output=True).returncode == 0


class StubLLM:
    model = "stub-model"

    def __init__(self):
        self.calls = []

    def structured(self, system, user, schema, stage=""):
        self.calls.append(system)
        return ClaimList(claims=[])


def _repo_with_prompt(path: Path, text: str) -> Path:
    (path / "eval").mkdir(parents=True)
    (path / "eval" / "kaggle_policy.py").write_text(f"EXTRACT_SYSTEM = {text!r}\n", encoding="utf-8")
    git = ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], check=True)
    return path


# ---------------------------------------------------------------- Task 1: choose a prompt without editing it
def test_default_cache_key_is_unchanged():          # old entries must stay findable
    h = hashlib.sha256()
    for part in ("m", kp.EXTRACT_SYSTEM, "ClaimList", kp.user_prompt(PAGE)):
        h.update(part.encode())
        h.update(b"\x00")
    assert kp.cache_key("m", PAGE) == h.hexdigest()


def test_cache_key_depends_on_system_and_salt():
    a = kp.cache_key("m", PAGE, system="X")
    assert a != kp.cache_key("m", PAGE, system="Y") and a != kp.cache_key("m", PAGE, "s", system="X")
    assert kp.cache_key("m", PAGE, system=kp.EXTRACT_SYSTEM) == kp.cache_key("m", PAGE)


def test_prompt_text_from_git_checks_the_hash(tmp_path):
    repo = _repo_with_prompt(tmp_path, "hello")
    assert kp.prompt_text_from_git("HEAD", hashlib.sha256(b"hello").hexdigest()[:12], repo=repo) == "hello"
    with pytest.raises(ValueError):
        kp.prompt_text_from_git("HEAD", "000000000000", repo=repo)


def test_prompt_text_from_git_leaves_no_module_behind(tmp_path):
    repo = _repo_with_prompt(tmp_path, "hello")
    before = set(sys.modules)
    kp.prompt_text_from_git("HEAD", hashlib.sha256(b"hello").hexdigest()[:12], repo=repo)
    assert set(sys.modules) == before and sys.modules["kaggle_policy"] is kp


def test_prompt_text_from_git_refuses_an_option_as_a_revision(tmp_path):
    repo = _repo_with_prompt(tmp_path, "hello")
    with pytest.raises(ValueError):
        kp.prompt_text_from_git("--output=/tmp/x", "000000000000", repo=repo)


@pytest.mark.skipif(not _has_commit("e42203d"), reason="history not available")
def test_base_prompt_is_recoverable_byte_for_byte():
    assert hashlib.sha256(kp.prompt_text_from_git("e42203d", "9566ad4b6b95").encode()).hexdigest()[:12] == "9566ad4b6b95"


def test_extractor_uses_the_given_system_and_keys_the_cache_by_it(tmp_path):
    llm = StubLLM()
    rk.Extractor(llm, tmp_path, False, "", system="SYS-A").run(PAGE)
    assert llm.calls == ["SYS-A"]
    assert (tmp_path / f"{kp.cache_key('stub-model', PAGE, '', system='SYS-A')}.json").exists()


def test_extractor_default_system_is_the_current_prompt(tmp_path):
    llm = StubLLM()
    rk.Extractor(llm, tmp_path, False).run(PAGE)
    assert llm.calls == [kp.EXTRACT_SYSTEM]


def test_select_prompt():
    assert rk.select_prompt("", "") == (kp.EXTRACT_SYSTEM, kp.PROMPT_VERSION)
    with pytest.raises(SystemExit):
        rk.select_prompt("e42203d", "")


@pytest.mark.skipif(not _has_commit("e42203d"), reason="history not available")
def test_select_prompt_from_a_commit_returns_the_text_and_its_version():
    text, version = rk.select_prompt("e42203d", "9566ad4b6b95")
    assert version == "9566ad4b6b95" and hashlib.sha256(text.encode()).hexdigest()[:12] == version


@pytest.mark.skipif(not _has_commit("e42203d"), reason="history not available")
def test_select_prompt_exits_when_the_commit_does_not_hold_the_expected_prompt():
    with pytest.raises(SystemExit):
        rk.select_prompt("e42203d", "000000000000")


def test_expect_prompt_without_a_commit_must_match_the_current_prompt():
    assert rk.select_prompt("", kp.PROMPT_VERSION) == (kp.EXTRACT_SYSTEM, kp.PROMPT_VERSION)
    with pytest.raises(SystemExit):
        rk.select_prompt("", "000000000000")


# ---------------------------------------------------------------- Task 2: estimate a run before it starts
from assessor.tracing import estimate_tokens  # noqa: E402

PAGE2 = {"page_id": "p2", "url": "https://acme.example/t", "title": "u", "text": "y" * 200}


def test_estimate_counts_input_exactly_and_output_by_the_measured_ratio():
    tin, tout = rk.estimate_call_tokens([PAGE, PAGE2], system="S")
    assert tin == estimate_tokens("S" + kp.user_prompt(PAGE)) + estimate_tokens("S" + kp.user_prompt(PAGE2))
    assert tout == round(tin * rk.OUT_PER_IN)
    assert rk.OUT_PER_IN == pytest.approx(282_140 / 3_408_090, rel=1e-3)


def test_estimate_of_no_pages_is_zero():
    assert rk.estimate_call_tokens([], system="S") == (0, 0)


def test_uncached_pages_splits_by_the_cache_key_of_this_prompt_model_and_salt(tmp_path):
    (tmp_path / f"{kp.cache_key('m', PAGE, 's1', system='SYS')}.json").write_text("{}")
    assert rk.uncached_pages([PAGE, PAGE2], tmp_path, "m", "s1", "SYS") == ([PAGE], [PAGE2])
    assert rk.uncached_pages([PAGE, PAGE2], tmp_path, "m", "s2", "SYS") == ([], [PAGE, PAGE2])        # another salt: nothing cached
    assert rk.uncached_pages([PAGE, PAGE2], tmp_path, "m", "s1", "OTHER") == ([], [PAGE, PAGE2])      # another prompt: nothing cached
    assert not list(tmp_path.glob("*.tmp"))


def test_estimate_line_states_pages_tokens_and_the_time_range():
    line = rk.format_estimate(n_call=2304, n_cached=10, tokens_in=3_400_000, tokens_out=280_000)
    assert line == "pages to call 2304 (cached 10); tokens in 3,400,000 / out 280,000; time 42-96 min at the measured pace"


# ---------------------------------------------------------------- Task 8: write a submission with the runner's self-check
class StubMetric:
    def __init__(self):
        self.calls = []

    def components(self, solution, submission):
        self.calls.append((solution, submission))
        return {}


ROWS = [{"id": "c__R1"}, {"id": "c__OVERALL"}]


def test_write_submission_writes_ids_in_order_with_defaults(tmp_path):
    metric = StubMetric()
    problems = rk.write_submission(tmp_path / "s.csv", ROWS, {"c__R1": "PASS"}, metric)
    assert problems == [] and (tmp_path / "s.csv").read_text().splitlines() == ["id,verdict", "c__R1,PASS", "c__OVERALL,CONDITIONAL"]
    assert len(metric.calls) == 1                                   # the host's scorer was asked to accept the file


def test_write_submission_reports_an_invalid_label(tmp_path):
    assert rk.write_submission(tmp_path / "s.csv", ROWS, {"c__R1": "MAYBE"}, StubMetric())


def test_write_submission_reports_what_the_hosts_metric_rejects(tmp_path):
    class Bad(StubMetric):
        def components(self, *a):
            raise ValueError("nope")
    assert any("metric.py rejected" in p for p in rk.write_submission(tmp_path / "s.csv", ROWS, {}, Bad()))
