"""gitops.py — one branch per fix; never destructive beyond files the agent itself touched."""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


def git(root: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {(r.stderr or r.stdout).strip()}")
    return r.stdout


def head(root: Path) -> str:
    return git(root, "rev-parse", "HEAD").strip()


def current_branch(root: Path) -> str:
    return git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()


def preconditions(root: Path) -> None:
    """Raise GitError with an actionable message unless it is safe to create a fix branch."""
    if subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root, capture_output=True).returncode != 0:
        raise GitError("Not a git repository. Run `git init` and make a first commit.")
    if subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=root, capture_output=True).returncode != 0:
        raise GitError("The repository has no commits yet. Make an initial commit first.")
    if current_branch(root) == "HEAD":
        raise GitError("HEAD is detached. Check out a branch first (e.g. `git checkout main`).")
    dirty = git(root, "status", "--porcelain").strip()
    if dirty:
        raise GitError("Working tree has uncommitted changes (commit or `git stash` them first):\n" + dirty[:600])


def create_branch(root: Path, issue_id: str) -> str:
    name = f"bugfix/{issue_id}"
    exists = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{name}"], cwd=root).returncode == 0
    if exists:
        from datetime import datetime
        name += "-" + datetime.now().strftime("%H%M%S")
    git(root, "checkout", "-b", name)
    return name


def diff(root: Path, paths: list[str], *, stat: bool = False) -> str:
    # include new (untracked) files in the diff via intent-to-add
    if paths:
        git(root, "add", "-N", "--", *paths, check=False)
    args = ["diff", "--stat" if stat else "--unified=3"]
    return git(root, *args, "--", *paths) if paths else ""


def identity_ok(root: Path) -> bool:
    return bool(git(root, "config", "user.name", check=False).strip() and git(root, "config", "user.email", check=False).strip())


def commit(root: Path, paths: list[str], message: str) -> str:
    if not identity_ok(root):
        raise GitError("git identity is not configured. Run: git config user.name \"Your Name\" && git config user.email you@example.com")
    git(root, "add", "--", *paths)
    git(root, "commit", "-m", message)
    return head(root)


def discard(root: Path, edited: list[str], created: list[str], branch: str, back_to: str) -> None:
    """Undo ONLY what the agent did: restore edited files, delete files it created, leave the branch."""
    if edited:
        git(root, "checkout", "--", *edited, check=False)
    for rel in created:
        try:
            (root / rel).unlink()
        except OSError:
            pass
    git(root, "reset", "-q", check=False)
    git(root, "checkout", back_to)
    git(root, "branch", "-D", branch, check=False)
