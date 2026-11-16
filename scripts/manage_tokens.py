"""Manage API tokens (assessor/auth.py). Only SHA-256 hashes are stored; a token is shown once, when created.

    python scripts/manage_tokens.py add --user sec.lead@yourco.example --groups employees,security --roles "Head of Information Security"
    python scripts/manage_tokens.py list
    python scripts/manage_tokens.py revoke --user sec.lead@yourco.example

The file is VRA_TOKENS_FILE (default: runs/tokens.json). Approver roles per tier are in AI-5.1:
T1 "Team lead" · T2 "Head of Information Security" · T3 "Head of Information Security" and "DPO".
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assessor.auth import APPROVERS, hash_token  # noqa: E402
from assessor.config import RUNS_DIR, SETTINGS  # noqa: E402
from assessor.db import audit  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", default=SETTINGS.tokens_file or str(RUNS_DIR / "tokens.json"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--user", required=True)
    a.add_argument("--groups", default="employees", help="comma-separated ACL groups: employees, security, procurement")
    a.add_argument("--roles", default="", help="comma-separated approver roles (AI-5.1)")
    sub.add_parser("list")
    r = sub.add_parser("revoke")
    r.add_argument("--user", required=True)
    args = ap.parse_args(argv)

    path = Path(args.file)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"tokens": []}
    if args.cmd == "list":
        for t in data["tokens"]:
            print(f"{t['user']:40} groups={','.join(t.get('groups', []))} roles={','.join(t.get('roles', [])) or '-'} sha256={t['sha256'][:12]}…")
        print(f"{len(data['tokens'])} token(s) in {path}")
        return 0
    if args.cmd == "revoke":
        before = len(data["tokens"])
        data["tokens"] = [t for t in data["tokens"] if t["user"] != args.user]
        n = before - len(data["tokens"])
        audit("token.revoked", os.environ.get("USER", "operator"), args.user, {"removed": n})
        print(f"Revoked {n} token(s) for {args.user}.")
    else:
        roles = [x.strip() for x in args.roles.split(",") if x.strip()]
        known = {r for rs in APPROVERS.values() for r in rs}
        for role in roles:
            if role not in known:
                print(f"Unknown approver role {role!r}. Known roles: {sorted(known)}", file=sys.stderr)
                return 2
        token = "vra_" + secrets.token_urlsafe(32)
        data["tokens"].append({"sha256": hash_token(token), "user": args.user,
                               "groups": [x.strip() for x in args.groups.split(",") if x.strip()], "roles": roles})
        audit("token.created", os.environ.get("USER", "operator"), args.user, {"groups": args.groups, "roles": roles})
        print(f"Token for {args.user} (shown once, store it in a password manager):\n\n  {token}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    print(f"Wrote {path}. Set VRA_TOKENS_FILE={path} (and VRA_ENV=prod) in .env.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
