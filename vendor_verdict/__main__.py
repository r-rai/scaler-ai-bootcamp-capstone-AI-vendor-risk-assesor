"""`python -m vendor_verdict run|selfcheck ...` (vendor_verdict.runner) and `python -m vendor_verdict gates ...` (vendor_verdict.gates)."""
import sys


def dispatch(argv: list[str]) -> int:
    if argv[:1] == ["gates"]:                           # its own parser: runner.build_parser() stays as it is
        from . import gates
        return gates.main(argv[1:])
    from . import runner
    return runner.main(argv)


if __name__ == "__main__":
    sys.exit(dispatch(sys.argv[1:]))
