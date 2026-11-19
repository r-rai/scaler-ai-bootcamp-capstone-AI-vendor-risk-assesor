"""`python -m vendor_verdict run ...` and `python -m vendor_verdict selfcheck ...`: see vendor_verdict.runner."""
import sys

from .runner import main

sys.exit(main())
