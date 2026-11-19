"""`python -m vendor_verdict run ...`: see vendor_verdict.runner."""
import sys

from .runner import main

sys.exit(main())
